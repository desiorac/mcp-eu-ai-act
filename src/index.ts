import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import * as fs from "fs";
import * as path from "path";

/**
 * EU AI Act Compliance Checker - Pure JS MCP Server
 * Scans projects to detect AI model usage and verify EU AI Act compliance
 */

const AI_MODEL_PATTERNS: Record<string, RegExp[]> = {
  openai: [
    /openai\.ChatCompletion/i,
    /openai\.Completion/i,
    /from openai import/i,
    /import openai/i,
    /gpt-3\.5/i,
    /gpt-4/i,
    /text-davinci/i,
  ],
  anthropic: [
    /from anthropic import/i,
    /import anthropic/i,
    /claude-/i,
    /Anthropic\(\)/i,
    /messages\.create/i,
  ],
  huggingface: [
    /from transformers import/i,
    /AutoModel/i,
    /AutoTokenizer/i,
    /pipeline\(/i,
    /huggingface_hub/i,
  ],
  tensorflow: [
    /import tensorflow/i,
    /from tensorflow import/i,
    /tf\.keras/i,
  ],
  pytorch: [
    /import torch/i,
    /from torch import/i,
    /nn\.Module/i,
  ],
  langchain: [
    /from langchain import/i,
    /import langchain/i,
    /LLMChain/i,
    /ChatOpenAI/i,
  ],
};

const RISK_CATEGORIES: Record<string, { description: string; requirements: string[] }> = {
  unacceptable: {
    description: "Prohibited systems (behavioral manipulation, social scoring, mass biometric surveillance)",
    requirements: ["Prohibited system - Do not deploy"],
  },
  high: {
    description: "High-risk systems (recruitment, credit scoring, law enforcement)",
    requirements: [
      "Complete technical documentation",
      "Risk management system",
      "Data quality and governance",
      "Transparency and user information",
      "Human oversight",
      "Robustness, accuracy and cybersecurity",
      "Quality management system",
      "Registration in EU database",
    ],
  },
  limited: {
    description: "Limited-risk systems (chatbots, deepfakes)",
    requirements: [
      "Transparency obligations",
      "Clear user information about AI interaction",
      "AI-generated content marking",
    ],
  },
  minimal: {
    description: "Minimal-risk systems (spam filters, video games)",
    requirements: [
      "No specific obligations",
      "Voluntary code of conduct encouraged",
    ],
  },
};

const BLOCKED_PATHS = [
  "/opt/claude-ceo", "/etc", "/root", "/home", "/var", "/proc",
  "/sys", "/dev", "/run", "/boot", "/usr", "/bin", "/sbin",
  "/lib", "/snap", "/mnt", "/media",
];

const MAX_FILES = 5000;
const MAX_FILE_SIZE = 1_000_000;
const CODE_EXTENSIONS = new Set([".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c"]);

function validatePath(projectPath: string): { safe: boolean; error: string } {
  let resolved: string;
  try {
    resolved = fs.realpathSync(projectPath);
  } catch {
    return { safe: false, error: `Invalid path: ${projectPath}` };
  }
  for (const blocked of BLOCKED_PATHS) {
    if (resolved === blocked || resolved.startsWith(blocked + "/")) {
      return { safe: false, error: `Access denied: scanning ${blocked} is not allowed for security reasons` };
    }
  }
  return { safe: true, error: "" };
}

function walkDir(dir: string, maxFiles: number): string[] {
  const results: string[] = [];
  const queue = [dir];
  while (queue.length > 0 && results.length < maxFiles) {
    const current = queue.shift()!;
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(current, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      if (results.length >= maxFiles) break;
      const fullPath = path.join(current, entry.name);
      if (entry.name.startsWith(".") || entry.name === "node_modules" || entry.name === "__pycache__") continue;
      if (entry.isDirectory()) {
        queue.push(fullPath);
      } else if (entry.isFile() && CODE_EXTENSIONS.has(path.extname(entry.name))) {
        try {
          const stat = fs.statSync(fullPath);
          if (stat.size <= MAX_FILE_SIZE) results.push(fullPath);
        } catch { /* skip */ }
      }
    }
  }
  return results;
}

function scanProject(projectPath: string) {
  const { safe, error } = validatePath(projectPath);
  if (!safe) return { error, detected_models: {} };
  if (!fs.existsSync(projectPath)) return { error: `Project path does not exist: ${projectPath}`, detected_models: {} };

  const files = walkDir(projectPath, MAX_FILES);
  const detectedModels: Record<string, string[]> = {};
  const aiFiles: { file: string; frameworks: string[] }[] = [];

  for (const filePath of files) {
    let content: string;
    try {
      content = fs.readFileSync(filePath, "utf-8");
    } catch { continue; }

    const fileDetections: string[] = [];
    for (const [framework, patterns] of Object.entries(AI_MODEL_PATTERNS)) {
      for (const pattern of patterns) {
        if (pattern.test(content)) {
          fileDetections.push(framework);
          if (!detectedModels[framework]) detectedModels[framework] = [];
          detectedModels[framework].push(path.relative(projectPath, filePath));
          break;
        }
      }
    }
    if (fileDetections.length > 0) {
      aiFiles.push({ file: path.relative(projectPath, filePath), frameworks: [...new Set(fileDetections)] });
    }
  }

  return { files_scanned: files.length, ai_files: aiFiles, detected_models: detectedModels };
}

function fileExists(projectPath: string, filename: string): boolean {
  return fs.existsSync(path.join(projectPath, filename)) || fs.existsSync(path.join(projectPath, "docs", filename));
}

function checkCompliance(projectPath: string, riskCategory: string) {
  if (!RISK_CATEGORIES[riskCategory]) {
    return { error: `Invalid risk category: ${riskCategory}. Valid: ${Object.keys(RISK_CATEGORIES).join(", ")}` };
  }

  const info = RISK_CATEGORIES[riskCategory];
  const readmeExists = fs.existsSync(path.join(projectPath, "README.md"));
  let status: Record<string, boolean> = {};

  if (riskCategory === "high") {
    status = {
      technical_documentation: ["README.md", "ARCHITECTURE.md", "API.md", "docs"].some(d => fs.existsSync(path.join(projectPath, d))),
      risk_management: fileExists(projectPath, "RISK_MANAGEMENT.md"),
      transparency: fileExists(projectPath, "TRANSPARENCY.md") || readmeExists,
      data_governance: fileExists(projectPath, "DATA_GOVERNANCE.md"),
      human_oversight: fileExists(projectPath, "HUMAN_OVERSIGHT.md"),
      robustness: fileExists(projectPath, "ROBUSTNESS.md"),
    };
  } else if (riskCategory === "limited") {
    const readmeLower = readmeExists ? fs.readFileSync(path.join(projectPath, "README.md"), "utf-8").toLowerCase() : "";
    const aiKeywords = ["ai", "artificial intelligence", "machine learning", "deep learning", "gpt", "claude", "llm"];
    status = {
      transparency: readmeExists || fileExists(projectPath, "TRANSPARENCY.md"),
      user_disclosure: aiKeywords.some(kw => readmeLower.includes(kw)),
      content_marking: false, // checked below
    };
    // Check content marking in .py files
    try {
      const pyFiles = walkDir(projectPath, 500).filter(f => f.endsWith(".py"));
      const markers = ["generated by ai", "généré par ia", "ai-generated", "machine-generated"];
      for (const f of pyFiles) {
        try {
          const c = fs.readFileSync(f, "utf-8").toLowerCase();
          if (markers.some(m => c.includes(m))) { status.content_marking = true; break; }
        } catch { /* skip */ }
      }
    } catch { /* skip */ }
  } else if (riskCategory === "minimal") {
    status = { basic_documentation: readmeExists };
  }

  const total = Object.keys(status).length;
  const passed = Object.values(status).filter(Boolean).length;

  return {
    risk_category: riskCategory,
    description: info.description,
    requirements: info.requirements,
    compliance_status: status,
    compliance_score: `${passed}/${total}`,
    compliance_percentage: total > 0 ? Math.round((passed / total) * 1000) / 10 : 0,
  };
}

function generateReport(projectPath: string, riskCategory: string) {
  const scan = scanProject(projectPath);
  if ("error" in scan && scan.error) return { error: scan.error };
  const compliance = checkCompliance(projectPath, riskCategory);
  if ("error" in compliance && compliance.error) return { error: compliance.error };

  const recommendations: string[] = [];
  const compStatus = (compliance as any).compliance_status || {};
  for (const [check, passed] of Object.entries(compStatus)) {
    if (!passed) recommendations.push(`MISSING: Create documentation for: ${check.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}`);
  }
  if (recommendations.length === 0) recommendations.push("All basic checks passed");
  if (riskCategory === "high") recommendations.push("WARNING: High-risk system - EU database registration required before deployment");
  else if (riskCategory === "limited") recommendations.push("INFO: Limited-risk system - Ensure full transparency compliance");

  return {
    report_date: new Date().toISOString(),
    project_path: projectPath,
    scan_summary: {
      files_scanned: (scan as any).files_scanned || 0,
      ai_files_detected: ((scan as any).ai_files || []).length,
      frameworks_detected: Object.keys((scan as any).detected_models || {}),
    },
    compliance_summary: {
      risk_category: riskCategory,
      compliance_score: (compliance as any).compliance_score || "0/0",
      compliance_percentage: (compliance as any).compliance_percentage || 0,
    },
    detailed_findings: {
      detected_models: (scan as any).detected_models || {},
      compliance_checks: compStatus,
      requirements: (compliance as any).requirements || [],
    },
    recommendations,
  };
}

// Factory function for creating fresh server instances
function createServer() {
  const srv = new McpServer({
    name: "EU AI Act Compliance Checker",
    version: "1.1.0",
  });

  srv.tool(
    "scan_project",
    "Scan a project to detect AI model usage (OpenAI, Anthropic, HuggingFace, TensorFlow, PyTorch, LangChain)",
    { project_path: z.string().describe("Absolute path to the project to scan") },
    async ({ project_path }) => ({
      content: [{ type: "text" as const, text: JSON.stringify(scanProject(project_path), null, 2) }],
    })
  );

  srv.tool(
    "check_compliance",
    "Check EU AI Act compliance for a given risk category",
    {
      project_path: z.string().describe("Absolute path to the project"),
      risk_category: z.enum(["unacceptable", "high", "limited", "minimal"]).default("limited").describe("EU AI Act risk category"),
    },
    async ({ project_path, risk_category }) => ({
      content: [{ type: "text" as const, text: JSON.stringify(checkCompliance(project_path, risk_category), null, 2) }],
    })
  );

  srv.tool(
    "generate_report",
    "Generate a complete EU AI Act compliance report with scan results, compliance checks, and recommendations",
    {
      project_path: z.string().describe("Absolute path to the project"),
      risk_category: z.enum(["unacceptable", "high", "limited", "minimal"]).default("limited").describe("EU AI Act risk category"),
    },
    async ({ project_path, risk_category }) => ({
      content: [{ type: "text" as const, text: JSON.stringify(generateReport(project_path, risk_category), null, 2) }],
    })
  );

  return srv;
}

// Sandbox server for Smithery scanning (returns unconnected server)
export function createSandboxServer() {
  return createServer();
}

// Default export for Smithery
export default createServer;

// Start server when run directly (only if not imported)
const isMainModule = typeof require !== "undefined" && require.main === module;
if (isMainModule) {
  (async () => {
    const srv = createServer();
    const transport = new StdioServerTransport();
    await srv.connect(transport);
  })().catch(console.error);
}
