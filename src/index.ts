import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { spawn } from "child_process";
import { z } from "zod";

/**
 * EU AI Act Compliance Checker - Smithery MCP wrapper
 * Delegates to Python server for actual compliance checking
 */

const server = new McpServer({
  name: "EU AI Act Compliance Checker",
  version: "1.1.0",
});

function runPython(toolName: string, args: Record<string, unknown>): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn("python3", ["-c", `
import json, sys
sys.path.insert(0, '.')
from server import EUAIActChecker
checker = EUAIActChecker(${JSON.stringify(String(args.project_path || "."))})
args = json.loads(${JSON.stringify(JSON.stringify(args))})
tool = ${JSON.stringify(toolName)}
if tool == "scan_project":
    result = checker.scan_project()
elif tool == "check_compliance":
    checker.scan_project()
    result = checker.check_compliance(args.get("risk_category", "limited"))
elif tool == "generate_report":
    scan = checker.scan_project()
    compliance = checker.check_compliance(args.get("risk_category", "limited"))
    result = checker.generate_report(scan, compliance)
else:
    result = {"error": f"Unknown tool: {tool}"}
print(json.dumps(result))
`], { cwd: __dirname + "/.." });

    let stdout = "";
    let stderr = "";
    child.stdout?.on("data", (d) => (stdout += d));
    child.stderr?.on("data", (d) => (stderr += d));
    child.on("close", (code) => {
      if (code === 0) resolve(stdout.trim());
      else reject(new Error(stderr || `Exit code ${code}`));
    });
  });
}

server.tool(
  "scan_project",
  "Scan a project to detect AI model usage (OpenAI, Anthropic, HuggingFace, TensorFlow, PyTorch, LangChain)",
  { project_path: z.string().describe("Absolute path to the project to scan") },
  async ({ project_path }) => ({
    content: [{ type: "text", text: await runPython("scan_project", { project_path }) }],
  })
);

server.tool(
  "check_compliance",
  "Check EU AI Act compliance for a given risk category",
  {
    project_path: z.string().describe("Absolute path to the project"),
    risk_category: z.enum(["unacceptable", "high", "limited", "minimal"]).default("limited").describe("EU AI Act risk category"),
  },
  async ({ project_path, risk_category }) => ({
    content: [{ type: "text", text: await runPython("check_compliance", { project_path, risk_category }) }],
  })
);

server.tool(
  "generate_report",
  "Generate a complete EU AI Act compliance report with scan results, compliance checks, and recommendations",
  {
    project_path: z.string().describe("Absolute path to the project"),
    risk_category: z.enum(["unacceptable", "high", "limited", "minimal"]).default("limited").describe("EU AI Act risk category"),
  },
  async ({ project_path, risk_category }) => ({
    content: [{ type: "text", text: await runPython("generate_report", { project_path, risk_category }) }],
  })
);

// Sandbox server for Smithery scanning
export function createSandboxServer() {
  return server;
}

// Default export for Smithery
export default server;
