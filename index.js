#!/usr/bin/env node
// Smithery stdio wrapper - embeds and spawns the Python MCP server
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const os = require("os");

// Read server.py at build time via esbuild's text loader workaround:
// We write it to a temp file at runtime
const serverCode = fs.readFileSync(path.join(__dirname, "server.py"), "utf-8");

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "mcp-eu-ai-act-"));
const tmpServer = path.join(tmpDir, "server.py");
fs.writeFileSync(tmpServer, serverCode);

const server = spawn("python3", [tmpServer], {
  stdio: "inherit",
});

function cleanup() {
  try { fs.unlinkSync(tmpServer); } catch {}
  try { fs.rmdirSync(tmpDir); } catch {}
}

server.on("error", (err) => {
  console.error("Failed to start Python MCP server:", err.message);
  cleanup();
  process.exit(1);
});

server.on("close", (code) => {
  cleanup();
  process.exit(code || 0);
});

process.on("SIGTERM", () => { server.kill(); });
process.on("SIGINT", () => { server.kill(); });
