import { createReadStream, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, resolve } from "node:path";

const arguments_ = process.argv.slice(2);
const value = (flag) => {
  const index = arguments_.indexOf(flag);
  if (index < 0 || index + 1 >= arguments_.length) {
    throw new Error(`Missing required ${flag}`);
  }
  return arguments_[index + 1];
};
const model = resolve(value("--model"));
const tokenizer = resolve(value("--tokenizer"));
const referenceIndex = arguments_.indexOf("--reference");
const reference = referenceIndex < 0 ? undefined : resolve(arguments_[referenceIndex + 1]);
const portIndex = arguments_.indexOf("--port");
const port = portIndex < 0 ? 4173 : Number(arguments_[portIndex + 1]);
const root = resolve(import.meta.dirname, "..");
let requestCount = 0;
const requestCounts = new Map();

const types = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".map", "application/json"],
  [".ledaweb", "application/octet-stream"],
  [".spartokn", "application/octet-stream"],
]);

const server = createServer((request, response) => {
  try {
    const path = new URL(request.url ?? "/", `http://${request.headers.host}`).pathname;
    if (path === "/request-count") {
      response.writeHead(200, { "content-type": "application/json", "cache-control": "no-store" });
      response.end(JSON.stringify({ requestCount, paths: Object.fromEntries(requestCounts) }));
      return;
    }
    ++requestCount;
    requestCounts.set(path, (requestCounts.get(path) ?? 0) + 1);
    const file = path === "/model.ledaweb" ? model
      : path === "/tokenizer.spartokn" ? tokenizer
      : path === "/reference.ledaref" && reference !== undefined ? reference
      : path === "/" || path === "/dev/" ? resolve(root, "dev/index.html")
      : resolve(root, `.${path}`);
    if (file !== model && file !== tokenizer && file !== reference && !file.startsWith(`${root}/`)) {
      response.writeHead(403).end("forbidden");
      return;
    }
    const size = statSync(file).size;
    response.writeHead(200, {
      "content-type": types.get(extname(file)) ?? "application/octet-stream",
      "content-length": size,
      "cache-control": "no-store",
    });
    const stream = createReadStream(file);
    stream.on("error", (error) => {
      if (!response.headersSent) {
        response.writeHead(500);
      }
      response.end(error.message);
    });
    stream.pipe(response);
  } catch (error) {
    response.writeHead(404).end(error instanceof Error ? error.message : "not found");
  }
});

server.listen(port, "127.0.0.1", () => {
  console.log(`Leda WebGPU diagnostic harness: http://127.0.0.1:${port}/dev/`);
});
