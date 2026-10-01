// llm_batch.mjs — batched GLM scorer runner for the engine_v2 cross-reranker.
//
// Usage:  bun run llm_batch.mjs <input.json> <output.json>
//
// input : {"items":[{"key","system","user"},...],
//          "concurrency":5, "timeout_ms":60000}
// output: {"results":{"<key>":{"ok":true,"content":"..."}
//                              | {"ok":false,"error":"..."}}}
//
// One SDK call per item, thinking disabled, per-call timeout enforced via
// Promise.race, worker-pool concurrency (default 5). Retry/backoff policy
// lives in the Python orchestrator (scripts/engine_v2/rerank.py).
import ZAI from 'z-ai-web-dev-sdk';
import { readFileSync, writeFileSync } from 'fs';

async function withTimeout(promise, ms, label) {
  let timer;
  const guard = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label}: timeout after ${ms}ms`)), ms);
  });
  try {
    return await Promise.race([promise, guard]);
  } finally {
    clearTimeout(timer);
  }
}

const [, , inPath, outPath] = process.argv;
if (!inPath || !outPath) {
  console.error('usage: bun run llm_batch.mjs <input.json> <output.json>');
  process.exit(2);
}

const cfg = JSON.parse(readFileSync(inPath, 'utf8'));
const items = Array.isArray(cfg.items) ? cfg.items : [];
const concurrency = Math.max(1, Math.min(8, cfg.concurrency || 5));
const timeoutMs = cfg.timeout_ms || 60000;
const results = {};

async function main() {
  const zai = await ZAI.create();
  let next = 0;

  async function worker() {
    while (next < items.length) {
      const i = next++;
      const it = items[i];
      try {
        const completion = await withTimeout(
          zai.chat.completions.create({
            messages: [
              { role: 'assistant', content: it.system },
              { role: 'user', content: it.user },
            ],
            thinking: { type: 'disabled' },
          }),
          timeoutMs,
          it.key,
        );
        const content = completion?.choices?.[0]?.message?.content ?? '';
        if (!content) throw new Error('empty completion');
        results[it.key] = { ok: true, content };
      } catch (e) {
        results[it.key] = { ok: false, error: String((e && e.message) || e) };
      }
    }
  }

  await Promise.all(
    Array.from({ length: Math.min(concurrency, items.length) }, worker),
  );
  writeFileSync(outPath, JSON.stringify({ results }, null, 1));
  const okCount = Object.values(results).filter((r) => r.ok).length;
  console.error(`llm_batch: ${items.length} items, ${okCount} ok`);
}

main().catch((e) => {
  console.error('llm_batch fatal:', e);
  process.exit(1);
});
