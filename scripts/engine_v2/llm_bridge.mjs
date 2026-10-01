// llm_bridge.mjs — stdin/stdout JSON bridge to the z-ai GLM SDK.
//
// Protocol:
//   stdin : {"system": "<system prompt>", "user": "<user prompt>",
//            "thinking": "enabled"|"disabled"}
//   stdout: {"ok": true, "content": "<completion>"}
//         | {"ok": false, "error": "<message>"}
//
// Quirk of this SDK (verified): the system prompt is passed as a message with
// role 'assistant'. Retries/timeouts/caching are handled on the Python side.
import ZAI from 'z-ai-web-dev-sdk';

async function main() {
  let raw = '';
  process.stdin.setEncoding('utf8');
  for await (const chunk of process.stdin) raw += chunk;
  let req;
  try {
    req = JSON.parse(raw);
  } catch (e) {
    console.log(JSON.stringify({ ok: false, error: 'bad request json: ' + e.message }));
    return;
  }
  try {
    const zai = await ZAI.create();
    const messages = [
      { role: 'assistant', content: req.system || '' },
      { role: 'user', content: req.user || '' }
    ];
    const thinking = req.thinking === 'enabled'
      ? { type: 'enabled' }
      : { type: 'disabled' };
    const c = await zai.chat.completions.create({ messages, thinking });
    const content = c?.choices?.[0]?.message?.content ?? '';
    console.log(JSON.stringify({ ok: true, content }));
  } catch (e) {
    console.log(JSON.stringify({ ok: false, error: String((e && e.message) || e) }));
  }
}

main();
