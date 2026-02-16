# How To "extact" Get `/context` Working In OpenCode

This guide gives the exact setup to make `/context` available and usable.

## 1) Put command + plugin files in the correct global location

OpenCode should load from:

- `~/.config/opencode/command/`
- `~/.config/opencode/plugins/`

Do **not** keep these under `~/.config/opencode/.opencode/` unless you also set `OPENCODE_CONFIG_DIR`.

## 2) Required files

You should have:

- `~/.config/opencode/command/context.md`
- `~/.config/opencode/plugins/context-usage.ts`
- `~/.config/opencode/plugins/tokenizer-registry.mjs`
- `~/.config/opencode/plugins/tokenizer-aliases.json`
- `~/.config/opencode/plugins/vendor/node_modules/...`

Keep `tokenizer-registry.d.ts` out of the plugin root (for example under `~/.config/opencode/plugins/types/`), so OpenCode does not try to load it as a runtime plugin.

## 3) Install vendor dependencies

```bash
npm install js-tiktoken@latest @huggingface/transformers@^3.3.3 --prefix ~/.config/opencode/plugins/vendor
```

## 4) Validate plugin + command registration

```bash
opencode debug config --print-logs --log-level DEBUG
```

Expected signals:

- plugin load includes `file:///Users/gaurangmathur/.config/opencode/plugins/context-usage.ts`
- command list includes `context`

## 5) Runtime behavior and model caveat

- `/context` calls the `context_usage` tool.
- On some routed/free models (for example `kimi-k2.5-free`), tokenizer resolution may fail or provider limits may return `429`.
- If that happens, retry with a supported model like `openai/gpt-5.2` or your configured Anthropic model.

## 6) Quick test commands

```bash
opencode run --print-logs --log-level DEBUG "/context"
opencode run -m openai/gpt-5.2 --print-logs --log-level DEBUG "/context"
```

If the second command works and the first fails, setup is correct and the remaining issue is model/provider-specific, not plugin installation.
