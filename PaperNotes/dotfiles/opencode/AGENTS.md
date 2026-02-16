# OpenCode Configuration

## Config File Location

The main OpenCode configuration file is located at:

**`~/.config/opencode/opencode.json`**

This path is used across all platforms (macOS, Linux, Windows).

## Key Configuration Details

- **Schema**: Uses `https://opencode.ai/config.json`
- **Plugin**: `opencode-antigravity-auth@latest` for Antigravity/Google OAuth authentication
- **Provider**: Google (via Antigravity) with Claude models

## Available Claude Models

| Model ID | Reasoning | Description |
|----------|-----------|-------------|
| `antigravity-claude-sonnet-4-5` | false | Claude Sonnet 4.5 (non-thinking) |
| `antigravity-claude-sonnet-4-5-thinking` | true | Claude Sonnet 4.5 with thinking |
| `antigravity-claude-opus-4-5-thinking` | true | Claude Opus 4.5 with thinking |

## Important Notes

- Use `antigravity-` prefix (not `gemini-`) for model names
- The `reasoning` field indicates whether the model supports extended thinking
- For issues with Claude models, check GitHub issues: https://github.com/NoeFabris/opencode-antigravity-auth/issues

## Related Files

- `~/.config/opencode/antigravity.json` - Plugin-specific configuration
- `~/.config/opencode/antigravity-accounts.json` - Account credentials
- `~/.local/state/opencode/model.json` - Recently used models
- `~/.cache/opencode/models.json` - Cached model registry
