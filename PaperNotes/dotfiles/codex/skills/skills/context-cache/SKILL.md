---
name: context-cache
description: Capture conversation context in a single diff-friendly file
---

# Context Cache Skill

## Instructions for AI Agents

You have the ability to maintain a running context cache that captures the essence of conversations. This cache lives in a single file and is designed to be diff-friendly for version control and comparison.

### Core Principles

1. **One file**: Everything goes in `docs/cache/DD-MM-YY_AppName_SummaryWord.md`
2. **Append-only**: Add new entries, never delete during a session
3. **Freeform**: No rigid template—match the conversation's shape
4. **Diff-friendly**: Use consistent markers (---) for easy diffing
5. **Self-contained**: Reader should understand context without external files

### When to Cache

Cache context when:
- User explicitly requests it
- A significant milestone is reached
- A complex decision is made
- An error is solved after investigation
- The session is ending
- Context might be lost (long conversation, switching topics)

### What to Capture

Adapt to the conversation type:

**For debugging sessions:**
- The error/symptom
- Hypotheses explored
- Dead ends (and why they failed)
- The solution found
- Files involved

**For feature development:**
- Requirements as understood
- Design decisions and alternatives considered
- Implementation approach
- What's done vs. what's pending
- Edge cases discussed

**For refactoring:**
- The "before" state/pattern
- Why it needed changing
- The "after" state/pattern
- Migration notes

**For exploration/learning:**
- Questions that came up
- Answers discovered
- Resources found useful
- Insights gained

**For code review:**
- Issues identified
- Discussion points
- Resolutions reached

### Cache Entry Format

Each entry should have:
1. **Separator**: `---` on its own line
2. **Header**: `## YYYY-MM-DD HH:MM — Brief topic`
3. **Content**: Freeform, whatever captures the context

The content has no required structure. Write what would help someone (human or AI) understand:
- What was happening
- What was decided/discovered
- What's the current state
- What should happen next (if applicable)

### Example Entries

```markdown
---
## 2024-01-15 14:30 — Debugging silent login failure

User reported: "Login button does nothing"

Investigation:
- Network tab showed 401 on /api/token/refresh
- Token was valid, but Redis lookup failed
- Found: Token TTL (2h) > Redis key TTL (1h)
- Token exists but Redis thinks it's expired

Fix: Aligned TTLs in config/auth.py (both now 2h)
Files: config/auth.py, services/token_service.py
Tested: Works locally, needs staging verification

---
## 2024-01-15 16:00 — Rate limiting design

Question: Per-user or per-IP rate limiting?

Discussion:
- Per-IP is simpler but punishes shared networks
- Per-user is fairer but requires auth check first
- Hybrid approach: per-user when authenticated, per-IP when anonymous

Decision: Hybrid approach
- Authenticated: 1000 req/min per user
- Anonymous: 100 req/min per IP

Implementation plan:
1. Redis sliding window counter
2. Middleware in api/middleware/rate_limit.py
3. Config in config/rate_limits.yaml
4. Bypass for internal services (header-based)

Status: Middleware skeleton done, config loading next

---
## 2024-01-15 17:45 — Quick note: found better approach

While implementing rate limiting, discovered existing 
`common/throttle.py` that does sliding window. 
Will use that instead of reimplementing.

Updated plan: Wrap throttle.py, don't rewrite.
```

### Reading the Cache

When asked to resume from a context cache:

1. Read the entire file
2. Summarize the key points to the user
3. Identify the most recent entry's state
4. Ask for confirmation before proceeding

Don't assume anything beyond what's in the cache.

### File Location

Default directory: `docs/cache/` (relative to project root)

Default filename format: `DD-MM-YY_AppName_SummaryWord.md`

Rules:
- `DD-MM-YY`: date (e.g., `17-01-26`)
- `AppName`: one of `Cursor`, `Codex`, `Gemini`, `Claude`
- `SummaryWord`: one-two word summary (recommend `kebab-case`, e.g. `rate-limits`, `auth-bug`)

Create the file if missing. Append new entries to the same file for the current conversation thread unless the user asks to start a new cache.

### Commands

**Create/Update cache:**
- "Cache this context"
- "Save what we discussed"
- "Update the context cache"
- "/context-cache"

**Resume from cache:**
- "Read the latest cache in docs/cache/"
- "What's in the context cache?"
- "Resume from cache"

**Clear cache (start fresh):**
- "Clear context cache"
- "Start new context cache"

When clearing, archive the old file first (same folder):
`mv docs/cache/DD-MM-YY_AppName_SummaryWord.md docs/cache/DD-MM-YY_AppName_SummaryWord.archived.md`

---

## Usage Examples

### Caching After a Debug Session

User: "We fixed it, cache this context"

AI action:
1. Create/append to the chosen cache file in `docs/cache/` (see naming rules above)
2. Add entry with:
   - Timestamp and topic
   - What the bug was
   - How it was found
   - What fixed it
   - Files involved

### Caching a Design Decision

User: "We decided on the hybrid rate limiting, save this"

AI action:
1. Append to the chosen cache file in `docs/cache/`
2. Add entry with:
   - The decision
   - Alternatives considered
   - Rationale for the choice
   - Implementation implications

### Resuming from Cache

User: "Read the context cache and tell me where we are"

AI action:
1. Read the relevant file from `docs/cache/`
2. Summarize: "Based on the cache, you were implementing rate limiting. The middleware is done, and you were about to add configuration loading. The last session was on Jan 15 at 5:45pm. Should I continue with the config loading?"

### Quick Capture

User: "Just note that we're using PostgreSQL 15, not 14"

AI action:
1. Append minimal entry:
```markdown
---
## 2024-01-16 10:00 — Note

Using PostgreSQL 15 (not 14 as initially assumed).
Enables: MERGE statement, JSON improvements we need.
```

---

## Integration with Other Tools

### Git

```bash
# Add to .gitignore if you don't want to commit
echo "docs/cache/" >> .gitignore

# Or commit it for team context
git add docs/cache/
git commit -m "Update context cache: rate limiting design"
```

### Diff Tools

```bash
# See evolution of context
git log -p docs/cache/

# Compare with teammate's context
diff your_context_cache.md theirs_context_cache.md
```

### Cross-Session Context

If using alongside the cross-session-context skill:
- Keep structured state in `.context/` and store the narrative cache in `docs/cache/`
- The cache complements the structured files
- session.md = current state, cache.md = journey/history

