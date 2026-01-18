# Project Memory

Accumulated learnings and decisions that persist across all sessions. This file grows slowly and captures institutional knowledge.

---

## Decisions Log

| Date | Decision | Rationale | Status |
|------|----------|-----------|--------|
| <!-- YYYY-MM-DD --> | <!-- What was decided --> | <!-- Why --> | <!-- Active/Revised/Deprecated --> |

---

## Learnings

### What Works

<!-- Successful patterns discovered during development -->
<!-- Example:
- Using dependency injection makes testing much easier
- Batch API calls to reduce latency
-->

### What Doesn't Work

<!-- Failed approaches to avoid repeating -->
<!-- Example:
- Recursive approach hit stack limits; switched to iterative
- Library X has memory leak; use Library Y instead
-->

---

## Conventions Discovered

<!-- Project-specific patterns found during development -->
<!-- Example:
- All API responses use snake_case
- Error codes follow HTTP status semantics
- Config files use YAML, not JSON
-->

---

## External Dependencies

| Dependency | Purpose | Version | Notes |
|------------|---------|---------|-------|
| <!-- name --> | <!-- why needed --> | <!-- version --> | <!-- gotchas --> |

---

## Gotchas

<!-- Things to watch out for that aren't obvious -->
<!-- Example:
- The staging DB has a 5-second connection timeout
- CI fails silently if GITHUB_TOKEN is missing
- Must run migrations before seeding
-->

---

## Useful Commands

<!-- Commands discovered during development that are frequently needed -->

```bash
# Example: Reset database
# dropdb myapp && createdb myapp && python manage.py migrate
```

---

## Links & References

| Resource | URL | Notes |
|----------|-----|-------|
| <!-- name --> | <!-- url --> | <!-- why useful --> |
