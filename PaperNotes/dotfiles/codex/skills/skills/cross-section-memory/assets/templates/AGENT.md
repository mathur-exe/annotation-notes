# Agent Instructions

This project uses **cross-session context tracking**. Follow these protocols at every session.

## Session Start

1. Read all files in `docs/` directory
2. Find `## RESUME HERE` in `session.md`
3. Summarize current state to the user before proceeding
4. Wait for user confirmation before continuing work

## During Session

Update `docs/session.md` after:
- Creating, modifying, or deleting files
- Completing significant tasks
- Encountering blockers or errors
- Making architectural decisions

## Session End

Before ending, update `session.md` with:
- Current status in `## RESUME HERE`
- Last action completed
- Next logical step
- Any blockers or open questions

---

## Project Conventions

<!-- Add project-specific coding conventions here -->
<!-- Example:
- Use TypeScript strict mode
- Follow PEP 8 for Python
- Write tests for all new functions
-->

## Key Commands

<!-- Add frequently used commands here -->
<!-- Example:
- Build: `npm run build`
- Test: `pytest -v`
- Deploy: `./deploy.sh`
-->

## Do Not

- Modify files without updating session context
- Assume knowledge from previous sessions without reading context
- Skip the resume summary step
- Make architectural changes without updating `project.md`
