---
name: cross-session-memory
description: Maintain project context across sessions and AI platforms
---

# Cross-Session Memory Skill

## Instructions for AI Agents

You are an AI coding assistant with the ability to maintain context across sessions using a file-based context system. This system is platform-agnostic and works across Cursor, Claude Code, Codex, and other AI coding tools.

### Context File Locations

All context files live in `docs/` (relative to the project root):

| File | Purpose |
|------|---------|
| `AGENT.md` | Instructions you read at every session start |
| `project.md` | Static project overview and architecture |
| `session.md` | Dynamic session state—**start here** |
| `memory.md` | Accumulated learnings and decisions |

### Session Start Protocol

When starting a new session:

1. **Read context files** in this order:
   - `docs/AGENT.md` (if exists)
   - `docs/project.md` (if exists)
   - `docs/session.md` (if exists)
   - `docs/memory.md` (if exists)

2. **Locate resume point**: Find `## RESUME HERE` section in `docs/session.md`

3. **Summarize state**: Tell the user:
   - What task was in progress
   - What was last completed
   - Any blockers or pending items

4. **Confirm before proceeding**: Ask the user if the summary is accurate

### During Session Protocol

Update `docs/session.md` after:

- Creating, modifying, or deleting files
- Completing significant tasks or milestones
- Encountering blockers or errors
- Making architectural decisions
- Changing direction or approach

### Session End Protocol

Before the session ends:

1. Update `## RESUME HERE` with:
   - Current status (in-progress, blocked, completed)
   - Last action attempted/completed
   - Next logical step
   - Any open questions or decisions needed

2. Update progress checkboxes if applicable

3. Archive session if switching contexts (optional)

### Initialization Protocol

When user requests context setup (no `docs/` context files exist):

1. Create `docs/` directory (if needed)
2. Create all four template files
3. Populate `project.md` with project information gathered from user
4. Initialize `session.md` with first session entry

---

## File Templates

### AGENT.md Template

```markdown
# Agent Instructions

This project uses cross-session context tracking. At the start of each session:

1. Read all files in `docs/` directory (focus on `AGENT.md`, `project.md`, `session.md`, `memory.md`)
2. Find `## RESUME HERE` in `session.md`
3. Summarize current state before proceeding
4. Update `session.md` after significant actions

## Project Conventions

<!-- Add project-specific conventions here -->

## Do Not

- Modify files without updating context
- Assume context from previous sessions
- Skip the resume summary step
```

### project.md Template

```markdown
# Project Overview

## Name
<!-- Project name -->

## Description
<!-- Brief description -->

## Goals
<!-- High-level objectives -->

## Architecture
<!-- System design, key components -->

## Tech Stack
<!-- Languages, frameworks, tools -->

## Key Files
<!-- Important files and their purposes -->

## Constraints
<!-- Limitations, requirements, boundaries -->
```

### session.md Template

```markdown
# Session Context

## RESUME HERE

**Status**: Not started
**Last Updated**: <!-- timestamp -->
**Platform**: <!-- cursor/claude-code/codex -->

### Current Task
<!-- What are we working on -->

### Last Action
<!-- What was just done -->

### Next Step
<!-- What should happen next -->

### Blockers
<!-- Any issues preventing progress -->

---

## Progress

### Completed
- [ ] <!-- Task 1 -->
- [ ] <!-- Task 2 -->

### In Progress
- [ ] <!-- Current task -->

### Pending
- [ ] <!-- Future task -->

---

## Session Log

### Session 1 - <!-- date -->
**Platform**: <!-- platform used -->
**Duration**: <!-- time spent -->

#### Accomplished
<!-- What was done -->

#### Decisions Made
<!-- Key decisions -->

#### Issues Encountered
<!-- Problems and solutions -->
```

### memory.md Template

```markdown
# Project Memory

Accumulated learnings and decisions that persist across sessions.

## Decisions

| Date | Decision | Rationale | Status |
|------|----------|-----------|--------|
<!-- Decisions log -->

## Learnings

### What Works
<!-- Successful patterns -->

### What Doesn't Work
<!-- Failed approaches to avoid -->

## Conventions Discovered

<!-- Project-specific patterns found during development -->

## External Dependencies

| Dependency | Purpose | Notes |
|------------|---------|-------|
<!-- External services, APIs, etc. -->

## Gotchas

<!-- Things to watch out for -->
```

---

## Usage Examples

### Initialize Context

User: "Set up context tracking for this project"

AI Response:
1. Create `docs/` directory (if needed)
2. Generate all four template files
3. Ask user about project details
4. Populate `project.md` with gathered information
5. Confirm setup complete

### Resume Session

User: "Let's continue where we left off"

AI Response:
1. Read `docs/session.md`
2. Find `## RESUME HERE` section
3. Summarize: "Last session you were working on [X]. You completed [Y] and were about to [Z]. There's a blocker regarding [W]. Is this accurate?"
4. Wait for confirmation
5. Proceed with next step

### End Session

User: "I need to stop for now"

AI Response:
1. Update `## RESUME HERE` with current state
2. Log session in Session Log section
3. Update any memory items if applicable
4. Confirm context saved
5. Suggest: "Context saved. When you resume, I'll pick up from [current state]."

### Transfer to Different Platform

User: "I'm switching to Cursor from Claude Code"

AI Response:
1. Ensure `docs/session.md` is fully updated
2. Add note: "Transferred from Claude Code on [date]"
3. Confirm all context files are saved
4. Explain: "Context is saved in `docs/`. When you open in Cursor, instruct it to read `docs/session.md` to resume."

---

