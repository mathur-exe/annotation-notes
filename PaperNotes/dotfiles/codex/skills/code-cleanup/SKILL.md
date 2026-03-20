---
name: code-cleanup
description: Simplify newly added or recently touched code and make it skimmable. Use when Codex is asked to clean up a branch, reduce complexity, shrink the number of states, remove unnecessary edits, replace defensive fallbacks with asserts, use discriminated unions for variant data, or make code easier to understand at a glance without changing behavior.
---

# Code Cleanup

## Goal

Make the touched code obvious on first read. Prefer the smallest diff that removes complexity, narrows state, and leaves only code that is strictly required.

## Cleanup Workflow

1. Define the exact cleanup surface.
   - Limit edits to code already being added or directly required by that change.
   - Remove unrelated edits and speculative abstractions.
2. Reduce the number of states.
   - Remove optionality that is not truly optional.
   - Reduce argument count.
   - Prefer one obvious shape over configurable parameter bags or override objects.
   - Use discriminated unions when a value can legitimately be one of several variants.
3. Make control flow easy to scan.
   - Prefer straight-line code and early returns.
   - Keep helpers to a minimum; do not split tiny logic into many functions.
   - Remove cleverness, deep nesting, and incidental indirection.
4. Fail loudly on impossible cases.
   - Exhaustively handle discriminated unions and other multi-shape objects.
   - Assert on unknown variants.
   - Assert when loading data that must exist.
5. Delete anything that is not earning its keep.
   - Remove dead branches, unused parameters, and unnecessary defaults.
   - Bias toward fewer lines of code.
   - Keep only behavior that is required for the current change.

## Rules

- Write extremely simple code. It should be skimmable and understandable at a glance.
- Minimize possible states by narrowing data and reducing arguments.
- Use discriminated unions to model legitimate variants.
- Exhaustively handle objects with multiple types and fail on unknown variants.
- Do not write defensive code when the type already tells you what exists.
- Use asserts when loading data or when something must exist.
- Be opinionated about parameters. Pass only what is required.
- Remove changes that are not strictly required.
- Bias toward fewer lines of code.
- Avoid complex or clever code.
- Avoid breaking logic into too many functions.
- Prefer early returns.
- Prefer asserts over `try`/`catch`, fallback defaults, and existence checks when the value is expected to be present.
- Do not pass overrides unless they are strictly necessary.
- Do not make arguments optional when callers always need to provide them.

## Preferred Patterns

- Change call sites instead of carrying compatibility arguments when the cleanup scope allows it.
- Inline one-use helpers when that makes the main path easier to read.
- Use a small `switch` on a discriminant instead of scattered conditionals.
- Replace "accept anything and normalize later" with a narrower input type.
- Delete fallback branches that should be impossible under the declared types.

## Avoid

- Optional parameter bags.
- "Just in case" defaults.
- Broad `try`/`catch` for normal control flow.
- Generic helpers that hide straightforward logic.
- Refactors that expand the diff without making the code more obvious.

## Exit Criteria

- The final code reads simply from top to bottom.
- The number of states and parameters is smaller than before.
- Unknown variants fail explicitly.
- Unnecessary edits and abstractions are gone.
