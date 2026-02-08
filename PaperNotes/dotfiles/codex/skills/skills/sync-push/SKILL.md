---
name: sync-push
description: Sync and commit blog changes between Obsidian and Website repos with same commit message
---

# /sync-push

Commits blog changes to both repositories (Obsidian and Website) with the same commit message, keeping them in sync.

## Workflow

1. **Detect current repository** (Obsidian or Website)
2. **Stage and commit current repo** with conventional commit message
3. **Run sync script** to transfer files between repos
4. **Switch to other repository**
5. **Check for uncommitted changes** in other repo
6. **Handle based on situation** (see scenarios below)
7. **Push both repositories** to remote

## Repository Paths

- **Obsidian**: `/Users/gaurangmathur/Gaurang/Digital Garden/annotation-notes`
- **Website**: `/Users/gaurangmathur/Gaurang/Code/_llm_os/mathur-exe.github.io`
- **Sync Script**: `.blog-sync/sync-blogs.sh` (relative to website repo)

## Scenarios

### Scenario A: Other repo is clean
- Stage synced changes
- Commit with **same message** as first repo
- Push both repos

### Scenario B: Other repo has uncommitted changes
**ASK USER**:
```
Website repo has uncommitted changes:
- modified: src/content/blog/some-file.md
- modified: src/styles/global.css

What would you like to do?
1. Commit website changes first, then sync
2. Stash website changes, sync, then restore
3. Skip website commit (only commit obsidian)
4. Abort entirely
```

### Scenario C: Sync produces no changes
- Skip commit to other repo (nothing to commit)
- Push only the first repo
- Inform user: "No blog changes to sync"

### Scenario D: Merge conflicts after sync
**ASK USER**:
```
Sync resulted in conflicts:
- src/content/blog/General Learnings.md (both modified)

Please resolve conflicts manually, then run /sync-push again.
```

## Implementation Steps

1. **Check current directory**
   ```bash
   pwd
   # Determine if we're in obsidian or website repo
   ```

2. **Stage and commit current repo**
   ```bash
   git add -A
   git status --short
   # Generate conventional commit message based on changes
   git commit -m "type: description"
   # Save commit message for second repo
   COMMIT_MSG=$(git log -1 --pretty=%B)
   ```

3. **Run sync script**
   ```bash
   cd /Users/gaurangmathur/Gaurang/Code/_llm_os/mathur-exe.github.io
   ./.blog-sync/sync-blogs.sh
   ```

4. **Switch to other repo**
   - If started in Obsidian → switch to Website
   - If started in Website → switch to Obsidian

5. **Check other repo status**
   ```bash
   git status --short
   # Check for uncommitted changes before sync
   git status --porcelain
   ```

6. **Handle based on scenario** (see above)

7. **Push both repos**
   ```bash
   git push origin $(git branch --show-current)
   ```

## Commit Message Format

Use conventional commits based on file changes:

- `feat:` - New blog posts, new content
- `fix:` - Corrections, typos, broken links
- `docs:` - README updates, documentation
- `style:` - Formatting, CSS changes
- `refactor:` - Structural changes, reorganizing
- `chore:` - Maintenance, config updates

**Examples**:
- `feat: add tensor transformation notes`
- `fix: correct math equation in general learnings`
- `docs: update blog sync documentation`

## Safety Checks

Before committing, verify:
- [ ] No debug console.log statements in code
- [ ] No temporary files (.tmp, .bak)
- [ ] No sensitive data (API keys, passwords)
- [ ] Changes are intentional

## Error Handling

### If sync script fails
- Abort operation
- Report error to user
- Don't commit either repo

### If commit fails (empty, hooks, etc.)
- Report specific error
- Ask user how to proceed
- Don't leave repos in inconsistent state

### If push fails (network, auth)
- Commits are local (good)
- Report push failure
- User can push manually later

## Success Output

```
✅ Committed Obsidian: "feat: add tensor notes"
🔄 Synced blog files
✅ Committed Website: "feat: add tensor notes"
📤 Pushed both repositories

Both repos are now in sync!
```

## Notes

- Always commit current repo first (source of truth)
- Sync runs between commits (not before)
- Same commit message ensures traceability
- User intervention only when conflicts arise
- Push happens automatically after both commits succeed
