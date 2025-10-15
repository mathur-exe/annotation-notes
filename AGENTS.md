# Repository Guidelines

## Project Structure & Module Organization
- `main.py` holds executable snippets and quick experiments; expand it into modules under `main/` if the script grows.
- `PaperNotes/` stores the core knowledge base. Keep research articles in `Literature Notes/`, running commentary in `Blogs/`, and supporting files in `Assets/` or `images/`.
- Dependency manifests live in `environment.yml` (conda) and `requirement.txt` (pip). Update both when you add APIs or libraries so collaborators can choose their tooling.

## Build, Test, and Development Commands
```bash
conda env create -f environment.yml  # first-time environment setup
conda activate conda_env             # activate shared environment
python main.py                       # run quick checks or demos
pytest                               # execute Python tests (once added)
```
Use `pip install -r requirement.txt` when conda is unavailable, and regenerate the file with `pip freeze > requirement.txt` after dependency changes.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents and descriptive, lowercase_with_underscores names for Python artifacts.
- Prefer type hints and docstrings for any shared utilities.
- For notes, use short, date-prefixed filenames such as `2025-01-12-transformer-basics.md` to keep chronological ordering, and embed diagrams in `PaperNotes/Assets/` with matching stems.

## Testing Guidelines
- Adopt `pytest` with tests located under a top-level `tests/` package that mirrors the source module layout.
- Name test files `test_<subject>.py` and functions `test_<behavior>()` for clarity.
- Run `pytest --maxfail=1 --disable-warnings` before submitting changes; add fixtures for sample prompts or datasets rather than hard-coding paths.

## Commit & Pull Request Guidelines
- Recent history favors terse, descriptive subject lines (for example, `update office notes & notes`). Keep messages imperative, under 72 characters, and add context in the body if the change spans multiple areas.
- Reference related notes or issues directly in the description, and attach screenshots when you alter assets.
- Pull requests should outline the intent, setup steps, and verification commands; flag any new dependencies and confirm that `pytest` completes locally.

## Notes Contribution Workflow
- Capture raw annotations in `PaperNotes/Welcome.md` or `Blogs/` before promoting polished summaries to `Literature Notes/`.
- Store reusable figures as vector-friendly sources (`.svg`, `.pdf`) inside `PaperNotes/Assets/`, and link them with relative paths so exported Markdown remains portable.
- When importing external PDFs, add a citation block at the top of the note and record metadata (title, authors, venue) for quick provenance checks.
