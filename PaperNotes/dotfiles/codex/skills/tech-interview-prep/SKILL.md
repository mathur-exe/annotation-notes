---
name: tech-interview-prep
description: Analyze a codebase, project statement, or resume to generate technical interview preparation for AI Engineer and Research Engineer roles (mid-to-senior). Tests deep understanding of what was built and why. Supports interactive mock interview and question bank generation. Use when the user wants to prepare for technical interviews, practice project deep-dives, generate interview questions from code or resume, or run a mock interview.
---

# Tech Interview Prep

Turn a codebase, project description, or resume into a rigorous technical interview session targeting AI Engineer / Research Engineer roles at mid-to-senior level.

## Activation

When invoked, determine the **input source** and **mode** before proceeding.

### Step 1: Identify Input Source

Ask the user (or infer from context) which input to analyze:

| Source | How to Analyze |
|--------|---------------|
| **Codebase** (default if inside a repo) | Walk the repo: directory structure, README, key modules, configs, tests. Identify tech stack, architecture patterns, data flow, and model pipeline. |
| **Resume** (PDF/text file) | Extract projects, technologies, accomplishments, scope, and impact metrics. Focus on items the user built vs. used. |
| **Project statement** (text) | Parse scope, objectives, technical approach, stack, and outcomes. |

If multiple sources are available (e.g., codebase + resume), combine them for richer context.

### Step 2: Choose Mode

Ask:
```
Which mode?
a) Question Bank — generates a document with questions + expected talking points
b) Mock Interview — interactive session (I ask, you answer, I follow up and evaluate)
c) Both — question bank first, then mock interview

Reply: a, b, or c
```

---

## Codebase Analysis Protocol

When analyzing a codebase, gather context in this order:

1. **Structure**: List top-level directories and key files (README, configs, entrypoints)
2. **Stack**: Identify languages, frameworks, ML libraries, infra tools
3. **Architecture**: Map major components and their relationships (data ingestion → processing → model → serving)
4. **Design decisions**: Note non-obvious choices (why this DB, why this model architecture, why this training strategy)
5. **Data flow**: Trace how data moves from raw input to final output/prediction
6. **Testing & quality**: Check for tests, CI/CD, linting, experiment tracking
7. **Gaps**: Note missing pieces (no tests, no monitoring, no docs) — these become interview questions too

Spend no more than 5 minutes on discovery. Read strategically — READMEs, main entrypoints, config files, and model definitions first.

---

## Question Generation Framework

Generate questions across these 6 dimensions. Weight toward dimensions 1-3 for AI/Research Engineer roles.

### Dimension 1: Architecture & Design Decisions (highest weight)

Test *why* choices were made, not just *what* was built.

**Patterns:**
- "Walk me through the high-level architecture of [project]. Why did you structure it this way?"
- "You chose [X] over [Y]. What drove that decision? What trade-offs did you accept?"
- "If you were starting this project today, what would you do differently?"
- "How does [component A] communicate with [component B]? Why that approach?"

### Dimension 2: Technical Deep-Dive

Test whether the candidate truly understands the internals vs. just used a library.

**Patterns:**
- "Explain how [specific component] works under the hood."
- "Walk me through what happens when [specific input] enters the system."
- "You used [library/algorithm]. How does it work internally? What are its failure modes?"
- "What happens if [edge case / failure scenario]?"
- "Show me the most complex part of this codebase and explain it."

### Dimension 3: ML/AI-Specific

Targeted at AI Engineer and Research Engineer roles.

**Patterns:**
- "How did you select this model architecture? What alternatives did you evaluate?"
- "Describe your training pipeline. How did you handle [data quality / class imbalance / distribution shift]?"
- "What metrics did you use to evaluate the model? Why those and not others?"
- "How do you monitor model performance in production? What does degradation look like?"
- "Explain the feature engineering decisions. Which features had the most impact?"
- "How would you retrain or fine-tune this model if the data distribution changed?"
- "What baselines did you compare against? How did you ensure a fair comparison?"

### Dimension 4: Scale, Trade-offs & Limitations

**Patterns:**
- "What are the main limitations of this system?"
- "How would this scale to 10x / 100x the current load?"
- "What's the latency profile? Where are the bottlenecks?"
- "If you had to cut scope by 50%, what would you keep and what would you drop?"
- "What technical debt exists? What would you prioritize paying down?"

### Dimension 5: Production & Deployment

**Patterns:**
- "How would you take this from prototype to production?"
- "What would your deployment strategy look like? How do you handle rollbacks?"
- "How do you ensure reproducibility of experiments/results?"
- "What observability would you add? What alerts would you set up?"

### Dimension 6: Collaboration & Process

**Patterns:**
- "How did you scope this project? How did you decide what to build first?"
- "What was the hardest technical challenge? How did you unblock yourself?"
- "How did you validate your approach before investing significant time?"
- "If a teammate disagreed with your architecture choice, how would you resolve it?"

---

## Mode: Question Bank

Generate a structured markdown document with 15-25 questions organized by dimension.

### Output Template

```markdown
# Technical Interview Prep: [Project Name]

## Project Summary
[2-3 sentence summary of what was analyzed]

## Tech Stack
[Bulleted list of key technologies identified]

---

## Architecture & Design Decisions
1. [Question]
   **Expected talking points:** [Key points the candidate should cover]

2. [Question]
   **Expected talking points:** [Key points]

## Technical Deep-Dive
3. [Question]
   **Expected talking points:** [Key points]
...

## ML/AI-Specific
...

## Scale, Trade-offs & Limitations
...

## Production & Deployment
...

## Collaboration & Process
...

---

## Rapid-Fire Questions
[5-8 short questions expecting concise answers — tests breadth]

## Red Flags to Avoid
- [Common weak answers or anti-patterns for this specific project]
```

### Guidelines for Question Bank
- Questions should be **specific to the actual project**, not generic
- Expected talking points should reference actual files, components, or decisions found in the source
- Include at least 2 "curveball" questions that probe edge cases or failure modes
- Include a mix of open-ended ("walk me through...") and pointed ("why X over Y?")

---

## Mode: Mock Interview

Run an interactive interview session. The agent plays the role of a senior technical interviewer.

### Interview Protocol

1. **Opening (1 question)**
   "Give me a 2-minute overview of [project]. What problem does it solve and what was your role?"

2. **Architecture probe (2-3 questions)**
   Start broad, then drill into the most interesting design decision.

3. **Technical deep-dive (3-4 questions)**
   Pick the most complex component. Ask how it works. Follow up on vague answers.

4. **ML/AI-specific (2-3 questions)**
   Model selection, evaluation, data pipeline decisions.

5. **Trade-offs & limitations (1-2 questions)**
   Push on what doesn't work well. Test intellectual honesty.

6. **Wrap-up (1 question)**
   "What would you do differently if you started over?"

### Interviewer Behavior Rules

- **Ask one question at a time.** Wait for the user's response before proceeding.
- **Follow up on vague answers.** If the user says "I used transformer architecture," ask "Which variant? Why that one? How did you handle [specific challenge]?"
- **Probe for depth.** Surface-level answers get a follow-up. The goal is to find the boundary of understanding.
- **Stay in character.** Do not break out of interviewer mode to explain or teach during the session.
- **Be respectful but direct.** A real interviewer won't accept hand-waving.
- **Track coverage.** Mentally note which dimensions have been covered. Ensure at least 4 of 6 are hit.

### After the Interview

When the session ends (user says "done", "stop", or all dimensions are covered), provide a structured evaluation:

```markdown
## Interview Evaluation

### Overall: [Strong / Moderate / Needs Work]

### Dimension Scores
| Dimension | Score (1-5) | Notes |
|-----------|-------------|-------|
| Architecture & Design | X | [Brief note] |
| Technical Deep-Dive | X | [Brief note] |
| ML/AI-Specific | X | [Brief note] |
| Trade-offs & Limitations | X | [Brief note] |
| Production & Deployment | X | [Brief note] |
| Collaboration & Process | X | [Brief note] |

### Strengths
- [What went well]

### Areas to Improve
- [Specific gaps with actionable suggestions]

### Suggested Study Topics
- [Topics to review before the real interview]
```

---

## Handling Multiple Projects (Resume Input)

When analyzing a resume with multiple projects:

1. List all identified projects with a one-line summary each
2. Ask the user which project(s) to focus on
3. If the user says "all," pick the 2-3 most substantial and generate questions for each
4. For cross-project questions, add a section on common themes and transferable patterns

---

## Anti-Patterns (What NOT to Do)

- Do not ask generic questions that could apply to any project ("Tell me about a time you worked on a team")
- Do not accept the user's first answer without probing deeper in mock interview mode
- Do not generate questions about technologies not actually used in the project
- Do not produce a wall of 50+ questions — keep it focused and high-signal
- Do not break interviewer character during mock mode to explain concepts
