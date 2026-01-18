<!-- SYSTEM_ACTIVATION_TOKEN: TRADING_SYSTEM_V2_FLUID -->
# Obsidian Trading System (Protocol V2)

> **Core Philosophy:** "Capture First, Synthesize Later."
> **Goal:** Build a high-signal library of revisable beliefs (Theory) derived from raw, honest exploration (Thinking), without friction blocking the flow.

---

## 1. The Separation of Concerns
We avoid the "Knowledge Graveyard" by strictly separating **Flow** from **Structure**.

### A. Thinking (Obsidian) = The Mine
- **Purpose:** Raw exploration, emotional venting, confusion processing.
- **Format:** Fluid, bullet-heavy, stream of consciousness (Dynalist style).
- **Rule:** Zero friction to start. Structure only happens at the "Checkout."

### B. Theory (Obsidian) = The Gold
- **Purpose:** Storing "Alpha" (Edge). Coherent, revisable beliefs about market mechanics.
- **Format:** First Principles (Observation → Mechanics → Edge → Lens).
- **Rule:** Every Theory must be derived from a Thinking Log (no unearned wisdom).

### C. Execution (Notion) = The Scoreboard
- **Purpose:** Tracking decisions, outcomes, and P&L.
- **Rule:** Notion is for *data*; Obsidian is for *ideas*.

---

## 2. Workflow Loop

1.  **Capture (Thinking Log):** 
    - You see something. You open a log. You dump the mental context (State: FOMO? Flow?).
    - You write without editing.
2.  **Extract (The Nugget):**
    - At the end of the session, you ask: *"Did I learn anything?"*
    - If YES: Write it in the "Nugget" section.
    - If NO: Close the file. It's just a log.
3.  **Refine (Theory Note):**
    - If a "Nugget" keeps reappearing, promote it to a **Theory Note**.
    - Deconstruct it: *Why does this work? Who is trapped?*
    - Create the "Lens" so you can spot it in real-time next time.

---

## 3. Agent Directives (System Prompt)
*If you are an AI reading this file, adopt the following persona and rules:*

### 🎭 Persona
- **In Thinking Mode:** You are a **Silent Scribe**. Do not interrupt the flow. Do not ask for structure. Only prompt for the "Nugget" when the user signals they are done.
- **In Theory Mode:** You are a **First-Principles Physicist**. Be skeptical. If the user states a claim, ask: *"What is the mechanical cause?"* or *"Who is the trapped participant?"* Do not accept "it just works" as an answer.

### ⚡ Action Rules
1.  **Bootstrapping:** If the user says "Activate Trading System", load these templates and rules into your context immediately.
2.  **Log Review:** When reviewing a `Thinking Log`, do not summarize the content. Instead, look for the **Emotional State** vs. **Outcome** correlation. (e.g., *"You seem to lose money when you write about 'revenge' or 'speed'."*)
3.  **Theory Check:** Ensure every `Theory Note` has a specific **"Lens"** (visual trigger). If it's abstract, ask the user to make it concrete.

---

## 4. Templates

### A. Fluid Thinking Log
*Optimized for speed and psychological context.*
```markdown
---
type: thinking-log
date: {{date}}
mental_state: neutral | flow | fomo | frustration
tags: []
---

## 🧠 Stream
*Dump the context. What are you seeing? What are you feeling?*
- 

---
<!-- EXTRACTION: Don't summarize. Just grab the gold. -->

## ⛏️ The Nugget
*Is there a new insight here? If yes, articulate it in one sentence.*

## 🔗 The Link
*Does this connect to an existing Theory? [[Theory Note]]*
```

### B. First Principles Theory Note
*Optimized for deep conviction and mechanical understanding.*
```markdown
---
type: theory-note
status: active
confidence: low | medium | high
created: {{date}}
---

## 1. Observation (The Puzzle)
*What market phenomenon keeps happening that looks important?*

## 2. The Mechanics (First Principles)
*Strip away the candles. What are the participants doing? Who is buying? Who is selling? Who is in pain? (The Physics)*

## 3. The Edge (Intuitive Solution)
*Given the mechanics, what is the elegant, high-probability outcome? Why does this opportunity exist?*

## 4. The Lens (Synthetic Flow)
*How do I now "see" the market through this concept? What is the visual signature or mental trigger?*
```