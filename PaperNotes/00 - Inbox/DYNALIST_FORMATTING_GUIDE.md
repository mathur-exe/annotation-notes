# Dynalist Formatting Guide

## Overview

This document codifies the structural formatting patterns used in Dynalist-style markdown content. These patterns represent a specific way of organizing information using hierarchical bullet points rather than traditional paragraphs.

## Core Philosophy

- **Bullet-first writing**: Information is organized in nested bullet points, not paragraphs
- **Hierarchical thinking**: Ideas flow from parent → child → grandchild levels
- **Atomic units**: Each bullet represents a discrete thought or concept
- **Visual scanning**: Structure enables quick scanning and synthesis

## Structural Patterns

### 1. Date-Based Entry Points

```markdown
- **!(YYYY-MM-DD) Entry Title**
```

**Purpose**: Primary organization unit, typically chronological
**Rules**:
- Use `!(date)` format for machine parsing
- Bold the entire entry point
- Date format: ISO 8601 (YYYY-MM-DD)

**Example**:
```markdown
- **!(2026-02-01) From Scalars to Tensors: The JEE Illusion**
```

### 2. Indentation Hierarchy

```markdown
- Level 1 (Parent/Entry)
	- Level 2 (Child/Detail)
		- Level 3 (Grandchild/Specific)
			- Level 4 (Implementation/Example)
```

**Rules**:
- Use tabs (not spaces) for indentation
- Each level represents increasing specificity
- Maximum practical depth: 4-5 levels
- Sibling items at same level share equal importance

**Visual Structure**:
```
• Parent concept
  ◦ Supporting detail
    ▪ Specific example
      ▫ Implementation note
```

### 3. Content Types by Level

#### Level 1: Entry Point
- **Format**: Date + Title (bold)
- **Purpose**: Main topic/container
- **Children**: Always present

#### Level 2: Core Concepts
- **Format**: Plain text or **Bold** for emphasis
- **Purpose**: Primary ideas, definitions, or assertions
- **Content**: Single sentences or short phrases

**Patterns**:
```markdown
	- Core concept statement here.
	- **Key Term**: Explanation or definition.
	- **The Assumption:** Labelled concept with colon.
```

#### Level 3: Elaboration & Examples
- **Format**: Regular text with occasional **bold** highlights
- **Purpose**: Expand on Level 2, provide examples
- **Content**: 1-2 sentences max per bullet

**Patterns**:
```markdown
		- Elaboration on the parent concept.
		- **Scenario:** Specific situation or example.
		- **Derivation of Failure** using method (equation here):
```

#### Level 4+: Technical Details
- **Format**: Equations, code, specific data
- **Purpose**: Mathematical proofs, technical specifics
- **Content**: LaTeX math, code snippets, precise values

**Patterns**:
```markdown
			- In the Y-direction: $j_y = \sigma E_y$
			- Substitute input ($E_y = 0$): $j_y = \sigma \cdot 0$
			- **Scalar Result:** $j_y = 0$
```

### 4. Special Markers

#### Intuition Blocks
```markdown
	- Intuition: Brief, conceptual explanation in plain language.
```
**Purpose**: Provide mental models before technical details
**Location**: Always early in hierarchy (Level 2-3)

#### Physical Reality Blocks
```markdown
	- Physical Reality: Real-world observation or constraint.
```
**Purpose**: Ground abstract concepts in reality
**Location**: After mathematical/scientific claims

#### Highlight Markers
```markdown
	- ==Key Concept==: Critical insight or conclusion.
```
**Purpose**: Draw attention to crucial points
**Rendering**: Yellow-green background (#9db83b)

#### Next/Action Items
```markdown
	- **Next:** Follow-up task or continuation.
```
**Purpose**: Indicate workflow continuation
**Location**: End of major sections

### 5. Mathematical Notation

**Inline Math**: Use single dollar signs
```markdown
	- The value of $\sigma$ determines conductivity.
```

**Display Math**: Use double dollar signs (standalone)
```markdown
		- **The Tensor Equation:**
			$$
			\begin{pmatrix} j_x \\ j_y \end{pmatrix} = \begin{pmatrix} \sigma_{xx} \& \sigma_{xy} \\ \sigma_{yx} \& \sigma_{yy} \end{pmatrix} \begin{pmatrix} E_x \\ E_y \end{pmatrix}
			$$
```

**Rules**:
- Place complex equations at deeper indentation levels (3-4)
- Label equations with bold text
- Reference variables in surrounding text

### 6. Section Numbering

```markdown
	- 1. The Roadblock: When Scalars Fail
		- ...
	- 2. The Resolution: The Rank-2 Tensor
		- ...
```

**Purpose**: Sequential, ordered progression
**Format**: Number + Period + Title (colon optional)
**Location**: Level 2 (direct child of entry point)

### 7. Tables

```markdown
		| Entity | Rank ($r$) | Components | Intuition |
		| --- | --- | --- | --- |
		| **Scalar** | **0** | $n^0 = 1$ | Single magnitude |
```

**Rules**:
- Place at Level 3 or deeper
- Keep columns narrow (4 max)
- Use bold for row headers
- Include mathematical notation inline

### 8. Checklists

```markdown
	- 4. Verification Checklist
		- (1) **Dimensional Analysis:** Do units match?
		- (2) **Symmetry Check:** Is matrix symmetric?
		- (3) **Edge Cases:** What about alignments?
```

**Format**: Numbered with parentheses, bold label, colon
**Purpose**: Verification steps or requirements

## Structural Anti-Patterns (Avoid)

❌ **Paragraphs in bullets**:
```markdown
	- This is a long paragraph that goes on and on without breaking into atomic thoughts. It should be split into multiple bullets instead.
```

❌ **Mixed indentation** (spaces + tabs):
```markdown
    - This uses spaces (bad)
	- This uses tabs (good)
```

❌ **Inconsistent depth jumps**:
```markdown
- Level 1
		- Level 3 (skipped Level 2)
```

❌ **Orphan bullets** (no siblings):
```markdown
- Parent
	- Only child (should this be merged with parent?)
```

## Layout Integration

### For Website Rendering

**Layout Mode**: `layout-wide`
**Features**:
- Full-width container (800px text column)
- Whitney font family (sans-serif, Dynalist-style)
- Background: #181818 (dark gray)
- Text: #eeeeee (off-white)
- Line height: 1.375 (tight)
- Font size: 16px

**Special Rendering**:
- `==highlight==` → Yellow-green background (#9db83b)
- Math equations → KaTeX rendering
- Nested bullets → Visual hierarchy with indentation

### For Obsidian Editing

**Recommended Plugins**:
- **Outliner**: For bullet navigation and manipulation
- **Indentation Guides**: Visual hierarchy lines
- **Highlight**: For ==syntax== support

**Settings**:
- Tab size: 4 spaces (displays as visual indent)
- Fold headings: Enabled
- Line numbers: Optional

## Mental Model for Agents

When processing Dynalist-style markdown:

1. **Think hierarchically**: Every bullet has a parent (except root)
2. **Parse depth first**: Process children before siblings
3. **Preserve indentation**: Tabs are semantic, not just visual
4. **Extract structure**: Date → Title → Sections → Details
5. **Render relationships**: Parent-child connections are meaningful
6. **Maintain atomicity**: Each bullet = one concept

## Example Complete Entry

```markdown
---
title: 'Sample Entry'
description: 'Demonstrates all structural patterns'
pubDate: '2026-01-01'
layoutType: 'wide'
---

- **!(2026-01-01) Sample Topic Title**
	- Opening context or definition here.
	- Intuition: Simple mental model for the concept.

	- **The Core Concept:** Central thesis or principle.
		- Supporting detail with **bold emphasis** on key terms.
		- ==Critical insight== that stands out visually.

	- 1. First Major Section
		- **Scenario:** Concrete example or situation.
			- Mathematical formulation: $E = mc^2$
			- **Result:** $E = 0$ when $m = 0$
		- Physical Reality: How this manifests in practice.

	- 2. Second Major Section
		- Elaboration with inline math like $\alpha$ and $\beta$.
			$$
			\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
			$$
		- Checklist:
			- (1) **Verify:** Check dimensional consistency
			- (2) **Test:** Edge cases and limits

	- **Next:** Continuation or follow-up topic.
```

## File Organization

**Location**: `src/content/blog/`
**Naming**: Use descriptive titles with spaces (e.g., `General Learnings.md`)
**Frontmatter Required**:
```yaml
---
title: 'Entry Title'
description: 'Brief description'
pubDate: 'YYYY-MM-DD'
layoutType: 'wide'  # Required for Dynalist rendering
---
```

## Version History

- **v1.0** (2026-02-02): Initial codification based on Dynalist analysis
- **Purpose**: Enable consistent formatting across agent interactions
- **Scope**: Structural patterns only (not content guidelines)

---

*This guide ensures any agent can understand, generate, or transform Dynalist-style markdown without requiring additional context or explanation.*
