# Technical Blog Stack Analysis

A summary of common technical blog architectures identified for high-performance and academic/research content.

---

## 1. The "Powerhouse" Static Stack (Jekyll + GitHub Pages)

**Reference:** [djdumpling.github.io](https://djdumpling.github.io/2026/01/01/frontier_training.html)

This is the standard for researchers and developers who want full control and professional mathematical rendering.

### **Tech Stack**

- **Generator:** Jekyll (Static Site Generator)
- **Hosting:** GitHub Pages
- **Themes:** Minima (Standard), Minimal Mistakes (Advanced)
- **Math Rendering:** MathJax (LaTeX support)
- **Syntax Highlighting:** Highlight.js or Rouge

### **How Content is Handled**

- **Workflow:** Write in `.md` (Markdown) -> Git Push -> Automatic HTML generation.
- **Hybrid Support:** Standard Markdown handles the text, but you can insert **Raw HTML** for custom image alignment or specialized layouts.
- **LaTeX Implementation:**
  - Wrap equations in `$$` for blocks or `$` for inline.
  - Requires adding the MathJax script to the site header (`head.html`).

---

## 2. The "Minimalist" Hosted Stack (Bear Blog)

**Reference:** [kalomaze.bearblog.dev](https://kalomaze.bearblog.dev/rl-lora-ddd/)

Focused on extreme speed, privacy, and zero configuration.

### **Tech Stack**

- **Platform:** Managed (bearblog.dev)
- **Framework:** None (Minimal CSS)
- **Philosophy:** No-Javascript (where possible), tiny page sizes (<5kb).

### **How Content is Handled**

- **Images:** Often hosted externally (e.g., GitHub Gists) to maintain platform speed.
- **Math:** Avoids heavy rendering libraries like MathJax. Instead, uses:
  - Plain text/code blocks.
  - Image-based formulas for complex LaTeX (e.g., using a service like CodeCogs or custom image exports).

---

## 3. Implementation Comparison

| Feature | Jekyll + GitHub Pages | Bear Blog |
| :--- | :--- | :--- |
| **Control** | Full (HTML/CSS/JS) | Limited (Platform-based) |
| **Math Quality** | High (SVG/HTML rendering) | Low (Text or Images) |
| **Setup Time** | ~1-2 hours | ~1 minute |
| **Local Preview** | Yes (`jekyll serve`) | No (Web Dashboard) |

---

## 5. Hybrid Content Structure (Markdown + HTML + LaTeX)

To replicate a high-quality technical blog that supports Markdown, HTML, and LaTeX, you start with a metadata header.

### **File Header (YAML Frontmatter)**

Jekyll and other static site generators use this "frontmatter" to define the layout and metadata of the page.

```markdown
---
layout: post
title: "Your Post Title"
date: 2026-01-01
use_math: true
---
```

In a technical blog like the ones analyzed, you don't have to choose just one format. Most static site engines (Jekyll, Hugo, etc.) allow you to interleave these formats seamlessly.

### **Text: Standard Markdown**

Use standard Markdown for 90% of your content. It handles headers, lists, and bold/italic text with the least friction.

### **Images: Raw HTML**

If you need precise control (e.g., centering, captions, side-by-side images, or specific widths), use raw HTML tags. The Markdown renderer will skip over these and place them directly into the output.

```html
<!-- Example: Centered image with a restricted width and caption -->
<figure style="text-align: center; margin: 2em 0;">
  <img src="/assets/neural_network.png" alt="Architecture Diagram" style="width: 80%; max-width: 600px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
  <figcaption style="margin-top: 10px; font-style: italic; color: #666;">
    Figure 1: Custom architecture using specific CSS styling.
  </figcaption>
</figure>
```

### **Math: Inline & Block LaTeX**

Equations are written in LaTeX syntax. For this to work, ensure your site header includes a script like **MathJax** or **KaTeX**.

- **Inline Math**: Use single `$`. For example: `$E=mc^2$`.
- **Block Math**: Use double `$$`. For example:

```latex
$$
\frac{\partial \mathcal{L}}{\partial w} = \sum_{i=1}^{n} (y_i - \hat{y}_i) x_i
$$
```

### **Full Example Layout**

```markdown
# My Deep Learning Post

Standard Markdown text goes here to explain the problem.

<div align="right">
  <img src="sidebar-thumb.jpg" width="150" title="Sidebar Visual">
</div>

The loss function is defined as:
$$ L(\theta) = - \frac{1}{N} \sum \log(p) $$

More Markdown follow-up.
```

---

## 6. Snippet Comparison: When to use what?

These snippets serve different roles. Here is a breakdown of their specific functions:

### **A. The "Math Engine" (Setup Code)**

```html
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
```

* **Purpose:** Background logic that **enables LaTeX rendering**. This doesn't display an image; it converts text math (like `$$ ... $$`) into formatted formulas.
- **Placement:** Put this once in your site's **header template** (`head.html`).

### **B. The "Professional Figure" (Centered Block)**

```html
<div style="text-align: center;">
  <img src="/assets/landscape.jpg" alt="Description" style="width: 80%; border-radius: 10px;">
  <p><em>Figure 1: This is centered using raw HTML.</em></p>
</div>
```

* **Purpose:** Used for **Main Diagrams**.
- **Behavior:** It takes up the full width of the page (blocking text) and centers the image. It allows for captions and specific styling (like rounded corners).

### **C. The "Floating Thumbnail" (Inline Wrap)**

```html
<img src="diagram.png" align="right" width="200">
```

* **Purpose:** Used for **Sidebar Visuals** or small thumbnails.
- **Behavior:** The image "floats" to the side (left/right) and allows the **main text to wrap around it**.
