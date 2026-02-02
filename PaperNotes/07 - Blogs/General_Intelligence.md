---
title: 'General Intelligence'
description: 'A journal of learning, insights, and discoveries across physics, mathematics, and engineering'
pubDate: '2026-02-02'
layoutType: 'wide'
---

- **!(2026-02-02) From Scalars to Tensors: Why Physics Gets More Complex**
	- **Intuition:** In introductory physics, we treat everything as simple numbers. But real materials have internal structure that breaks this assumption.

	- **The JEE Illusion: Scalars Everywhere**
		- In basic physics, constants like conductivity ($\sigma$), permittivity ($\epsilon$), and permeability ($\mu$) are taught as single numbers.
		- **The hidden assumption:** Materials are **isotropic** — they behave the same way regardless of direction.
		- **The math:** $\vec{j} = \sigma \vec{E}$ implies current always flows parallel to the applied field.
		- Reality check: This only works for "smooth soup" materials with no internal structure.

	- **When Scalars Break Down: The Crystal Problem**
		- **Scenario:** Imagine a crystal with atoms arranged in diagonal "pipes" or channels.
		- You apply an electric field strictly along the X-axis: $\vec{E} = (E_x, 0)$
		- **Scalar prediction:** $j_y = \sigma E_y = \sigma \cdot 0 = 0$
			- The math says: No current in Y-direction because no field in Y-direction.
		- **Physical Reality:** The diagonal atomic structure "steers" electrons.
			- Even pushing electrons Right (X), they drift Up (Y) because that's the path of least resistance through the crystal lattice.
			- Reality: $j_y \neq 0$ even when $E_y = 0$
		- ==The Roadblock==: Scalar math cannot describe a sideways response to a forward push. A single number can only scale a vector, not rotate it.

	- **The Resolution: Rank-2 Tensors**
		- Replace scalar $\sigma$ with a **matrix** (Rank-2 Tensor) that allows "cross-talk" between dimensions.
		- **The Tensor Equation:**
			$$
			\begin{pmatrix} j_x \\ j_y \end{pmatrix} = \begin{pmatrix} \sigma_{xx} & \sigma_{xy} \\ \sigma_{yx} & \sigma_{yy} \end{pmatrix} \begin{pmatrix} E_x \\ E_y \end{pmatrix}
			$$
		- Now calculate $j_y$ using the full matrix:
			$$
			j_y = \sigma_{yx} E_x + \sigma_{yy} E_y
			$$
		- Even with $E_y = 0$, the $\sigma_{yx}$ component (cross-coupling term) allows $E_x$ to produce current in Y.
		- ==The math finally matches reality.==

	- **Understanding Tensor "Rank"**
		- "Rank" is the exponent that determines data structure complexity.
		
		| Entity | Rank ($r$) | Components in $n$-D ($n^r$) | Intuition |
		| --- | --- | --- | --- |
		| **Scalar** | **0** | $n^0 = 1$ | Single magnitude, no direction |
		| **Vector** | **1** | $n^1 = n$ | Requires 1 direction per component |
		| **Matrix/Tensor** | **2** | $n^2$ | Requires 2 directions (input → output) |
		
		- **Note:** In linear algebra, "rank" means linearly independent rows. In tensors, **Rank** = number of indices needed. A conductivity matrix is always Rank-2, regardless of row independence.

	- **Key Takeaway**
		- Scalars work for isotropic materials (no preferred direction).
		- Tensors are needed for anisotropic materials (crystals, strained silicon, composites).
		- The "grain" or internal structure of materials creates directional dependencies that require matrix representations.

	- **Next:** How do these tensors transform when we rotate the coordinate system? (Change of basis)

---

## Future Entry Template

- **!(YYYY-MM-DD) Entry Title Here**
	- **Intuition:** One-sentence mental model of the concept.

	- **Core Concept**
		- Main idea or principle.
		- **Key assumption:** What's being assumed?
		- **The math:** Key equation or relationship.

	- **The Problem/Scenario**
		- **Scenario:** Concrete situation where the concept applies.
		- **Expected result:** What simple theory predicts.
		- **Physical Reality:** What actually happens.
		- ==The insight==: Why the gap exists.

	- **The Solution/Resolution**
		- How the problem is solved.
		- **Key equation:**
			$$
			\text{Mathematical formulation here}
			$$
		- ==Why it works==: The crucial insight.

	- **Summary Table (if applicable)**
		| Concept | Property | Value/Description |
		| --- | --- | --- |
		| Item 1 | Property A | Value |

	- **Key Takeaway**
		- The main lesson or principle to remember.

	- **Next:** Follow-up topic or question.
