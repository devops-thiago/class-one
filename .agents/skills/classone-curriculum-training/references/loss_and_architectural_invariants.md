# Loss Functions & Architectural Invariants

## 1. Focal Loss Regularization (`RLCDLoss`)

In policy compliance and safety classification, standard cross-entropy suffers from class imbalance and easy sample saturation:

$$\mathcal{L}_{\text{CE}} = -\log(p_t)$$

When easy samples dominate gradients, model weights do not adapt to subtle, ambiguous boundary cases.

`RLCDLoss` modulates cross-entropy with a focal factor:

$$\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

- Default focal gamma: $\gamma = 2.0$.
- Down-weights well-classified examples ($p_t \approx 0.9 \implies (1 - 0.9)^2 = 0.01$).
- Focuses 99% of gradient update on borderline ambiguous items.
- Result in ClassOne training: average loss dropped from 0.63 down to 0.35.

---

## 2. Bilinear Interaction Scorer (`ChoiceHead`)

Instead of standard dot-product similarity, `ChoiceHead` constructs a 4-component interaction feature vector:

$$\phi(q, k) = [q;\, k;\, |q - k|;\, q \odot k]$$

Where:
- $q \in \mathbb{R}^D$: Query representation vector.
- $k \in \mathbb{R}^D$: Candidate option representation vector.
- $|q - k|$: Absolute difference capturing distance in embedding space.
- $q \odot k$: Element-wise product capturing alignment.

A feedforward projection scores the concatenated $4D$-dimensional feature:

$$\text{Score} = W_2 \cdot \text{GeLU}(W_1 \cdot \phi(q, k)) + \frac{q \cdot k}{\sqrt{D}}$$

This architecture preserves option contrast even when candidate embeddings are close in cosine distance.

---

## 3. Scale-Invariant Normalized Confidence Formula

For any categorical or ordinal decision with $K$ options:

$$\text{Confidence} = \frac{K \cdot P_{\max} - 1}{K - 1}$$

Properties:
- When $P_{\max} = \frac{1}{K}$ (uniform random guess), $\text{Confidence} = 0.0$.
- When $P_{\max} = 1.0$ (complete certainty), $\text{Confidence} = 1.0$.
- Independent of number of candidates $K$, enabling fair thresholding across binary ($K=2$) and 4-way choice ($K=4$) decisions.
