# Lesson 5 — Relational Inductive Bias: Encoding Physics into Graph Neural Networks
*Course: Introduction to Graph Neural Networks for Physical Systems*

---

## 1. Why “Inductive Bias” Matters
In machine learning an **inductive bias** is any built-in assumption that narrows the set of solutions a model can represent and *prefers* some explanations over others.
Mathematically, learning can be viewed as searching for a function

$$
f^\star = \underset{f\in\mathcal{H}}{\arg\min}\;\mathcal{L}(f;\mathcal{D})
$$

where
* $ \mathcal{D} $ = training data,
* $ \mathcal{L} $ = loss,
* $ \mathcal{H} $ = hypothesis space.

Inductive bias is **how we choose $ \mathcal{H} $** and influences:

* **Generalisation** – better bias ⇒ fewer samples needed.
* **Interpretability** – built-in structure can be inspected.
* **Search efficiency** – smaller $ \mathcal{H} $ ⇒ faster optimisation.

> Analogy — *Finding a restaurant*.
> If you *assume* good food is likely downtown (bias), you search fewer streets than if you roam the whole city.

### Two flavours
| Type | Meaning | Example |
|------|---------|---------|
| **Restriction bias** | Some functions are **impossible** to represent. | Linear regression can only learn linear mappings. |
| **Preference bias** | All functions are representable, but some are **easier** to learn. | Weight decay favours small-norm solutions. |

Sources: [32], [34]

---

## 2. Inductive Bias in Classical Deep Nets (Quick recap)
* **CNNs**: spatial locality & translation *equivariance* → great for images.
* **RNNs**: sequential ordering → great for time series / language.

These are **not** well-suited for irregular particle systems because those systems revolve around **relationships**, not pixels or time-steps laid out on a fixed grid.

---

## 3. GNNs Introduce a *Relational* Inductive Bias
Graph Neural Networks assume:

1. **Entities as nodes** ($ v\in V $)
2. **Relations as edges** ($ (u,v)\in E $)
3. **Information flows through edges** (message passing).

Hence the hypothesis space is restricted to functions that can be expressed as **iterated local interactions** – exactly how most physical systems work.

### Canonical Message-Passing Equation

$$
\boxed{
\mathbf{h}_v^{(k)} =
\phi_{\text{update}}\!\Bigl(
    \mathbf{h}_v^{(k-1)},
    \underset{u\in\mathcal{N}(v)}{\bigoplus}\;
    \phi_{\text{message}}\!\bigl(\mathbf{h}_v^{(k-1)}, \mathbf{h}_u^{(k-1)}, \mathbf{e}_{uv}\bigr)
\Bigr)}
$$

* $ \mathbf{h}_v^{(k)} $ = embedding of node $ v $ after $ k $ layers
* $ \mathcal{N}(v) $ = neighbour set
* $ \oplus $ = permutation-invariant aggregation (sum/mean/max/attention)
* $ \phi_{\text{message}},\phi_{\text{update}} $ = learnable functions (often MLPs)

Because $ \oplus $ ignores ordering, the whole network is **permutation-equivariant** by construction [14].

---

## 4. Why This Bias Is Golden for Physics
Physical laws are *local but universal*: each particle experiences forces from nearby particles, independent of any ID number you give them. Embedding this principle brings three tangible benefits:

1. **Sample efficiency** – model doesn’t waste capacity “discovering” that particle #17 and #42 obey the same rule.
2. **Better out-of-distribution (OOD) generalisation** – new particle counts or configurations are handled naturally.
3. **Physical plausibility** – easier to enforce conservation laws or symmetry constraints.

> Real-life parallel — *Lego bricks*: no matter which brick you pick, the **rule of connection** is the same. Designs (graphs) may differ, but the local interface (stud + tube) never changes.

Sources: [24], [30], [35], [36]

---

## 5. Injecting *More* Physics via Bias
The basic relational bias can be refined to reflect domain knowledge.

| Engineering knob | Physical meaning | Example |
|------------------|------------------|---------|
| **Edge features** $ \mathbf{e}_{uv} $ | Encodes pairwise geometry or material constants. | Relative displacement $ \Delta\mathbf{x} $, spring constant $ k $. |
| **Aggregation choice** | Assumes how influences combine. | *Sum* for additive forces, *attention* for rare but dominant collisions. |
| **Equivariant layers** | Enforce symmetry groups (e.g. SE(3)). | Rotation-equivariant networks preserve angular momentum [38]. |
| **Global nodes / readouts** | Capture conserved totals. | A “universe” node accumulating total energy to impose conservation. |

### Example: N-body Gravitation
Newtonian force between bodies $ i,j $:

$$
\mathbf{F}_{ij} = G\,\dfrac{m_i m_j}{\|\mathbf{r}_{ij}\|^{3}}\,\mathbf{r}_{ij}
$$

A GNN can approximate this by:
* **Node feature**: $ (m_i,\mathbf{x}_i) $
* **Edge feature**: $ \mathbf{r}_{ij}=\mathbf{x}_j-\mathbf{x}_i $ and its norm
* **Message function** learns the $ 1/r^{2} $ dependence.
Because every timestep repeats the same local rule, the inductive bias matches the physics perfectly, leading to rapid convergence with few examples.

---

## 6. Pitfalls & How to Avoid Them

| Issue | Cause | Mitigation |
|-------|-------|-----------|
| **Wrong graph** | Missing edges → missing forces. | Use physically motivated radii or learned connectivity. |
| **Bias-variance trade-off** | Bias too strong → can’t fit data nuances. | Leave some flexibility (e.g., learnable edge weights). |
| **Oversmoothing** | Too many layers blur distinctions. | Residual/skip connections or layer normalisation [16]. |

---

## 7. Recap Checklist

- [x] **Inductive bias** narrows hypothesis space.
- [x] **Relational bias** in GNNs matches particle interactions.
- [x] Message passing equation guarantees **permutation equivariance**.
- [x] Injecting additional physics (edge features, equivariance) boosts performance.
- [x] Careful design avoids common pitfalls like missing edges or oversmoothing.

---

[14]: https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book-Chapter_5-GNNs.pdf
[16]: https://huggingface.co/blog/intro-graphml
[24]: https://www.epcc.ed.ac.uk/whats-happening/articles/accelerating-smoothed-particle-hydrodynamics-graph-neural-networks
[30]: https://arxiv.org/html/2504.13768v1
[32]: https://jduarte.physics.ucsd.edu/phys139_239/lectures/11_GNNs.pdf
[34]: https://www.kolena.com/guides/understanding-machine-learning-inductive-bias-with-examples-2/
[35]: https://escholarship.org/content/qt0p02k1qw/qt0p02k1qw_noSplash_5e255b645da5f744ed823631d4bd1026.pdf?t=ssy865
[36]: https://www.researchgate.net/publication/325557043_Relational_inductive_biases_deep_learning_and_graph_networks
[38]: https://arxiv.org/html/2402.12449v2