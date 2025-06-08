# Lesson 3 — Fundamental Symmetries: Permutation Invariance & Equivariance  

*Graph Neural Networks gain much of their power in physics‐based problems by **respecting how Nature ignores arbitrary labels**.  
In this lesson we unpack what that sentence really means and how GNNs achieve it.*

---

## 1. Why should we care about permutations?

Imagine simulating a glass of water that contains 50 000 fluid particles.  
Whether you list those particles as  

$$ [p_1, p_2, p_3, \dots, p_{50000}] $$

or randomly shuffle the order

$$ [p_{1042}, p_{37}, p_{499}, \dots, p_3] $$

**the physics is identical**.  
If a model’s prediction *changes* just because you permuted an internal list, you have introduced a non-physical artefact.

Traditional fully-connected networks are sensitive to input order.  
GNNs are deliberately designed not to be.

---

## 2. Two related, but different, symmetry notions

Let **P** be any permutation matrix (i.e. a re-ordering of a list).  
Let **x** be the vector that concatenates all particle features.

| Symmetry type | Formal definition | Intuition |
|---------------|------------------|-----------|
| **Permutation invariance** | $f(Px) = f(x)$ | The *value* you predict for the whole system should not change if you shuffle inputs.   Example: total mass, total energy. |
| **Permutation equivariance** | $f(Px) = P f(x)$ | The *ordering* of the outputs follows the ordering of the inputs.   Example: velocity predicted for particle $i$ still ends up in the $i$-th slot after a permutation. |

Those definitions generalise the better-known translation equivariance of CNNs [3].  
Replace “shifting pixels” with “shuffling particles” and the analogy is exact.

---

## 3. A quick detour: the group-theoretic point of view

The set of all permutations of *N* objects forms the **symmetric group $S_n$**.  
Writing the property in group language is convenient:

*   f is **G-invariant** $\Leftrightarrow f(g \cdot x) = f(x) \quad \forall g \in G$
*   f is **G-equivariant** $\Leftrightarrow f(g \cdot x) = \rho(g) f(x) \quad \forall g \in G$,

where $\rho$ is a representation of the group (often the very same permutation action).  
For permutations we usually take $\rho(g)=g$ itself.

Why mention this?  
Because the exact same algebra will later let us talk about **$SE(3)$-equivariance** (rotations & translations in 3-D space) with hardly any notational changes [17].

---

## 4. How GNNs guarantee permutation equivariance

### 4.1 Message passing plumbing

A typical GNN layer does three steps

1.  **Message computation**  
    $m_{u \to v} = \phi_M(h_u, h_v, e_{uv})$
2.  **Aggregation (permutation *invariant*)**  
    $M_v = \mathbf{AGG} \{ m_{u \to v} : u \in \mathcal{N}(v) \}$
3.  **Update (shared weights)**  
    $h'_v = \phi_U(h_v, M_v)$

The *only* place where neighbour messages are combined is the **AGG** step.  
As long as AGG is a multiset function such as **sum**, **mean**, **max** or **attention pooling**, its output is order-independent [14].

Because every node repeats exactly the same computation with weight sharing, the overall layer obeys

$H' = f_{\text{layer}}(PH) = P f_{\text{layer}}(H)$.

Stacking layers preserves that property, hence the entire network is permutation equivariant by design [18].

### 4.2 Graph-level readout

To predict a global scalar (say, total energy) we add a readout  

$y = \text{READOUT} \{ h_v : v \in V \}$  

with another permutation-invariant function — usually a simple sum or mean.  
That turns equivariance (node-wise) into **invariance** (graph-wise).

---

## 5. Concrete example

Take three particles in 1-D with positions $x = [1, 4, 7]$.  
A GNN predicts accelerations $a = [0.2, -0.3, 0.1]$.

Now apply permutation $P = (1 \ 3)$ that swaps the first and third particle:

• Input becomes $\tilde{x} = [7, 4, 1]$  
• GNN outputs $\tilde{a} = [0.1, -0.3, 0.2] = Pa$

Graph-level **kinetic energy** estimate  
$E = \frac{1}{2} \sum m v^2$  
is identical whether computed on $(x,a)$ or $(\tilde{x},\tilde{a})$ → invariance.

---

## 6. Analogy: Passengers & bus tickets

Think of a city bus with 50 passengers.  
Passengers buy tickets; the driver only cares about **how many tickets were sold** (invariant) while the ticket inspector cares that **each person’s ticket matches that person** (equivariant).  
If everyone swaps seats (a permutation), the driver’s count stays the same, but the inspector’s mapping of “ticket ↔ seat” must permute the same way.

---

## 7. Why physicists love these symmetries

1.  **Data efficiency** – the network need not “re-learn” that $p_1$ and $p_2$ are interchangeable; the architecture encodes it [11].  
2.  **Generalisation** – models trained on 1 000-particle systems can run on 10 000-particle systems because counting order is irrelevant.  
3.  **Faithfulness to laws** – most conservation laws are *label-free*. Encoding symmetry reduces the risk of unphysical behaviour.

---

## 8. Coding checklist

| Item | Quick test |
|------|------------|
| Aggregator is permutation-invariant | `torch.equal(f(x), f(x[torch.randperm(N)]))` |
| Output ordering matches input | `torch.allclose(f(Px), P @ f(x))` |
| Graph-level head is invariant | `f_graph(Px) == f_graph(x)` |

If any of these fail, revisit your aggregation or remember to **batch sort**.

---

## 9. Beyond permutations: the road ahead

• **$SE(3)$-equivariance** — rotations & translations in 3-D molecular systems [17].  
• **Gauge equivariance** — fields on meshes.  
• We will encounter these richer symmetry groups when building force-fields and turbulence models later in the course.

---

## 10. Summary

Permutation invariance/equivariance is the **bedrock symmetry** for particle GNNs.  
By building it into the architecture we:

*   eliminate spurious dependence on arbitrary ordering,  
*   gain enormous data-efficiency, and  
*   align the network with fundamental physics.

In the next lesson we will see how message passing turns this symmetry plus local interactions into a full-blown **relational inductive bias**.

---

[3]: https://blog.paperspace.com/pooling-and-translation-invariance-in-convolutional-neural-networks/
[11]: https://arxiv.org/html/2403.17410v2#:~:text=Permutation%20invariance%2C%20in%20the%20context,of%20the%20set%20are%20arranged.
[12]: https://arxiv.org/html/2403.17410v2
[14]: https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book-Chapter_5-GNNs.pdf
[17]: https://proceedings.neurips.cc/paper_files/paper/2023/file/6cde6435e111671b04f4574006cf3c47-Paper-Conference.pdf
[18]: https://ai.stackexchange.com/questions/40931/why-is-the-output-of-my-graph-neural-network-not-permutation-equivariant
[21]: https://www.baeldung.com/cs/graph-attention-networks