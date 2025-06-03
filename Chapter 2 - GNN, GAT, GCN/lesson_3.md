# Lesson 3 — Fundamental Symmetries: Permutation Invariance & Equivariance  

*Graph Neural Networks gain much of their power in physics‐based problems by **respecting how Nature ignores arbitrary labels**.  
In this lesson we unpack what that sentence really means and how GNNs achieve it.*

---

## 1. Why should we care about permutations?

Imagine simulating a glass of water that contains 50 000 fluid particles.  
Whether you list those particles as  

```
[ p₁,  p₂,  p₃, …, p₅₀₀₀₀ ]
```  

or randomly shuffle the order

```
[ p₁₀₄₂, p₃₇, p₄₉₉, …, p₃ ]
```

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
| **Permutation invariance** |  f(P x) = f(x) | The *value* you predict for the whole system should not change if you shuffle inputs.   Example: total mass, total energy. |
| **Permutation equivariance** |  f(P x) = P f(x) | The *ordering* of the outputs follows the ordering of the inputs.   Example: velocity predicted for particle *i* still ends up in the *i*-th slot after a permutation. |

Those definitions generalise the better-known translation equivariance of CNNs [3].  
Replace “shifting pixels” with “shuffling particles” and the analogy is exact.

---

## 3. A quick detour: the group-theoretic point of view

The set of all permutations of *N* objects forms the **symmetric group Sₙ**.  
Writing the property in group language is convenient:

* f is **G-invariant** ⇔ f(g·x) = f(x) ∀ g ∈ G*  
* f is **G-equivariant** ⇔ f(g·x) = ρ(g) f(x) ∀ g ∈ G*,  

where ρ is a representation of the group (often the very same permutation action).  
For permutations we usually take ρ(g)=g itself.

Why mention this?  
Because the exact same algebra will later let us talk about **SE(3)-equivariance** (rotations & translations in 3-D space) with hardly any notational changes [17].

---

## 4. How GNNs guarantee permutation equivariance

### 4.1 Message passing plumbing

A typical GNN layer does three steps

1. **Message computation**  
   m<sub>u→v</sub> = φ<sub>M</sub>(h<sub>u</sub>, h<sub>v</sub>, e<sub>uv</sub>)
2. **Aggregation (permutation *invariant*)**  
   M<sub>v</sub> = 𝐀𝐆𝐆 { m<sub>u→v</sub> : u ∈ 𝒩(v) }
3. **Update (shared weights)**  
   h′<sub>v</sub> = φ<sub>U</sub>(h<sub>v</sub>, M<sub>v</sub>)

The *only* place where neighbour messages are combined is the **AGG** step.  
As long as AGG is a multiset function such as **sum**, **mean**, **max** or **attention pooling**, its output is order-independent [14].

Because every node repeats exactly the same computation with weight sharing, the overall layer obeys

  H′ = f<sub>layer</sub>(P H) = P f<sub>layer</sub>(H).

Stacking layers preserves that property, hence the entire network is permutation equivariant by design [18].

### 4.2 Graph-level readout

To predict a global scalar (say, total energy) we add a readout  

 y = READOUT { h<sub>v</sub> : v ∈ V }  

with another permutation-invariant function — usually a simple sum or mean.  
That turns equivariance (node-wise) into **invariance** (graph-wise).

---

## 5. Concrete example

Take three particles in 1-D with positions x = [1, 4, 7].  
A GNN predicts accelerations a = [0.2, −0.3, 0.1].

Now apply permutation P = (1 3) that swaps the first and third particle:

• Input becomes x̃ = [7, 4, 1]  
• GNN outputs ã = [0.1, −0.3, 0.2] = P a

Graph-level **kinetic energy** estimate  
E = ½ Σ m v²  
is identical whether computed on (x,a) or (x̃,ã) → invariance.

---

## 6. Analogy: Passengers & bus tickets

Think of a city bus with 50 passengers.  
Passengers buy tickets; the driver only cares about **how many tickets were sold** (invariant) while the ticket inspector cares that **each person’s ticket matches that person** (equivariant).  
If everyone swaps seats (a permutation), the driver’s count stays the same, but the inspector’s mapping of “ticket ↔ seat” must permute the same way.

---

## 7. Why physicists love these symmetries

1. **Data efficiency** – the network need not “re-learn” that p₁ and p₂ are interchangeable; the architecture encodes it [11].  
2. **Generalisation** – models trained on 1 000-particle systems can run on 10 000-particle systems because counting order is irrelevant.  
3. **Faithfulness to laws** – most conservation laws are *label-free*. Encoding symmetry reduces the risk of unphysical behaviour.

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

• **SE(3)-equivariance** — rotations & translations in 3-D molecular systems [17].  
• **Gauge equivariance** — fields on meshes.  
• We will encounter these richer symmetry groups when building force-fields and turbulence models later in the course.

---

## 10. Summary

Permutation invariance/equivariance is the **bedrock symmetry** for particle GNNs.  
By building it into the architecture we:

* eliminate spurious dependence on arbitrary ordering,  
* gain enormous data-efficiency, and  
* align the network with fundamental physics.

In the next lesson we will see how message passing turns this symmetry plus local interactions into a full-blown **relational inductive bias**.

---

## References

[3] Translation Invariance & Equivariance in Convolutional Neural Networks – Paperspace Blog  
[11] *Permutation invariance, in the context…* – arXiv:2403.17410v2  
[12] *On permutation-invariant neural networks* – arXiv  
[14] *The Graph Neural Network Model* – McGill GRL book chapter  
[17] *Approximately Equivariant Graph Networks* – NeurIPS 2023  
[18] *Why is the output of my graph neural network not permutation equivariant?* – AI StackExchange  
[21] Graph Attention Networks – Velickovic et al.
