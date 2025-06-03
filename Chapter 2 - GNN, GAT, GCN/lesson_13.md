# Lesson 13 – Graph Attention Networks (GAT): Learning Dynamic Interaction Weights  


## 1. Why bother with attention on graphs?  

Graph Convolutional Networks (GCN) propagate information with *fixed* (degree–based) weights.  
That is perfectly fine when every neighbour is equally informative, but physical and real-world graphs are rarely so egalitarian:

* In a turbulent fluid pocket, the particle about to collide with me is more relevant than a distant one.  
* In a social network, a close friend influences my opinion more than an occasional acquaintance.

Graph Attention Networks (GAT) replace those hard-coded weights with **learnable, data-dependent coefficients**, letting the model *decide* who to listen to at each step [6][54].

> **Analogy (crowd-conversation):**  
> Stand in a noisy conference hall. You automatically focus on a nearby speaker using keywords you care about and tune out background chatter.  
> GAT’s attention mechanism performs the same selective “focus” for every node.

---

## 2. Recap: Message passing without and with attention  

For a node \(i\) with neighbours \(\mathcal{N}(i)\) and features \(\mathbf{h}_i\):

| Step | Classical GCN | GAT |
|------|---------------|-----|
| Aggregate | \(\displaystyle \mathbf{m}_i = \frac{1}{|\mathcal{N}(i)|}\sum_{j\in\mathcal{N}(i)}\mathbf{h}_j\) | \(\displaystyle \mathbf{m}_i = \sum_{j\in\mathcal{N}(i)} \underbrace{\alpha_{ij}}_{\text{learned}}\;\mathbf{h}_j\) |
| Update | \(\mathbf{h}_i' = \sigma(\mathbf{W}\mathbf{m}_i)\) | Same, but with learned \(\alpha_{ij}\) |

Therefore, the heart of GAT is computing those attention weights \(\alpha_{ij}\).

---

## 3. GAT layer — step-by-step mathematics  

### 3.1 Linear projection  

\[
\mathbf{g}_i \;=\; \mathbf{W}\, \mathbf{h}_i
\]

The shared matrix \(\mathbf W \in \mathbb{R}^{F' \times F}\) maps every node to a new feature space.

### 3.2 Pair-wise attention score  

\[
e_{ij} \;=\; \mathrm{LeakyReLU}\!\bigl(\, \mathbf{a}^{\top}\,[\mathbf{g}_i \,\|\, \mathbf{g}_j] \bigr)
\]

* \([\cdot\|\cdot]\) = concatenation  
* \(\mathbf{a} \in \mathbb{R}^{2F'}\) is a learnable vector.

### 3.3 Normalisation (softmax over the neighbourhood)  

\[
\alpha_{ij} \;=\;\frac{\exp(e_{ij})}{\displaystyle \sum_{k\in\mathcal{N}(i)} \exp(e_{ik})}
\]

Now \(\sum_{j\in\mathcal{N}(i)} \alpha_{ij}=1\).

### 3.4 Weighted aggregation and non-linear update  

\[
\mathbf{h}_i' \;=\; \sigma\!\Bigl(\, \sum_{j\in\mathcal{N}(i)} \alpha_{ij}\; \mathbf{g}_j \Bigr)
\]

\(\sigma\) is usually ELU or ReLU.

---

## 4. Multi-head attention  

Multiple heads (say \(K\)) learn independent attention patterns:  

\[
\mathbf{h}_i' \;=\; \big\|_{m=1}^{K} 
               \sigma\!\Bigl(\sum_{j\in\mathcal{N}(i)} 
               \alpha_{ij}^{(m)}\;\mathbf{W}^{(m)}\mathbf{h}_j \Bigr)
\]

* **Concatenation** (shown above) is common for intermediate layers.  
* **Averaging** heads is typical in the final layer for stability [6].

Multi-heads improve model expressivity and stabilise training (each head sees a slightly different view of the neighbourhood).

---

## 5. Desirable properties  

| Property | Why it matters | How GAT achieves it |
|----------|----------------|---------------------|
| **Permutation equivariance** | Node ordering should not change the output. | Coefficients \(\alpha_{ij}\) depend only on features, not on ordering. |
| **Inductive generalisation** | Works on unseen graphs / timesteps. | Attention is computed *locally* per edge, no global spectral basis required [6][56]. |
| **Sparse & parallel friendly** | Crucial for millions of edges in particle simulations. | Edge-wise computations fit well with GPU kernels and sparse libraries. |

---

## 6. Where do GATs shine in physical simulations?  

1. **Highly anisotropic interactions**  
   * Example: Near a solid boundary, fluid particles on the “fluid side” matter more than those across the wall.  
   * GAT can up-weight the fluid-side neighbours automatically [55].

2. **Event-driven phenomena**  
   * A pending collision produces a brief yet dominant force; learned attention spikes exactly when needed.

3. **Heterogeneous multi-physics**  
   * Different material types (fluid, elastic solid) encoded in node features – attention learns cross-material coupling strength.

---

## 7. Practical tips for implementation  

| Hyper-parameter | Typical range | Impact |
|-----------------|---------------|--------|
| Heads \(K\) | 4 – 8 | More heads ⇢ richer patterns, higher memory. |
| Feature dim. \(F'\) | 32 – 128 per head | Balance expressivity vs. over-fitting. |
| Negative-slope in LeakyReLU | 0.1 – 0.2 | Stabilises gradients. |
| Dropout on \(\alpha_{ij}\) | 0.0 – 0.6 | Regularises, prevents attention collapse. |

**Code snippet (PyTorch Geometric)**  

```python
from torch_geometric.nn import GATConv
conv = GATConv(
        in_channels  = F,
        out_channels = F_out,
        heads        = 8,
        concat       = True,      # set False for final layer
        dropout      = 0.2)
h_out = conv(x, edge_index)      # x: [N,F], edge_index: [2,E]
```

---

## 8. Limitations and remedies  

* **Quadratic neighbourhood cost** – identical to GCN, but multi-head scaling adds memory.  
  *Mitigation:* sub-sampling neighbours or using *sparse attention* tricks.

* **Attention collapse** – all weights become similar.  
  *Mitigation:* increase temperature (softmax sharpness), add dropout on coefficients.

---

## 9. Summary  

Graph Attention Networks endow message passing with **learnable, context-aware weights**, letting each node decide *whom* to trust.  
For physics, this means:

* capturing fine-grained, direction-dependent forces,  
* maintaining permutation equivariance and efficiency,  
* generalising to unseen particle counts or mesh topologies.

In the grand scheme, GAT is a natural evolution from CNN → GCN … injecting the *attention revolution* of NLP into graphs.

---

## References  

[6] Graph Attention Networks — Petar Veličković. <https://petar-v.com/GAT/>  
[21] “Graph Attention Networks” — Baeldung on CS. <https://www.baeldung.com/cs/graph-attention-networks>  
[54] GAT Explained — Papers With Code. <https://paperswithcode.com/method/gat>  
[55] A. Wong, “Graph Neural Networks — Part 2: GAT vs. GCNs.” Towards Data Science. <https://towardsdatascience.com/graph-neural-networks-part-2-graph-attention-networks-vs-gcns-029efd7a1d92/>  
[56] Graph Neural Networks Part 2 — GATs vs. GCNs. <https://towardsdatascience.com/graph-neural-networks-part-2-graph-attention-networks-vs-gcns-029efd7a1d92/>  
[57] Annotated GAT implementation — NN LabML. <https://nn.labml.ai/graphs/gat/index.html>  
