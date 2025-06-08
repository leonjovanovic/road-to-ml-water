# Lesson 12 — Graph Convolutional Networks (GCN): Neighborhood Averaging with Normalisation  

> “GCNs do on graphs what CNNs do on images: they learn **local filters** that can be reused anywhere on the structure.”  

---

## 1. Why do we need a new “convolution”?  

Images live on a **regular grid** – every pixel has exactly eight neighbours.  
Particles in a fluid, atoms in a molecule or papers in a citation network live on **irregular graphs** where  

* each node has a **different** number of neighbours,  
* the notion of “left / right / up / down” is meaningless.  

A GCN generalises the idea of a convolutional kernel to such graphs by **averaging** (or summing) the feature vectors of neighbouring nodes and then applying a learnable linear transform.  

Real-life analogy (for irregular neighbourhoods)  
: Imagine a group chat where each participant posts a short status.  
  1. You collect the latest status of everyone directly connected to you.  
  2. You average their sentiments (“+1” for good mood, “−1” for bad).  
  3. You pass that average through a personal “filter” (your perspective) to decide your next mood.  
  Repeating this process makes everyone’s mood gradually influenced by friends-of-friends, etc.  

---

## 2. Mathematical formulation  

For a graph $G = (V,E)$ with  

* feature matrix $H^{(l)} \in \mathbb{R}^{N\times d_l}$ at layer $l$,  
* adjacency matrix $A$ (binary or weighted),  
* trainable weight matrix $W^{(l)} \in \mathbb{R}^{d_l \times d_{l+1}}$,  

a **GCN layer** is

$$
H^{(l+1)} \;=\; \sigma \!\left( \, \tilde{A}\; H^{(l)} \; W^{(l)} \right), \tag{1}
$$

with  

$$
\tilde{A} \;=\; D^{-\tfrac12}\!\left(A + I\right)\! D^{-\tfrac12}. \tag{2}
$$

* $I$ adds **self-loops** so a node keeps its own information.  
* $D$ is the diagonal degree matrix of $A+I$: $D_{ii} = \sum_j (A+I)_{ij}$.  
* The symmetric normalisation $D^{-\frac12}$ *prevents high-degree nodes from dominating* the average [50].  
* $\sigma$ is usually **ReLU**.  

After $K$ stacked layers every node has integrated information from its **$K$-hop neighbourhood**.  

> ⚠️  Without the normalisation term, features of highly connected nodes can explode, leading to training instability [50].

---

## 3. Intuitive link to CNNs  

| Image CNN                                   | Graph CNN (GCN)                                   |
|---------------------------------------------|---------------------------------------------------|
| 3 × 3 kernel slides over grid               | Normalised adjacency “slides” over graph          |
| Same weights reused at each pixel location  | $W^{(l)}$ reused at each node                   |
| 1-hop receptive field per layer             | 1-hop receptive field per layer                   |
| Pooling enlarges receptive field            | Stacking layers enlarges receptive field          |

Because **parameter sharing** is retained, the total number of weights does **not** depend on $N$, allowing the model to generalise to graphs of unseen size [47].

---

## 4. Step-by-step computation (one layer)

1. **Add self-loops**  
   $A' = A + I$

2. **Compute degrees & normalise**  
   $D_{ii} = \sum_j A'_{ij}$  
   $\tilde{A} = D^{-1/2} A' D^{-1/2}$

3. **Message aggregation**  
   $M = \tilde{A} \, H^{(l)}$ — every row is the (scaled) *average* of neighbour features.

4. **Linear projection**  
   $Z = M \, W^{(l)}$

5. **Non-linearity**  
   $H^{(l+1)} = \text{ReLU}(Z)$

Each step can be implemented efficiently with **sparse matrix–vector products**, crucial for large particle graphs.

---

## 5. Worked example — Two-particle mini fluid  

Suppose we model two water particles **A** and **B** with 1-dim. feature “temperature”.  

| node | $T^{(0)}$ |
|------|------------|
| A    | 90 °C      |
| B    | 10 °C      |

Adjacency $A = \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix}$.  
After adding self-loops and normalising, both nodes simply average:  

$$
\tilde{A} =
\frac12
\begin{bmatrix}
1 & 1\\
1 & 1
\end{bmatrix}
\quad\Longrightarrow\quad
M = \tilde{A} H^{(0)} =
\frac12
\begin{bmatrix}
100\\
100
\end{bmatrix}.
$$

If $W^{(0)} = 1$ and $\sigma$ is identity, each particle’s new temperature is 50 °C — exactly what you would expect after **perfect heat conduction**.

---

## 6. GCN for physics — why it works, when it breaks  

Advantages  
* **Local inductive bias** — many physical laws (e.g. pressure forces in SPH) depend only on nearby neighbours.  
* **Permutation equivariance** — reordering particles does not affect results.  
* **Few parameters** — good when data is scarce.  

Limitations  
* **Uniform weighting**: all neighbours contribute equally (unless additional edge weights are injected).  
* **Isotropic**: cannot prioritise a particular direction; anisotropic turbulence may be under-modelled.  
* **Over-smoothing**: stacking many layers can make every node look alike [44].  

> A common cure for the last point is to add **skip (residual) connections** or switch to **Graph Attention Networks (Lesson 13)**, which learn *adaptive* weights [55].

---

## 7. Implementation snippet (PyTorch Geometric)

```python
import torch
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_ch, hidden, out_ch):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden)   # Eq. (1) internally
        self.conv2 = GCNConv(hidden, out_ch)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)
```

`edge_index` is the **sparse COO** list of edges, automatically normalised inside `GCNConv`.  

---

## 8. Connection to the bigger picture  

GCNs laid the groundwork for almost **all** subsequent GNN variants:  

* Add **edge features** → Message-Passing Neural Networks.  
* Learn **weights per neighbour** → Graph Attention Networks.  
* Enforce **geometric symmetries** → SE(3)-equivariant GNNs.  

Understanding the humble GCN therefore unlocks the door to the entire family of graph-based models used in state-of-the-art physical simulators.

---

## 9. Key take-aways  

1. A GCN layer blends each node’s features with a **degree-normalised average** of its neighbours.  
2. Self-loops and symmetric normalisation keep training stable.  
3. Stacking $K$ layers lets information travel $K$-hops – analogous to widening the receptive field in CNNs.  
4. Simplicity is both strength (few parameters) and weakness (uniform, isotropic weighting).  

---

## References  

[44] Math Behind Graph Neural Networks – Rishabh Anand.  
[47] Graph convolutional neural networks – Matthew N. Bernstein.  
[50] Graph Convolutional Networks (GCN) – TOPBOTS.  
[52] Graph Convolutional Networks (GCNs): Architectural Insights and Applications – GeeksforGeeks.  
[55] Graph Neural Networks Part 2: Graph Attention Networks vs GCNs – Towards Data Science.  

[44]: https://rish-16.github.io/posts/gnn-math/
[47]: https://mbernste.github.io/posts/gcn/
[50]: https://www.topbots.com/graph-convolutional-networks/
[52]: https://www.geeksforgeeks.org/graph-convolutional-networks-gcns-architectural-insights-and-applications/
[55]: https://towardsdatascience.com/graph-neural-networks-part-2-graph-attention-networks-vs-gcns-029efd7a1d92/