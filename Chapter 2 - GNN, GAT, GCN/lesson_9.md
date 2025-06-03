# Lesson 9 – Message-Passing Neural Networks (MPNNs)  
*(The Information-Flow Blueprint of GNNs)*

---

## 1. Why “message passing” at all?

When a physical phenomenon is discretised into particles (or any interacting entities), the **local state of one particle changes because of its neighbours**.  
Graph Neural Networks capture this intuition literally: information is passed along graph edges in the same way that

* heat diffuses among molecules, or  
* rumours propagate through a social network.  

Formally, almost every modern GNN layer can be placed inside the **MPNN framework** introduced by Gilmer et al. (2017). It provides a clean recipe that we will follow again and again in later lessons [5], [14], [29].

---

## 2. Notation

| Symbol | Meaning |
|:------:|---------|
| \(G=(V,E)\) | graph with nodes \(V\) and edges \(E\) |
| \(h_v^{(k)}\) | feature (embedding) of node \(v\) at layer \(k\) |
| \(e_{uv}\)    | (optional) feature on edge \((u,v)\) |
| \(\mathcal N(v)\) | neighbours of \(v\) |
| \(M^{(k)}(\cdot)\) | learnt **message** function |
| \(\text{AGG}^{(k)}\) | permutation-invariant **aggregation** |
| \(U^{(k)}(\cdot)\) | learnt **update** function |

---

## 3. The three core steps

### 3.1 Message

Each edge sends a vector‐valued message from \(u\) to \(v\):

\[
m_{u\!\rightarrow\! v}^{(k)} \;=\; 
M^{(k)}\!\!\bigl(h_u^{(k-1)},\,h_v^{(k-1)},\,e_{uv}\bigr)
\tag{1}
\]

*Typical choices*: an MLP, a linear layer, or even a simple element-wise product when speed is crucial [5], [22].

### 3.2 Aggregate  _(permutation invariance is mandatory!)_

\[
\tilde m_v^{(k)} \;=\;
\text{AGG}^{(k)}\!\bigl\{\,m_{u\!\rightarrow\! v}^{(k)} \;\big|\; u\in\mathcal N(v)\bigr\}
\tag{2}
\]

Common aggregators—**sum, mean, max, attention—will be dissected in Lesson 10.**  
Whatever you pick, Equation (2) must give the **same result no matter how the neighbours are ordered**, ensuring permutation equivariance [14].

### 3.3 Update

\[
h_v^{(k)} \;=\;
U^{(k)}\!\bigl(h_v^{(k-1)},\,\tilde m_v^{(k)}\bigr)
\tag{3}
\]

Often \(U^{(k)}\) is another MLP followed by a non-linearity (ReLU/ELU).  
Some physics-oriented papers replace it with GRU/LSTM gates to capture longer temporal dependencies [22].

---

## 4. Multi-hop propagation & “graph receptive field”

One MPNN layer lets information travel **one hop**.  
After \(K\) layers, node \(v\) has seen everything in its \(K\)-hop neighbourhood:

\[
h_v^{(K)} \longrightarrow \text{context of radius } K
\]

Analogy: drop a stone in a pond—ripples spread outward hop by hop. The deeper the network, the larger the “ripples”, but also the higher the risk of *over-smoothing* (Lesson 14 will cover this).

---

## 5. Worked example – “Predicting temperature in a metal plate”

Imagine a thin metal plate represented by a graph whose nodes are lattice points, edges connect immediate neighbours.

1. **Initial features** \(h_v^{(0)}\): current temperature at each point.  
2. **Message**: each node sends its temperature to neighbours.  
3. **Aggregate**: a node averages incoming temperatures.  
4. **Update**: new state is a weighted mix of its own and aggregated temps, approximating the heat equation.

Even with this naive configuration an MPNN emulates finite-difference heat diffusion; adding learnable parameters lets it rediscover (and accelerate) the PDE solution [5].

*(This example also hints why GNN surrogates can be many times faster than traditional solvers while remaining faithful to physics.)*

---

## 6. Implementation sketch (PyTorch Geometric)

```python
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class SimpleMPNN(MessagePassing):
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='mean')          #   <-- Aggregator
        self.mlp = torch.nn.Sequential(        #   <-- Message & Update share params
            torch.nn.Linear(2*in_dim, out_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, edge_index):
        # x: [num_nodes, in_dim]
        edge_index, _ = add_self_loops(edge_index)
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):               # i ← j
        return self.mlp(torch.cat([x_i, x_j], dim=-1))

    def update(self, aggr_out, x):
        return self.mlp(torch.cat([x, aggr_out], dim=-1))
```

Lines `message` and `update` directly implement Eqs. (1) – (3).

---

## 7. Design knobs & best practices

| Decision | Practical considerations |
|----------|--------------------------|
| Message function \(M\) | Should reflect the *type* of interaction (e.g., subtract positions for relative vectors). |
| Edge features \(e_{uv}\) | Crucial in physics; encode distance, direction, material boundaries. |
| Aggregator choice | Sum for forces, mean for feature normalisation, attention for anisotropic effects. |
| Number of layers \(K\) | Related to physical interaction range; too small ⇒ misses long-range effects, too large ⇒ over-smoothing. |
| Parameter sharing | Weights \(M^{(k)},U^{(k)}\) can be shared across layers (GraphSage-style) or distinct. |

---

## 8. Key take-aways

1. **MPNN is the backbone of almost all GNN variants.**  
2. The triplet *(message, aggregate, update)* mirrors how real physical influence propagates.  
3. Ensuring **permutation invariance** in aggregation keeps models physically sane.  
4. Layer depth ↔ reachable interaction distance.  
5. Careful design of message and edge features lets the network *rediscover* known laws (e.g., Newton’s, Navier–Stokes) with fewer data.

---

## References  

[^5]: Physics Simulation With Graph Neural Networks Targeting Mobile – Arm Community (2024) <https://community.arm.com/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/physics-simulation-graph-neural-networks-targeting-mobile>  
[^14]: “The Graph Neural Network Model” – W. L. Hamilton *et al.* (GRL Book, 2020) <https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book-Chapter_5-GNNs.pdf>  
[^22]: “Graph Neural Network: In a Nutshell” – K. P. Selvam (2024) <https://karthick.ai/blog/2024/Graph-Neural-Network/>  
[^29]: “Understanding Message Passing in GNNs” – E. Benjaminson (2023) <https://sassafras13.github.io/GNN/>  
