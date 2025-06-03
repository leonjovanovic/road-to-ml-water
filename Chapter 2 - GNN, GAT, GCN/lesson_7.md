
# Lesson 7 – Multi-Level Features: Designing **Node**, **Edge** & **Graph** Attributes  

## 1. Why “multi-level” matters
Classic ML pipelines often treat every input as a flat vector.  
Graphs are richer: information lives **on the entities themselves, on their pairwise
relationships, and sometimes on the whole structure**.  
Failing to expose any of these layers to the network forces it to _infer_ them from
scratch—wasting capacity and data.

<img src="https://i.imgur.com/Xg47mZY.png" width="640"/>

*Analogy.*  
Think of running a delivery company:
* every **driver** (node) has a current location and package load,  
* every **road** (edge) has a travel time and toll,  
* the **whole map** (graph) has a weather alert.  
Optimal routing clearly needs all three pieces.

---

## 2. Node features \( \mathbf{x}_v \)

| Domain                    | Typical node attributes                           |
|---------------------------|---------------------------------------------------|
| Social network            | age, interests, #followers                        |
| Molecule                  | atom type, charge, 3-D coordinates                |
| Lagrangian fluid sim (SPH)| position \((x,y,z)\), velocity \((v_x,v_y,v_z)\), mass, phase flag |

### 2.1 Mathematical view  
All node attributes are stacked row-wise in a matrix  

\[
\mathbf{X}\in\mathbb{R}^{N \times D_{\text{node}}}
\]

where \(N\) = #nodes and \(D_{\text{node}}\) = feature dimension [22].

### 2.2 Good practice check-list  
1. **Physical units** – keep them explicit or normalised consistently.  
2. **Categoricals** – use one-hot or learned embedding.  
3. **Scale invariance** – velocities of order 10⁴ handle differently from positions of order 10⁻².

---

## 3. Edge features \( \mathbf{e}_{ij} \)

Edges carry the semantics of interactions.

### 3.1 Geometry first  
For particle \(i\) and neighbour \(j\)

\[
\Delta\mathbf{x}_{ij} = \mathbf{x}_j - \mathbf{x}_i,\qquad  
d_{ij}= \lVert \Delta\mathbf{x}_{ij}\rVert_2
\]

These two numbers alone supply direction **and** magnitude of the pairwise relation [45].

### 3.2 Domain-specific extras  
* **Molecules** – bond type (single, double, aromatic).  
* **Computer networks** – bandwidth, latency.  
* **Fluid boundaries** – surface normal, restitution coefficient.

Edge attributes are stored in  

\[
\mathbf{E}\in\mathbb{R}^{|\mathcal{E}| \times D_{\text{edge}}}
\]

where \(|\mathcal{E}|\) is #edges.

> **Tip (stability).**  
> If distances vary across orders of magnitude, feed the _normalised_ value  
> \( \hat d_{ij}=d_{ij}/r_{\text{cutoff}} \) to avoid vanishing gradients.

---

## 4. Graph-level features \( \mathbf{g} \)

Sometimes the entire graph has a label: total energy, toxicity, weather alert level…
We obtain a single embedding via a **readout**:

\[
\mathbf{h}_G = \text{READOUT}\Big(\{\mathbf{h}_v^{(L)}\}_{v\in V}\Big)
\]

Common readouts: **sum, mean, max pooling, attention pooling** [22].

For regression you may append an MLP  
\( \hat y = \text{MLP}(\mathbf{h}_G) \).

---

## 5. Putting the tensors together

```python
# PyTorch Geometric style
data = Data(x=node_feat_matrix,          # N × D_node
            edge_index=edge_index_long, # 2 × |E|
            edge_attr=edge_feat_matrix, # |E| × D_edge
            y=graph_label)              # optional
```

> **Shortcut for physicists.**  
> You can store positions only once (in `x`) and recompute Δx, d on-the-fly in
> the GNN’s message function, saving GPU memory.

---

## 6. Inductive bias & expressivity

* Node attributes ≙ _local state_  
* Edge attributes ≙ _interaction rules_  
* Graph attributes ≙ _global constraints_

By exposing all three, you inject **relational inductive bias** [32];  
the network spends capacity on _how_ particles
talk to each other, not on _whether_ they can.

---

## 7. Example: predicting splash height

Goal: given an SPH snapshot, predict the future maximum splash height.

1. **Nodes** – particle position, velocity, density.  
2. **Edges** – Δx, distance, neighbour type (fluid ↔ boundary).  
3. **Graph feat.** – fill level of the container.

A GNN aggregates to node embeddings, applies a **global readout**, then
regresses a single scalar height.  
Because both edge geometry and a global water-level cue are present, the
network learns _when_ and _where_ a wave turns into a splash.

---

## 8. Common pitfalls

| Pitfall                                  | Remedy |
|------------------------------------------|--------|
| Ignoring edges (‘fully-connected’ graph) | Dramatic O(N²) cost; restrict radius |
| Mixing units (m vs cm)                   | Normalise or learn separate scalings |
| Sparse features with many zeros          | Use embeddings or log-scaling         |
| Missing categorical edge types           | Encode as one-hot or embedding        |

---

## 9. Summary

Multi-level features are the **raw material** for message passing.
Design them with physics insight:

* Nodes = _what a particle knows about itself_  
* Edges = _how it speaks to neighbours_  
* Graphs = _the bigger rules of the game_

Well-chosen attributes turn a generic GNN into a
compact, data-efficient simulator.

---

## References

1. Graph Neural Network: In a Nutshell&nbsp;[22]  
2. Graph Neural Networks – Comprehensive Guide&nbsp;[15]  
3. Exploiting Edge Features in Graph Neural Networks&nbsp;[45]  
4. Lecture 11: Graph Neural Networks (inductive bias)&nbsp;[32]
