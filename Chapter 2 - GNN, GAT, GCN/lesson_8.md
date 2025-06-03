# Lesson 8 Storing Graphs Efficiently  
Adjacency Matrix • Adjacency List • Edge List Trade-offs
---

Graph Neural Networks (GNNs) are only as fast and memory-friendly as the data structures they live on.  
In classical deep-learning libraries a single image tensor fits neatly into a dense 4-D array; in graph ML we must describe **which node talks to which other node**.  
Choosing the wrong structure can turn a lightning-fast model into a memory hog that never leaves the GPU queue.  

> Real-life analogy  
> • **Edge List** A one–column “guest book” at a wedding – every new friendship is just another line.  
> • **Adjacency List** Each guest has her own contact list in a phone.  
> • **Adjacency Matrix** You print a huge table with *every* pair of guests and tick a box if they know each other.

Below we formalise these three options, analyse their complexity, and discuss which one modern GNN libraries adopt by default.

---

## 1 Notation and Problem Size

* $N$ = number of nodes (particles, atoms, users …)  
* $E$ = number of edges  
* $\text{deg}(v)$ = degree of node $v$ (how many neighbours it has)

For fluid or molecular simulations we often observe:

* $E \approx N \times \text{const}$ (each particle only interacts with a local neighbourhood)  
* $\displaystyle \frac{E}{N^2} \ll 1 \quad$ → **very sparse graphs**

---

## 2 Adjacency Matrix

### Definition  

A square matrix $\mathbf A \in \{0,1\}^{N \times N}$ (or real-valued if edges are weighted):

```math
A_{ij} \;=\;
\begin{cases}
1 & \text{if there is an edge } (i,j)\\[6pt]
0 & \text{otherwise}
\end{cases}
```

### Characteristics  

| Aspect | Value |
|--------|-------|
| **Space (dense)** | $\mathcal O(N^{2})$ |
| **Add / remove edge** | $\mathcal O(1)$ |
| **Check edge $(i,j)$** | $\mathcal O(1)$ |
| **Iterate over neighbours of $i$** | $\mathcal O(N)$ |

### When is it useful?  

* **Very small or very dense** graphs  
* Spectral GNN variants that need $\mathbf D^{-1/2}(\mathbf A + \mathbf I)\mathbf D^{-1/2}$ pre-computed in one shot  

### Pitfall  

For $N=10^5$ particles the matrix holds $10^{10}$ entries – > 80 GB in single precision!

[15]

---

## 3 Adjacency List

### Definition  

A Python-like pseudo code representation:

```python
adj = {
    0: [2, 7, 13],
    1: [5],
    2: [0, 4],
    ...
}
```

Each key is a node, the value is a dynamic array of direct neighbours.

### Complexity  

| Aspect | Value |
|--------|-------|
| **Space** | $\mathcal O(N + E)$ |
| **Add edge** | $\mathcal O(1)$ (amortised) |
| **Remove or check edge $(i,j)$** | $\mathcal O(\text{deg}(i))$ |
| **Iterate neighbours** | $\mathcal O(\text{deg}(i))$ |

Because $\text{deg}(i)$ is small in physical simulations, all critical operations are effectively constant time.

### Why GNN toolkits love it  

PyTorch Geometric and DGL internally translate this list into two flat tensors  
`edge_index = [src, dst]` (shape `2 × E`) – a GPU-friendly edge list variant, but lookup semantics of adjacency lists.  

[40]

---

## 4 Edge List

### Definition  

Simply a list of all pairs (optionally with weights):

```python
edges = [
    (0, 2), (0, 7), (0, 13),
    (1, 5),
    (2, 0), (2, 4),
    ...
]
```

### Complexity  

| Aspect | Value |
|--------|-------|
| **Space** | $\mathcal O(E)$ |
| **Append edge** | $\mathcal O(1)$ |
| **Check edge** | $\mathcal O(E)$ (linear search) |
| **Iterate neighbours** | $\mathcal O(E)$ unless additionally indexed |

### Typical use

* **Data loading / serialisation** (CSV, JSON)  
* Passing a *static* edge list to a GNN layer that does not need random edge look-ups  

[43]

---

## 5 Putting Numbers on It

Imagine an SPH simulation with  

* $N = 50\,000$ fluid particles  
* average $\text{deg}=30$ (kernel radius) → $E \approx 1.5\,\text{M}$ edges

| Representation | Memory Footprint (float32) |
|----------------|----------------------------|
| **Adjacency Matrix** | $N^2 \times 4\text{ B} \;≈\; 10\,\text{GB}$ |
| **Adjacency List** | $(N + E) \times 4\text{ B} \;≈\; 6.0 MB$ |
| **Edge List** | $2E \times 4\text{ B} \;≈\; 12 MB$ |

Adjacency lists beat the matrix by three orders of magnitude – the difference between fitting on-chip or paging to disk.

---

## 6 Sparse Matrices & GPU Kernels

While the **API** may expose an adjacency list, the heavy linear algebra inside a GNN layer often uses a *sparse COO tensor*:

```math
\mathbf{H}^{(k+1)} = \sigma\!\bigl( \mathbf{\hat A} \mathbf{H}^{(k)} \mathbf{W}^{(k)} \bigr)
```

where $\mathbf{\hat A}$ stores only the non-zero indices and their values.  
Libraries such as **cuSPARSE** exploit this to multiply in $\mathcal O(N + E)$ instead of $N^2$.

---

## 7 Guidelines for Practice

1. **Physics = sparsity** ⇒ favour adjacency/edge list formats.  
2. **Dynamic graphs** (particles move every time-step):  
   * Keep node order fixed → only rebuild `edge_index`.  
   * Use spatial hashing or k-d trees to cut neighbour search from $N^2$ to $N\log N$.  
3. **Batched mini-graphs**: concatenate `edge_index` and add an *offset* to node indices – PyG does this transparently.  
4. **Need fast edge queries?** Build a *hybrid*: edge list for GPU kernel, hash map for CPU queries.  

---

## 8 Take-away

Efficient storage is not a boring back-office detail; it is the enabler that lets your GNN push millions of particles through a single RTX-4090 in real time.  
Grasp these trade-offs now and you will know exactly where to look when your “clever” model runs out of memory.

---

## References  

[15] Graph Neural Networks (GNNs) – Comprehensive Guide. https://viso.ai/deep-learning/graph-neural-networks/  
[40] GeeksforGeeks. “Introduction to Graph Data Structure.” https://www.geeksforgeeks.org/introduction-to-graphs-data-structure-and-algorithm-tutorials/  
[43] AlgoDaily. “Implementing Graphs: Edge List, Adjacency List, Adjacency Matrix.” https://algodaily.com/lessons/implementing-graphs-edge-list-adjacency-list-adjacency-matrix  
