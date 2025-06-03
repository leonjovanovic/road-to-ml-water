# Lesson 6 – Graph Theory Basics: The Language of Graph Neural Networks  
*(course: Introduction to GNNs for Physical Systems)*

---

## 6.1  Why We Need Graphs at All
Before we can let a Graph Neural Network (GNN) crunch on data, we must decide **how to describe that data as a graph**.  Traditional tensors (images, sequences, etc.) assume regular ordering; graphs remove that assumption and let us work with *arbitrary* connectivity.  
Think of graphs as the **blueprints** that tell the GNN what can talk to what.

*Real-life anchor* –  
*Social media*: every user is a node, a “follow” relation is an edge.  
*Fluid simulation*: every particle is a node, proximity defines an edge.  
The mathematics behind both is identical; only the **meaning** differs.

---

## 6.2  Formal Definition

A (simple) graph is an ordered pair  

\[
G = (V, E)
\]

* **\(V\)** – a finite set of *vertices* (or *nodes*).  
* **\(E\)** – a set of *edges*; each edge connects one or two vertices. [15]

If the context demands it, we enrich this definition with *attributes* (node features, edge features, global features).  
For now we stick to bare structure.

---

## 6.3  Nodes (Vertices)

*What they are*: the fundamental entities we want to reason about.  
*Examples*  
* Person in a social graph  
* Atom in a molecule  
* Particle in SPH fluid simulation [24]

Nodes may carry **labels/features** (age, atomic number, velocity, …).  
In raw graph theory these labels are optional, but in GNNs they are almost always present.

---

## 6.4  Edges (Links)

### 6.4.1  Directed vs Undirected  
* **Directed**  – ordered pair \((u,v)\); information can flow one way (Twitter *follow*) [15].  
* **Undirected** – unordered set \(\{u,v\}\); relation is symmetric (Facebook *friend*).

For GNNs, the direction matters because it decides which messages are exchanged during *message passing*.

### 6.4.2  Weighted Edges  
An edge weight \(w_{uv}\in\mathbb{R}\) quantifies strength, distance or cost (road length between two cities).  
In a **weighted adjacency matrix**

\[
A_{uv} =
\begin{cases}
w_{uv} & \text{if an edge exists} \\
0      & \text{otherwise}
\end{cases}
\]

the same structure stores both topology and magnitude [15].

### 6.4.3  Self-loops & Multigraphs (optional, but common in GNNs)  
* **Self-loop**: edge from a node to itself; often added intentionally (\(A \leftarrow A+I\)) so a GNN can mix a node’s own features with messages from neighbours (GCN normalization, Lesson 12).  
* **Multigraph**: multiple parallel edges (e.g. different chemical bonds between same atoms).

---

## 6.5  Graph Representations in Code

| Representation | Memory (dense) | Memory (sparse) | Lookup \(A_{ij}\) | Iterate neighbors | Typical use | Notes |
|----------------|---------------|-----------------|-------------------|-------------------|-------------|-------|
| **Adjacency matrix** | \(O(N^2)\) | same | \(O(1)\) | \(O(N)\) | dense graphs, matrix algebra | wasteful when graph is sparse [43] |
| **Adjacency list**   | — | \(O(N+E)\) | \(O(\text{deg}(i))\) | \(O(\text{deg}(i))\) | large sparse graphs | default in PyG/DGL [40] |
| **Edge list**        | — | \(O(E)\) | \(O(E)\) | \(O(E)\) | data import/export | simplest to store |

Where  
\(N=|V|\),  \(E=|E|\),  \(\text{deg}(i)\) = degree of node \(i\).

*Practical tip*: Particle simulations are extremely sparse—each particle interacts with only a handful of neighbours—so adjacency lists plus **sparse tensors** are the norm in modern GNN libraries.

---

## 6.6  Degrees and the Degree Matrix

For node \(i\)

\[
d_i \;=\;\sum_{j} A_{ij}
\]

and  

\[
D = \operatorname{diag}(d_1,\dots,d_N)
\]

The degree often appears in normalized operators (GCN: \(\tilde A = D^{-1/2}(A+I)D^{-1/2}\)).  Intuitively, this rescales the contribution of each neighbour so high-degree nodes do not dominate the aggregation.

---

## 6.7  Putting It All Together – From Abstraction to Physics

1. **Choose nodes** – every SPH particle becomes a vertex.  
2. **Define edges** – connect particles within smoothing length \(h\).  
3. **Pick storage** – adjacency list for speed; build it every timestep using spatial hashing.  
4. **Attach features** – position, velocity, mass, type.  
5. **Feed into GNN** – the graph now tells the network *who can talk to whom*; the features tell it *what to talk about*.

*Analogy* –  
Imagine a group chat where only neighbours within Bluetooth range can communicate.  The **participant list** is \(V\); the **Bluetooth links** are \(E\).  Step out of range, and your edge disappears in the next timestep.

---

## 6.8  Key Take-aways

* Graphs generalise grids; they remove the assumption of regular ordering.  
* Vertices and edges are enough to model social media, molecules **and** fluids.  
* Different edge types (direction, weight) encode different physical semantics.  
* Choice of data structure (matrix vs list) can make or break scalability.  
* Degrees and adjacency matrices later re-appear in GNN layers (GCN, GAT).

Next lesson we dive into **multi-level features** and see how raw graphs become *rich* inputs for a GNN.

---

## 6.9  Sources  
[15] viso.ai – “Graph Neural Networks (GNNs) – Comprehensive Guide”  
[24] EPCC – “Accelerating Smoothed Particle Hydrodynamics with GNNs”  
[40] GeeksForGeeks – “Introduction to Graph Data Structure”  
[41] DataCamp – “Introduction to Graph Theory”  
[43] AlgoDaily – “Implementing Graphs: Edge List, Adjacency List, Adjacency Matrix”