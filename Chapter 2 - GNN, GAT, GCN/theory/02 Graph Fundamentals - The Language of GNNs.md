# Graph Fundamentals: The Language of GNNs

To effectively utilize Graph Neural Networks, a solid understanding of fundamental graph theory concepts is essential. These concepts provide the vocabulary and framework for how GNNs represent and process data.  

## Nodes, Edges, and Graphs: Core Definitions and Types

A graph, in its formal definition, is an ordered pair $G=(V,E)$, where $V$ represents a set of vertices (or nodes), and $E$ represents a set of edges (or links) that denote connections between pairs of these vertices [15]. This abstract structure allows for the modeling of relationships between various entities.

**Nodes (Vertices)** are the fundamental units of a graph. They represent individual entities, data points, or locations within the modeled system [15]. Each node can be assigned a label or remain unlabeled, depending on the specific application [40]. For instance, in a social network, each person would be a node; in a molecular structure, each atom would be a node.

**Edges (Links/Arcs)** are the connections or relationships that exist between pairs of nodes [15]. Edges can possess different characteristics:  
• **Directed Edges:** These indicate a one-way relationship, where the connection flows from one node to another. An example is a “follows” relationship on Twitter, where user A follows user B, but B does not necessarily follow A back [15].  
• **Undirected Edges:** These represent a two-way, mutual relationship. A friendship on Facebook is an example, where if A is friends with B, then B is also friends with A [15].  
• **Weighted Edges:** Edges can have numerical values associated with them, known as weights, which indicate the strength, cost, or capacity of the connection. For example, in a transportation network, the weight of an edge might represent the distance or travel time between two cities [15].

Graphs are highly versatile and can model a wide array of real-world systems. Examples include social networks (representing friendships or interactions), molecular structures (atoms as nodes, bonds as edges), citation networks (papers citing each other), transportation systems (cities as nodes, roads/routes as edges), and, critically for this discussion, particle systems in physics simulations [15].

The versatility of graph structures to model diverse real-world systems, ranging from abstract relationships to concrete physical interactions, underscores their increasing relevance in machine learning, particularly where traditional grid-based or sequential representations prove inadequate. The underlying observation is that graphs can effectively represent complex systems like social networks, molecules, transportation infrastructure, and physical particle systems [15].

## Representing Data on Graphs: Node, Edge, and Graph-Level Features

Beyond their fundamental structure, graphs become truly informative when endowed with features—attributes or properties associated with their various components. These features are typically encoded as numerical vectors, making them amenable to machine learning algorithms [22].

### Node Features (Node Attributes)

Node features are properties specific to individual entities represented by nodes. These features provide local information about each element in the system.  
• **Examples:** In a social network graph, node features for a user might include their age, gender, or country of residence [22]. In a molecular graph, an atom node could have features describing its chemical type, charge, or spatial coordinates [37]. For particle-based fluid simulations, node features are crucial and would typically include a particle’s current position $(x, y, [z])$, its current velocity $(v_x, v_y, [v_z])$, its mass, and its particle type (e.g., fluid, boundary, rigid body), often represented using one-hot encoding or learned embeddings. Other relevant physical properties such as density, pressure, or temperature could also be included.  
• **Role in GNNs:** These node features serve as the initial input to the first GNN layer. As information propagates through the network, these features are iteratively updated and enriched through the message-passing mechanism, forming learned node embeddings that capture both intrinsic properties and contextual information from the node’s neighborhood [22].

### Edge Features (Edge Attributes)

Edge features are properties that describe the relationships between entities, residing on the edges connecting nodes. These features allow the model to understand the nature or strength of interactions.  
• **Examples:** In a social network, an edge feature might represent the strength or duration of a friendship. In a molecular graph, edge features could describe the bond type (single, double, triple) or bond order between two atoms. For fluid simulations, edge features are particularly useful for capturing the geometric relationship between interacting particles. These could include the relative displacement vector between particles $(\Delta x, \Delta y, )$ or the Euclidean distance between them. Normalized versions of these features or other interaction-specific properties, such as normal vectors for boundary interactions, can also be used.  
• **Role in GNNs:** Edge features can be directly incorporated into the message-passing mechanism. This allows the GNN to learn more nuanced interactions, as the messages exchanged between nodes are not only based on node properties but also on the characteristics of their connection [5].

### Graph-Level Features

Graph-level features represent properties or characteristics of the entire graph structure. These are typically derived from aggregating information across all nodes and edges.  
• **Examples:** For a molecular graph, graph-level features could be its overall toxicity, solubility, or a prediction of its aroma [15]. In a fluid simulation, a graph-level feature might represent the total energy of the fluid system or its overall volume.  
• **Role in GNNs:** Graph-level features are typically obtained by applying a “readout” function (e.g., sum, mean, max pooling, or attention pooling) over the final node and/or edge embeddings. These aggregated representations are then used for graph classification or regression tasks, where the goal is to predict a property of the entire system [22].

The ability of GNNs to process node, edge, and potentially graph-level features simultaneously, unlike traditional neural networks that often flatten or ignore relational data, is a direct enabler for learning complex, multi-faceted physical phenomena [15].

## Adjacency Matrix and Other Graph Representations

Graphs can be stored and manipulated in computers using various data structures, each with its own advantages and disadvantages regarding memory efficiency and operational speed.

**The Adjacency Matrix** is a common representation for graphs. It is a square matrix of size $N \times N$, where $N$ is the number of nodes in the graph [15]. An entry $A[i][j]$ in the matrix indicates the presence or absence of an edge between node $i$ and node $j$. For unweighted graphs, $A[i][j]$ is typically 1 if an edge exists and 0 otherwise. For weighted graphs, $A[i][j]$ can store the weight of the edge [40]. In an undirected graph, the adjacency matrix is symmetric ($A[i][j]=A[j][i]$), while for directed graphs, it is asymmetric. A key limitation of the adjacency matrix is its potential for high memory usage, especially for large graphs that are sparse. In such cases, the matrix will contain many zeros, leading to inefficient storage [43]. It also does not inherently capture node or edge features without additional, separate matrices.

**The Adjacency List** offers a more memory-efficient alternative for sparse graphs. In this representation, each node has a list (or array) of its direct neighbors [40]. For example, if node A is connected to nodes B and C, its adjacency list entry would be A:. This structure is highly efficient for adding or removing edges, typically performing in $O(1)$ constant time. It is also efficient for iterating over a node’s neighbors. However, checking if an edge exists between two arbitrary nodes can be less efficient, potentially requiring a search through a list proportional to the node’s degree [40].

**The Edge List** is another straightforward representation, consisting of a simple list of pairs, where each pair $(u, v)$ denotes an edge between node $u$ and node $v$ [43]. This is often the most basic way to represent graph connectivity.

Modern graph deep learning libraries, such as PyTorch Geometric (PyG) and Deep Graph Library (DGL), provide sophisticated abstractions for working with graph data. These libraries often internally optimize between these representations based on the sparsity of the graph and the specific operations being performed, abstracting away much of the low-level management from the user.

The choice of graph representation, such as an adjacency matrix versus an adjacency list, has direct practical implications for computational efficiency and memory usage in GNN implementations, particularly for large-scale particle simulations. Adjacency matrices can be memory-intensive and often sparse for large graphs, leading to inefficient storage and computation, as a significant portion of the matrix would be filled with zeros [43]. Conversely, adjacency lists are considerably more memory-efficient for sparse graphs, as they only store existing connections [40]. For particle systems, especially in fluid dynamics, the number of particles (nodes) can be very large, but each particle typically interacts only with a relatively small, local neighborhood. This results in a highly sparse connectivity pattern.

### Table: Comparison of Graph Representations

| Representation Type | Description | Space Complexity (Dense) | Space Complexity (Sparse) | Add Edge | Remove Edge | Check Edge | Iterate Neighbors | Typical Use Cases/Advantages | Disadvantages |
|---------------------|-------------|--------------------------|---------------------------|----------|-------------|------------|-------------------|------------------------------|---------------|
| Adjacency Matrix | A square matrix where $A[i][j]$ indicates an edge (or its weight) between node $i$ and $j$. | $O(N^{2})$ | $O(N^{2})$ | $O(1)$ | $O(1)$ | $O(1)$ | $O(N)$ | Quick edge lookup, simple for dense graphs. | High memory for sparse graphs, slow for iterating all neighbors. |
| Adjacency List | Each node has a list of its direct neighbors. | N/A | $O(N+E)$ | $O(1)$ | $O(\text{degree})$ | $O(\text{degree})$ | $O(\text{degree})$ | Memory efficient for sparse graphs, efficient for neighbor traversal. | Slower edge lookup, can be complex to implement. |
| Edge List | A simple list of pairs $(u, v)$ representing each edge. | N/A | $O(E)$ | $O(1)$ (append) | $O(E)$ | $O(E)$ | $O(E)$ | Simplest to store, good for initial data loading. | Inefficient for most graph operations (neighbor lookup, etc.). |

*N: number of nodes, E: number of edges, degree: number of neighbors for a specific node.*

---

The versatility of these representations, along with their respective trade-offs, highlights the importance of selecting an appropriate data structure tailored to the graph’s sparsity and the specific computational requirements of the GNN application.

## Sources

[5]: https://community.arm.com/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/physics-simulation-graph-neural-networks-targeting-mobile  
[15]: https://viso.ai/deep-learning/graph-neural-networks/  
[22]: https://karthick.ai/blog/2024/Graph-Neural-Network/  
[37]: https://cs.stanford.edu/people/jure/pubs/pretrain-iclr20.pdf  
[40]: https://www.geeksforgeeks.org/introduction-to-graphs-data-structure-and-algorithm-tutorials/  
[43]: https://algodaily.com/lessons/implementing-graphs-edge-list-adjacency-list-adjacency-matrix  
[45]: http://proceedings.mlr.press/v129/yang20a/yang20a.pdf