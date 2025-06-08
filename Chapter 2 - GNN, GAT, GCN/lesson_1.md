# Lesson 1 – Motivation: Why Graph Neural Networks for Physics?

> “If you keep trying to fit an ocean into a spreadsheet,  
>  sooner or later you will run out of columns.”

Traditional deep-learning work-horses such as Multi-Layer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs) flourish when the data look like fixed-length vectors or regular grids.  
Physical systems—fluids, plasmas, granular media, molecular assemblies—rarely comply.  
Before we dive into the mechanics of Graph Neural Networks (GNNs), we need to understand **why the usual tools stumble** and what, exactly, GNNs bring to the table.

---

## 1. What Makes Physical Systems “Difficult” for Standard NNs?

1. **Irregular Geometry**  
   Particles (or atoms, or stars…) occupy arbitrary positions in space; there is no natural pixel-grid.
2. **Variable Entity Count**  
   A simulation step can spawn or delete particles (think of breaking waves or combustion by-products).
3. **Rich, Pairwise (or Higher-Order) Interactions**  
   Forces depend on *who* is close to *whom* and change continuously as the configuration evolves.
4. **Hard Physical Constraints**  
   Conservation laws (mass, momentum, energy) are non-negotiable; predictions that violate them are useless.

These traits imply that *relationships* matter more than isolated features—exactly where vector- or grid-centric architectures suffer.

---

## 2. The Case Against MLPs

MLPs treat their input as a single, flat vector.

* **Fixed size** If the particle count changes, you must pad, crop, or re-order—each destroys information.  
* **Order sensitivity** Swapping particles changes the input vector and therefore the output, despite the physics being identical.  
* **Lost geometry** Distances and directions must be hand–engineered into the vector; the network does **not** know that index 17 and index 312 were spatial neighbours.

Even with infinite capacity, approximation error for an MLP on smooth functions shrinks only as  

$$
\varepsilon(n)\;=\;O\!\left(n^{-1}\right) \tag{1}
$$

where $n$ is the number of hidden units [1].  
That does *not* rescue you from poor inductive bias: the network is still blind to particle permutations or locality.

---

## 3. The Case Against CNNs

CNNs shine on images because a small convolutional kernel exploits **translation equivariance**:

$$
f\bigl(T_{\mathbf{\delta}}\,x\bigr) \;=\; T_{\mathbf{\delta}}\,f(x) \tag{2}
$$

Shift the picture → shift the features; nothing else changes [3].

For unstructured particles, however:

* **No grid, no kernel** There is no well-defined “left neighbour” or “pixel (i,j)”.  
* **Scale & orientation variance** Inter-particle distances span several orders of magnitude; the same kernel width cannot cover all.  
* **Pooling trouble** Max/average pooling builds *translation invariance* by discarding precise positions—fatal when you actually need them [7].  

CNNs can *emulate* irregular domains by voxelising space, but:
* you must pick a grid resolution (aliasing vs. memory blow-up),  
* empty voxels dominate memory,  
* fine geometric detail is blurred.

---

## 4. The Missing Piece: Incorporating Physics

Beyond geometry, classic architectures struggle with domain knowledge:

* They learn dynamics *only* from data; enforcing conservation laws or symmetries requires ad-hoc losses or post-processing [9].  
* Training data are expensive—obtained from high-resolution simulations or costly experiments—yet MLPs/CNNs frequently demand millions of diverse samples [10].

---

## 5. A (Brief) Look Ahead at Graph Neural Networks

Graphs give each particle its *own* node and wire edges only to actual neighbours.  
GNNs then:

* **Respect variable size** Add or delete nodes on the fly.  
* **Are permutation-equivariant** Re-labelling nodes merely re-orders outputs.  
* **Transmit information locally, iteratively** Akin to how physical influences propagate.  
* **Embed physics priors** Neighbourhood aggregation, edge features (distance, direction), and conservation-aware losses come naturally.

We will formalise these ideas in the next lessons; for now, remember the high-level moral:

> **When relationships drive the dynamics, the model must make relationships first-class citizens.**

---

## 6. Analogy Corner  🔍

**Spreadsheet vs. Social Network**

*MLP/CNN world* Trying to capture a crowd’s behaviour in a spreadsheet: each row is a person, and you log their latest GPS coordinate.  
If Alice and Bob swap rows, your “MLP” thinks the world has changed.  
Adding a new visitor forces you to add a row and retrain formulas.

*GNN world* Think instead of a social network graph: nodes are people, edges are who’s talking to whom.  
Add or remove people, shuffle the user-id numbers—*relationships* stay intact.  
A GNN reads the same graph regardless of spreadsheet order and updates only the altered neighbourhoods.

---

## 7. Key Take-aways

1. **Irregular, interacting data break the assumptions of MLPs (fixed vector) and CNNs (regular grid).**  
2. **Physical fidelity** requires architectures that *respect permutation symmetry, variable size, and local interactions*.  
3. **GNNs** are purpose-built to meet those requirements; the rest of this course will unpack *how*.

---


[1]: https://arxiv.org/html/2504.11397v1  
[2]: https://ar5iv.labs.arxiv.org/html/1805.00915  
[3]: https://blog.paperspace.com/pooling-and-translation-invariance-in-convolutional-neural-networks/  
[7]: https://milvus.io/ai-quick-reference/what-are-some-issues-with-convolutional-neural-networks  
[9]: https://www.mdpi.com/2673-2688/5/3/74  
[10]: https://arxiv.org/html/2504.08766v1