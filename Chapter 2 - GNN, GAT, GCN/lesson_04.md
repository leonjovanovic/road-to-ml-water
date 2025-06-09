# Lesson 4 – The Lagrangian View: Mapping Particle Dynamics to Graphs  

---

## 1.  Eulerian vs Lagrangian: Two Lenses on Motion  

| Perspective | What is “fixed”? | Typical CFD discretisation | Intuition |
|-------------|-----------------|----------------------------|-----------|
| **Eulerian** | Spatial grid cells | Finite-volume / finite-difference | Sit at a weather station and record how the wind blows past you. |
| **Lagrangian** | Material points (particles) | Smoothed-Particle Hydrodynamics (SPH) | Follow a leaf floating down a river. |

*Why we care*: GNNs excel when **entities interact locally and move freely**.  
That is exactly the setting of the Lagrangian picture.

---

## 2.  A 60-second Primer on SPH  

In SPH the fluid is represented by  _N_  particles.  
Key idea: physical fields are reconstructed from neighbours by a smoothing kernel $W$.

$$
\rho_i \;=\; \sum_{j} m_j\,W\bigl(\|\,\mathbf x_i-\mathbf x_j\|,\,h\bigr)
$$

* $ \rho_i $ – density at particle _i_  
* $ m_j $ – neighbour’s mass  
* $ h $ – smoothing length (the **connectivity radius**)  

All other forces (pressure, viscosity, surface tension …) are built from the same “sum‐over-neighbours” template [23].

---

## 3.  Why a Graph is the Natural Data Structure  

| SPH concept | Graph analogue |
|-------------|----------------|
| Particle | **Node** |
| “Within h” neighbour | **Edge** |
| Kernel-weighted interaction | **Message** |

The graph is *dynamic*: at every time-step we rebuild the edge set because particles move.  
Efficient spatial hashing or k-d trees are therefore as important as the neural net itself [24].

---

## 4.  Formalising the Mapping  

### 4.1  Node features  

* Position $ \mathbf x_i $  
* Velocity $ \mathbf v_i $  
* Mass $ m_i $  
* Type flag (fluid / boundary / rigid) → one-hot encoding  

### 4.2  Edge features  

For each edge $ (i,j) $:

$$
\mathbf e_{ij}=\bigl[\, \mathbf r_{ij},\;\|\mathbf r_{ij}\|\bigr], 
\qquad \mathbf r_{ij} = \mathbf x_j-\mathbf x_i
$$

Optional: pre-compute kernel weights or boundary normals.

### 4.3  Message-passing ≈ SPH operator  

A generic message-passing layer can be written as  

$$
\mathbf h_i^{(k+1)}
\;=\;
\phi\!\Bigl(
\mathbf h_i^{(k)},
\sum_{j\in\mathcal N(i)} 
\psi\bigl(\mathbf h_i^{(k)},\mathbf h_j^{(k)},\mathbf e_{ij}\bigr)
\Bigr)
$$

Compare this with the SPH summation formula above – structurally identical!  
The neural networks $ \psi $ and $ \phi $ simply learn **how** to weight and mix the neighbour information rather than using a hand-crafted kernel [25].

---

## 5.  Analogy Corner – “Gossiping Boids” 🦜  

Imagine a flock of birds (boids) that can only “gossip” with birds inside a small bubble around them.

1. Every second, each bird collects gossips from its nearby friends.  
2. It updates its own belief (e.g. where to fly next).  
3. The neighbourhood relationships change as the flock moves, so the gossip network is rebuilt every second.

Replace “gossip” with “force messages” and you have an SPH-to-GNN mapping.

---

## 6.  Why Use a GNN Instead of Direct SPH?  

| Goal | Traditional SPH | SPH → GNN surrogate |
|------|-----------------|---------------------|
| **Speed** | CPU/GPU heavy; Δt limited by stability | 10-100× faster roll-outs reported [5], [24] |
| **Learning effects** (e.g. turbulence) | Needs expensive sub-grid models | GNN learns them from data |
| **Portability** | Desktop/cluster | Real-time on mobile [5] |

The GNN does **not** replace physics; it *approximates* the expensive force evaluation while respecting necessary symmetries (permutation equivariance) [22].

---

## 7.  Putting It Together – Mini-Pipeline Example  

1.  Simulation time-step $t$
2.  Build neighbour list → edge index list
3.  Assemble node & edge feature tensors
4.  GNN forward pass → predict accelerations $ \hat{\mathbf{a}}_i $
5.  Symplectic Euler update:
    *   $ \mathbf{v}_{i}^{t+1} = \mathbf{v}_{i}^{t} + \Delta t \cdot \hat{\mathbf{a}}_i $
    *   $ \mathbf{x}_{i}^{t+1} = \mathbf{x}_{i}^{t} + \Delta t \cdot \mathbf{v}_{i}^{t+1} $
6.  Repeat

Because step 2 dominates for large particle counts, spatial data-structures are critical research territory [24].

---

## 8.  Common Pitfalls & Practical Tips  

* **Graph size explosion** – keep the connectivity radius as small as physical fidelity allows.  
* **Edge feature scale** – normalise distances by $h$ so the network does not have to learn absolute units.  
* **Boundary handling** – treat walls as static nodes with special type‐embedding; include normal vectors in edge features.  
* **Validation** – monitor conserved quantities (mass, momentum) during roll-outs to detect drift early [30].

---

## 9.  Summary & Outlook  

The Lagrangian viewpoint converts a fluid into a *living, breathing graph*.  
GNNs then operate as physics-respecting function approximators that:

1. Obey permutation symmetry by construction,  
2. Exploit local interactions through message passing,  
3. Scale to millions of particles with the right data-structures.

In the next lessons we will dissect the **message-passing machinery** itself and see how different aggregation strategies (sum, mean, attention) influence the learned dynamics.

---

[5]: https://community.arm.com/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/physics-simulation-graph-neural-networks-targeting-mobile
[22]: https://karthick.ai/blog/2024/Graph-Neural-Network/
[23]: https://arxiv.org/html/2402.06275v1
[24]: https://www.epcc.ed.ac.uk/whats-happening/articles/accelerating-smoothed-particle-hydrodynamics-graph-neural-networks
[25]: https://www.researchgate.net/publication/358604218_Graph_neural_network-accelerated_Lagrangian_fluid_simulation
[30]: https://arxiv.org/html/2504.13768v1