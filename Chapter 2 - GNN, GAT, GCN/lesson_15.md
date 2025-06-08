# Lesson 15 – Building **Dynamic Graphs** for Particle-Based Simulations  
*How to translate a moving physical system into a new graph at every timestep*

---

## 1 Why “dynamic” graphs?

In molecular dynamics, SPH fluid solvers or granular media, each particle is a **node** whose neighbours change continuously.  
If particle *A* swims away from particle *B*, the edge (A,B) should disappear; if *C* drifts closer, a new edge (A,C) should be created.  
Unlike a molecule with a fixed bond list, the *connectivity radius* in fluids is purely geometric, so the graph must be **rebuilt** (or at least **updated**) at every simulation step [1][2].

<div align="center">

| property | static molecule | flowing water |
|----------|-----------------|--------------|
| edge semantics | chemical bond | proximity  |
| typical update rate | never | every Δt |
| consequence for GNN | one graph object | a stream of graphs |

</div>

---

## 2 Choosing the connectivity rule

### 2.1 Smoothing length *h*

Most particle solvers already define a **smoothing length** *h* used in kernel evaluations.  
We simply adopt the same rule for our graph:

$$
\text{edge}(i,j)=
\begin{cases}
1 & \text{if } \|\,\mathbf{x}_j-\mathbf{x}_i\|_2 \le h \\
0 & \text{otherwise}
\end{cases}
$$

where  

* $\mathbf{x}_i$ – current position of particle *i*  
* $\|\cdot\|_2$ – Euclidean norm

This guarantees that the GNN “sees” exactly the particles that would exchange forces in a traditional SPH kernel.

### 2.2 Fixed-*k* nearest neighbours (k-NN)

An alternative is to connect each node to its *k* closest neighbours, even in highly non-uniform regions.  
Pros: bounded degree → constant memory.  
Cons: may ignore a physically relevant neighbour if *k* is too small, or create unphysical long-range edges in low-density regions.

---

## 3 Efficient neighbour search  

A naïve all-pairs search is $\mathcal{O}(N^2)$ – prohibitive for 100 k particles.  
Well-known spatial data structures drop the cost to (roughly) $\mathcal{O}(N \log N)$ or even $\mathcal{O}(N)$:

* **Uniform spatial hashing / cell linked lists** – classical SPH trick; constant-time binning if cell width ≥ *h*.  
* **k-d tree** – balanced tree in 3-D, good when particle density varies a lot.  
* **GPU radix sort + prefix sum** – used by recent GNN-accelerated SPH engines on mobiles [3].

> Analogy – finding friends at a concert  
> Instead of shouting everyone’s name (all-pairs), you first split the crowd into blocks by seat number (spatial hashing) and only shout inside *your* block.

---

## 4 Constructing the data tensors

For each timestep *t* we build

1. **Node feature matrix**  
   $$
   \mathbf{X}^{(t)} \in \mathbb{R}^{N \times D_\text{node}}
   $$
   Typical channels  
   – position, velocity, mass, one-hot type, density, pressure …

2. **Edge index list** (sparse)  
   $$
   \mathbf{E}^{(t)} = 
     \begin{bmatrix}
       i_1 & i_2 & \dots \\
       j_1 & j_2 & \dots
     \end{bmatrix}
   $$
   where $(i_\ell ,j_\ell)$ is the ℓ-th directed edge.

3. **Edge feature matrix**  
   $$
   \mathbf{F}^{(t)} \in \mathbb{R}^{|\mathbf{E}^{(t)}| \times D_\text{edge}}
   $$
   Common channels  
   – relative displacement $\Delta\mathbf{x}_{ij}$, distance $d_{ij}$, maybe a boundary normal.

> Real-life mapping  
> • Position → GPS of a car  
> • Velocity → its speedometer  
> • Edge feature → relative bearing and distance to neighbouring cars

Most GNN libraries (PyTorch Geometric, DGL) accept the triple `(X, E, F)` directly.  
In PyG:

```python
data = Data(x=X, edge_index=E, edge_attr=F)
```

---

## 5 Static rebuild vs. incremental update

| strategy | complexity | GPU friendliness | notes |
|----------|------------|------------------|-------|
| **Full rebuild** every Δt | simple; same cost each step | easy | safest; used in many papers [2] |
| **Incremental** update | cheaper if particles move little | tricky – scattered updates | needs book-keeping of cell transfers |

For highly turbulent flows the full rebuild is often chosen because neighbour lists change anyway.

---

## 6 Edge features in practice

Edge attributes are not mandatory, but they boost accuracy because **force laws depend on geometry**.

Example – 2-D SPH kernel weight  
$$
w_{ij} = \left( 1 - \frac{d_{ij}}{h} \right)^3_+
$$

You can let the network *learn* such a weight by feeding $d_{ij}$ as a feature.  
Attention-based GNNs (GAT) sometimes use only $\Delta\mathbf{x}_{ij}$ and let the attention mechanism infer $d_{ij}$ implicitly.

---

## 7 Putting it all together – algorithm per timestep Δt

1. Update positions $\mathbf{x}_i$ and velocities $\mathbf{v}_i$ from previous physics step.  
2. **Neighbour search** → obtain edge list.  
3. Build $(\mathbf{X}^{(t)},\mathbf{E}^{(t)},\mathbf{F}^{(t)})$.  
4. **GNN forward** pass → predict accelerations $\mathbf{a}_i$ (or forces).  
5. Integrate with symplectic Euler (Lesson 17).  
6. Go to next timestep.

Pseudocode:

```python
for t in range(num_steps):
    E = neighbour_search(x, h)           # step 2
    data = make_graph(x, v, type, E)     # steps 3
    a = gnn(data)                        # step 4
    v += dt * a                          # integration (simplified)
    x += dt * v
```

---

## 8 Common pitfalls

* **Edge churn** – a particle near the cutoff radius may flicker in/out of the graph each frame → noisy forces.  
  Mitigation: add *hysteresis* (two radii: connect if ≤ $h_\text{in}$, drop if ≥ $h_\text{out}$).  
* **Memory blow-up** – in 3-D each node may have ~150 neighbours.  
  Use half-edges (store one direction) or limit *k*.  
* **Race conditions on GPU** – when two threads insert edges into the same adjacency list.  
  Rely on library primitives or sort-based batching.  

---

## 9 Big-picture take-away

Dynamic graph construction is the **bridge** between the continuous world of moving particles and the discrete world where GNNs operate.  
If that bridge is **slow** or **inaccurate**, no amount of fancy message passing will fix the simulation.  
Therefore, spend as much engineering effort on neighbour search and data layout as on the neural network itself.

---

## Sources

[1]: https://www.epcc.ed.ac.uk/whats-happening/articles/accelerating-smoothed-particle-hydrodynamics-graph-neural-networks
[2]: https://www.researchgate.net/publication/358604218_Graph_neural_network-accelerated_Lagrangian_fluid_simulation
[3]: https://community.arm.com/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/physics-simulation-graph-neural-networks-targeting-mobile