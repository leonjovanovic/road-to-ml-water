# Lesson 18 – End-to-End Workflow: GNN-Driven Physics Simulation in Practice  

In the previous lessons we dissected every building block of a physics-aware Graph Neural Network (GNN).  
Now we put the pieces together and walk through a **complete simulation loop** – from raw particle data to physically-consistent long-rollout predictions.

> You already know how to train CNNs end-to-end for images.  
> A GNN-based simulator follows the same *encode → process → decode* logic, but with extra steps to (1) build a dynamic graph and (2) marry ML with classical numerical integration.

---

## 1. Big-Picture Pipeline

```text
┌────────────────────┐
│ Raw particle state │  (positions, velocities, etc.)
└─────────┬──────────┘
          ▼
  (1) Graph Construction ──►  (2) Encoder  ──►  (3) Processor (GNN L layers)
          ▲                                      │
          │                                      ▼
  Update particle positions◄──Integrator◄──(5) Decoder◄──(4) Acceleration prediction
```

At every time-step *t* we  

1. build a *fresh graph* from the current particle configuration,  
2. run the network to predict accelerations,  
3. integrate those accelerations to get the next state,  
4. repeat until we reach the desired simulation horizon.

---

## 2. From Particles to Graphs  

### 2.1 Node Features  

| Symbol | Meaning | Typical range |
|--------|---------|---------------|
| $\mathbf{x}_i$ | position $(x,y,z)$ | metres |
| $\mathbf{v}_i$ | velocity $(v_x,v_y,v_z)$ | m s⁻¹ |
| $m_i$ | mass | kg |
| $\mathbf{c}_i$ | one-hot / embedding of **particle class** (fluid, boundary, rigid) | – |
| Optional | density, pressure, temperature | domain-specific |

### 2.2 Edge Construction  

Two particles *i* and *j* are connected iff  

$$
\|\mathbf{x}_i-\mathbf{x}_j\|_2 < r_\text{kernel}
$$

where $r_\text{kernel}$ is the SPH smoothing length [23].  
Efficient neighbour search (spatial hashing, k-d tree) keeps the cost near **O(N)** rather than **O(N²)** [25].

**Edge features**

* Relative displacement $\Delta\mathbf{x}_{ij}=\mathbf{x}_j-\mathbf{x}_i$  
* Distance $d_{ij}=\lVert\Delta\mathbf{x}_{ij}\rVert_2$  
* (For boundaries) normal vector $\mathbf{n}_{ij}$  

---

## 3. Encoding – Lifting Raw Features into Latent Space  

A shallow MLP shared across nodes converts the raw feature vector  
$\mathbf{f}_i = \big[\mathbf{x}_i,\mathbf{v}_i,m_i,\mathbf{c}_i\ldots\big]$  
into a **D-dimensional embedding** $\mathbf{h}^{(0)}_i$.

> Analogy  
> Think of the encoder as assigning every particle a “business card” summarising who it is before the meeting begins.

---

## 4. Processing – Message Passing Core  

Any GCN, GAT or bespoke equivariant layer can be stacked **L** times.  
At layer $l$ the generic update is  

$$
\begin{aligned}
\mathbf{m}^{(l)}_i &= \text{AGG}\_{\;j\in\mathcal{N}(i)}\;
      \phi_\text{msg}\!\big(\mathbf{h}^{(l-1)}_i,\mathbf{h}^{(l-1)}_j,\mathbf{e}_{ij}\big)  
      \\[2pt]
\mathbf{h}^{(l)}_i &= \phi_\text{upd}\!\big(\mathbf{h}^{(l-1)}_i,\mathbf{m}^{(l)}_i\big)
\end{aligned}
$$

* $\phi_\text{msg}$ – an MLP that builds the message  
* **AGG** – *sum / mean / attention* (Lesson 10)  
* $\phi_\text{upd}$ – an MLP or GRU (Lesson 11)

After **L** hops each particle “knows” about its L-neighbourhood (pressure waves, long-range forces, …) [14].

---

## 5. Decoding – Predicting Accelerations  

A final shared MLP maps $\mathbf{h}^{(L)}_i$ to the acceleration  

$$
\hat{\mathbf{a}}_i = g_\theta\!\big(\mathbf{h}^{(L)}_i\big) \in \mathbb{R}^3
$$

Why accelerations?  
* They are the quantity directly linked to forces via **Newton’s 2ⁿᵈ law**  
  $$
  \mathbf{F}_i = m_i\,\mathbf{a}_i
  $$  
  – exactly what message passing is approximating [26].  
* Empirically yields *stable long rollouts* compared with predicting positions outright [30].

---

## 6. Numerical Integration  

We now convert $\hat{\mathbf{a}}_i$ into future positions and velocities.

### 6.1 Forward Euler (simple but fragile) [59]

$$
\begin{aligned}
\mathbf{v}^{t+1}_i &= \mathbf{v}^{t}_i + \Delta t\,\hat{\mathbf{a}}_i \\[4pt]
\mathbf{x}^{t+1}_i &= \mathbf{x}^{t}_i + \Delta t\,\mathbf{v}^{t}_i
\end{aligned}
$$

Unconditionally unstable for many stiff problems.

### 6.2 Symplectic Euler (recommended) [59]

$$
\begin{aligned}
\mathbf{v}^{t+1}_i &= \mathbf{v}^{t}_i + \Delta t\,\hat{\mathbf{a}}_i \\[4pt]
\mathbf{x}^{t+1}_i &= \mathbf{x}^{t}_i + \Delta t\,\mathbf{v}^{t+1}_i
\end{aligned}
$$

• **Conditionally stable** if $\Delta t$ is smaller than a CFL-like limit.  
• Better energy behaviour for Hamiltonian systems.

> Real-life analogy  
> Forward Euler is like driving by always steering based on where you *were* looking a moment ago – you over-correct late.  
> Symplectic Euler first updates your direction, then immediately moves you that way.

---

## 7. Rolling Out Over Time  

```python
state = init_simulation()
for t in range(T):
    graph = build_graph(state)          # O(N) with spatial hashing
    acc   = gnn(graph)                  # forward pass
    state = integrator(state, acc)      # Euler / Symplectic
```

Key details  

* **Dynamic graphs** – edges rebuilt each step (Lesson 15).  
* **Back-prop through time** – for training multi-step losses you usually unroll **K** steps and compute  
  $$
  \mathcal L = \frac1{K}\sum_{k=1}^{K}\big\|\hat{\mathbf{x}}^{t+k} - \mathbf{x}^{t+k}\big\|^2
  $$  
  plus optional energy / momentum regularisers [30].

---

## 8. Training Considerations  

| Topic | Practical tip |
|-------|---------------|
| **Loss** | Mix one-step acceleration MSE with roll-out position error to balance short-term accuracy and long-term stability. |
| **Curriculum** | Start with 1-step prediction; gradually increase rollout horizon as the model improves. |
| **Batching** | PyG / DGL batch disconnected graphs into one big block-diagonal sparse tensor → GPU friendly [5]. |
| **Regularisation** | L2 weight decay + gradient clipping; physics-informed terms (e.g., divergence penalty for incompressibility). |

---

## 9. Evaluation & Diagnostics  

1. **Numeric metrics** – L2 error on positions / velocities, energy drift, divergence.  
2. **Visual inspection** – splash realism, vortex shedding, etc.  
3. **Generalisation tests** – new particle counts, novel boundary shapes, different Δt values [25].

---

## 10. Computational Bottlenecks & Tricks  

* **Neighbour search** can dominate runtime when $N>10^5$.  
  *Use GPU radix hashing or sampled edges to stay real-time* [24].  
* **Memory** – message tensors scale with E (edges). Use radius cut-off & sparse ops.  
* **Mixed precision** – FP16 often acceptable; watch for instabilities in integration.

---

## 11. Worked Example – Real-Time Splash on a Mobile GPU  

Arm’s demo [5] fits a 6-layer GNN (~1 M parameters) that predicts accelerations for <2 k fluid particles at 60 Hz on a smartphone.  
* **Edge radius**: 3 × particle diameter  
* **Integrator**: Symplectic Euler, Δt ≈ 1.6 ms  
* Achieves **20×** speed-up over CPU SPH while preserving splash shape and jet breakup details.

---

## 12. Key Takeaways

* A GNN simulator alternates **learning** (message passing) with **physics-aware integration**.  
* Predicting **accelerations** keeps ML and classical mechanics in their natural roles.  
* **Dynamic graph construction** is as important to optimise as the neural net itself.  
* Stable integrators (symplectic, Verlet, RK4) are essential for long, credible rollouts.  
* When designed carefully, GNN simulators can leap from HPC clusters to *interactive* applications on commodity hardware.

---

[5]: https://community.arm.com/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/physics-simulation-graph-neural-networks-targeting-mobile
[14]: https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book-Chapter_5-GNNs.pdf
[23]: https://arxiv.org/html/2402.06275v1
[24]: https://www.epcc.ed.ac.uk/whats-happening/articles/accelerating-smoothed-particle-hydrodynamics-graph-neural-networks
[25]: https://www.researchgate.net/publication/358604218_Graph_neural_network-accelerated_Lagrangian_fluid_simulation
[26]: https://www.themoonlight.io/en/review/neural-sph-improved-neural-modeling-of-lagrangian-fluid-dynamics
[30]: https://arxiv.org/html/2504.13768v1
[59]: https://phys-sim-book.github.io/lec1.4-explicit_time_integration.html
[60]: https://x-engineer.org/euler-integration/