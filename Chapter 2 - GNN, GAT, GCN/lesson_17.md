# Lesson 17 — From Prediction to Motion: Euler & Symplectic Integration Schemes  

**Prerequisite:** You understand how a GNN can predict per–particle accelerations (Lesson 16).  
**Goal of this lesson:** Translate those accelerations into updated positions and velocities while keeping the simulation numerically stable and physically plausible.

---

## 1 | Why do we need a numerical integrator?

The GNN gives us instantaneous accelerations  
$$
\mathbf{a}^{(n)} = \bigl[a_x^{(n)}, a_y^{(n)}, a_z^{(n)}\bigr]
$$
for every particle at timestep $n$.  
Physics, however, is expressed in continuous time:

$$
\frac{d\mathbf{v}}{dt}= \mathbf{a}(t)\,,\qquad
\frac{d\mathbf{x}}{dt}= \mathbf{v}(t)
$$

To move forward from the discrete prediction $\mathbf{a}^{(n)}$ to the next state $(\mathbf{x}^{(n+1)},\mathbf{v}^{(n+1)})$, we must **integrate** these differential equations over the interval $\Delta t$.  

> **Big-picture analogy:**  
> Think of your smartphone’s map app. GPS gives you *instantaneous* velocity. To know where you’ll be 5 s from now, it numerically integrates that velocity over time, repeatedly correcting the path as new GPS data arrive. Our integrator plays the same role for particles.

---

## 2 | The Forward (Explicit) Euler Method

The most straightforward explicit integrator:

$$
\begin{aligned}
\mathbf{x}^{(n+1)} &= \mathbf{x}^{(n)} + \Delta t \,\mathbf{v}^{(n)} \\
\mathbf{v}^{(n+1)} &= \mathbf{v}^{(n)} + \Delta t \,\mathbf{a}^{(n)}
\end{aligned} \tag{1}
$$

* **Pros**  
  * Two trivial vector additions — *fast*.  
* **Cons**  
  * **Unconditionally unstable** for many stiff or highly non-linear systems [59].  
  * Energy can **blow up** even with small $\Delta t$.

If you try Forward Euler on a frictionless pendulum, its amplitude spirals outward. In fluid simulation, splashes may gain unphysical energy after a few hundred steps.

---

## 3 | The Symplectic (Semi-Implicit) Euler Method

A minimal tweak yields a **symplectic** integrator that preserves Hamiltonian structure better [59]:

$$
\begin{aligned}
\mathbf{v}^{(n+1)} &= \mathbf{v}^{(n)} + \Delta t \,\mathbf{a}^{(n)} \\
\mathbf{x}^{(n+1)} &= \mathbf{x}^{(n)} + \Delta t \,\mathbf{v}^{(n+1)}
\end{aligned} \tag{2}
$$

* **Key difference:** The new position uses the **updated** velocity.  
* **Properties**  
  * **Conditionally stable** — stable if $\Delta t$ is below a problem-specific limit.  
  * **Energy behaviour:** Does **not** drift systematically; it oscillates around the true value, a desirable trait for long rollouts.  
  * Still $O(1)$ per particle.

### Quick stability check  

For a 1-D harmonic oscillator $x'' + \omega^2 x = 0$, Symplectic Euler is stable if  

$$
\Delta t < \frac{2}{\omega}.
$$

That bound offers a concrete criterion when choosing $\Delta t$ for oscillatory regimes in fluids (e.g., surface tension waves).

---

## 4 | Choosing the Time Step $\Delta t$

| Trade-off | Small $\Delta t$ | Large $\Delta t$ |
|-----------|-----------------|-----------------|
| Accuracy  | High            | Low             |
| Wall-clock time | Slow (more steps) | Fast            |
| Stability | Safe            | Risky / diverges |

*Rule of thumb:* Start with the $\Delta t$ that your original SPH or CFD solver used, then experiment upward while watching kinetic energy curves.  

*GNN interaction:* Because the network was trained at some $\Delta t_{\text{train}}$, very large deviations can create a distribution shift. Fortunately, recent work shows moderate generalisation to unseen $\Delta t$ values if the training set contained varied dynamics [30].

---

## 5 | Where to place the integrator in code?

```python
# Pseudo-Python (PyTorch)
acc = gnn(graph)                    # predict accelerations  (m/s²)
vel = vel + dt * acc                # symplectic: update v first
pos = pos + dt * vel                # then update x
graph = rebuild_graph(pos)          # dynamic connectivity (Lesson 15)
```

Keeping the integrator **outside** the GNN keeps the network size small and lets you swap better integrators later (e.g., Runge–Kutta 4). Some researchers fold the update into the decoder to co-optimise parameters [30]; both designs work, choose for engineering convenience.

---

## 6 | Error Propagation & How the Integrator Interacts with GNN Noise

Even a well-trained GNN has residual error $\varepsilon^{(n)}$ in its acceleration prediction. Forward Euler *adds* that error to velocity and then position. Over **T** steps the positional error can grow $\mathcal{O}(T\,\varepsilon)$.

Symplectic Euler behaves better:

1. It damps pure positional drift because updated velocities quickly feed back into the dynamics.  
2. Energy oscillations limit runaway growth, making long rollouts more robust in practice [26].

---

## 7 | Beyond Euler: When should I upgrade?

| Scheme | Order | Extra Cost | When to consider |
|--------|-------|-----------|------------------|
| Mid-point / Heun | 2 | ×2 force evals | You need smoother trajectories for visual FX |
| RK4 | 4 | ×4 evals | High-precision offline simulations |
| Learned integrators | ? | Mixed | Combine integrator & GNN errors end-to-end |

For mobile or real-time engines, Symplectic Euler is often the sweet spot [5]. For research prototypes aimed at precise quantitative metrics, higher-order schemes can pay off.

---

## 8 | Real-Life Example: Car Suspension  

A car wheel hits a bump; the suspension acts like a damped spring.  

1. **GNN** predicts acceleration of wheel hub based on spring compression, damper, and tyre-road contact.  
2. **Integrator** updates the hub’s velocity and position at 1 kHz.  
3. **Symplectic Euler** keeps the total mechanical energy (potential + kinetic) bounded, so the simulated wheel doesn’t oscillate forever or explode upward.  

Without a stable integrator, the car in your racing game would jitter or fly into the sky after a few bumps.

---

## 9 | Summary

1. **Accelerations are not enough** — integrate them!  
2. **Forward Euler** is simple but can blow up.  
3. **Symplectic Euler** is nearly as cheap and **much** more stable.  
4. The integrator sits *after* the GNN; its timestep $\Delta t$ is a critical hyper-parameter.  
5. Stable integration + accurate GNN = long, believable rollouts.

---

## 10 | Further Reading

1. **Explicit Time Integration – Physics-Based Simulation** [59]  
2. **Euler Integration Method for Solving Differential Equations** [60]  
3. **Neural SPH: Improved Neural Modeling of Lagrangian Fluid Dynamics** [26]  
4. **Equi-Euler GraphNet** (integration folded into GNN) [30]  
5. **Physics Simulation with Graph Neural Networks Targeting Mobile** [5]

---

## References

[5]: <https://community.arm.com/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/physics-simulation-graph-neural-networks-targeting-mobile>
[26]: <https://www.themoonlight.io/en/review/neural-sph-improved-neural-modeling-of-lagrangian-fluid-dynamics>
[30]: <https://arxiv.org/html/2504.13768v1>
[59]: <https://phys-sim-book.github.io/lec1.4-explicit_time_integration.html>
[60]: <https://x-engineer.org/euler-integration/>
