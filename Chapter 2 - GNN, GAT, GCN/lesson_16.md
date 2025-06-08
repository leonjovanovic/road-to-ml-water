# Lesson 16 – Choosing Prediction Targets: **Why GNNs Predict Accelerations (or Forces)**


## 1. Where We Are in the Pipeline  

Encoder → **Processor (GNN)** → **Decoder** → **Numerical Integrator**  
In every roll-out step the GNN answers one simple (but crucial) question:

> “Given the current state of all particles, **what quantity should I predict so that classical mechanics can take it from there?**”

The three obvious choices are  
1. next positions  
2. next velocities  
3. **accelerations / forces**  

Today we justify why option 3 has become the de-facto standard in modern GNN-based simulators.

---

## 2. Why the Output Quantity Matters  

Choosing the wrong target is like buying the wrong lens for your camera: the picture can still be taken, but noise, distortion and lost details creep in.

* Numerical **stability** – small per-step errors explode over thousands of steps.  
* **Physical consistency** – predictions must obey conservation laws.  
* **Trainability** – the target must be smooth enough for gradient-based learning, yet expressive enough to reconstruct the future state.

---

## 3. Option A: Predicting Next **Positions**  

### Pro  
* Intuitive – directly get the future coordinates.

### Con  
* The network must implicitly learn **two integrations** (acceleration → velocity, velocity → position) every step.  
* Large extrapolation gap → tiny error early becomes meters of drift later (think of throwing darts while standing on a moving treadmill).  

Result: even with teacher forcing during training, long roll-outs diverge quickly [26][].

---

## 4. Option B: Predicting Next **Velocities**  

### Pro  
* Removes one layer of integration.  
* Lower variance than positions.

### Con  
* Still requires the network to accumulate forces over **unknown duration Δt**.  
* Energy can slowly leak because force symmetry is not enforced.  

Good for some rigid-body tasks, yet often unstable for fluids and granular media [30][].

---

## 5. Option C: Predicting **Accelerations / Forces** – The Physics-Aligned Choice  

### How Nature Works  
Newton’s second law  

$$
\mathbf{F} = m\mathbf{a}
$$

states that **forces cause accelerations**, then classical integrators recover velocities and positions. Matching this causal chain pays off:

1. **Locality** – accelerations depend only on *current* neighbours, a perfect fit for message passing.  
2. **Lower dynamic range** – no accumulation over time, gradients stay well-behaved.  
3. **Plug-and-play with legacy solvers** – any off-the-shelf integrator (Euler, Verlet, Runge-Kutta, …) can consume the prediction.  
4. **Conservation laws** – easier to add symmetric or energy-conserving loss terms because force pairs act along edges [23][], [25][].

### Real-Life Analogy  
Think of GPS navigation. The map algorithm computes *turn-by-turn acceleration* (“brake here, steer left”), not “where you’ll be in 30 minutes”. Your car’s dynamics (wheels + engine) integrate those tiny commands into the full trajectory. The two-component system – planner + vehicle – mirrors **GNN predictor + integrator**.

---

## 6. Mathematical Formulation  

Let  

* $ \mathbf{x}_i^n , \mathbf{v}_i^n $  – position/velocity of particle *i* at step *n*  
* $ \mathbf{a}_i^n = \text{GNN}_\theta(\mathcal{G}^n)_i $ – acceleration predicted by the GNN  

### Symplectic (Semi-Implicit) Euler – Popular with GNNs [59][]

$$
\begin{aligned}
\mathbf{v}_i^{\,n+1} &= \mathbf{v}_i^{\,n} + \Delta t \, \mathbf{a}_i^{\,n} \\
\mathbf{x}_i^{\,n+1} &= \mathbf{x}_i^{\,n} + \Delta t \, \mathbf{v}_i^{\,n+1}
\end{aligned}
$$

Why not **Forward Euler**?  
Forward Euler updates $ \mathbf{x} $ with **old** velocity $ \mathbf{v}_i^n $. Energy errors accumulate and the simulation can “blow up” unless Δt is extremely small [59][].

---

## 7. Practical Tips  

| Pitfall | Mitigation |
|---------|-----------|
| **Unit mismatch** (e.g. cm s⁻² vs m s⁻²) | Normalise inputs and outputs; store scale factors. |
| **Very stiff contacts** (rigid collisions) | Predict *impulse* or use learned contact–specific heads. |
| **Diverse particle types** | One-hot encode type, or use learned embeddings concatenated to node features. |
| **Long-horizon drift** | Add loss on multi-step roll-outs; optionally regularise total momentum / energy. |
| **Variable Δt at inference** | Train with Δt sampled from a range to improve robustness [25][]. |

---

## 8. Big-Picture View  

Predicting accelerations turns the GNN into a **learned force field** surrounding every particle. Traditional integrators then replay centuries-old physics to turn that field into motion.  
This **division of labour** combines the expressive power of deep learning with the reliability of numerical analysis – a recurring theme in modern scientific ML.

---

## 9. Summary  

* Output choice is *not* cosmetic – it drives stability, accuracy and interpretability.  
* **Accelerations/forces** respect the causal hierarchy of mechanics and mesh perfectly with message passing.  
* Pairing a GNN force predictor with a stable integrator such as Symplectic Euler enables roll-outs thousands of steps long without catastrophic drift.  

In short, “predict forces, let physics do the rest.”

---

## 10. Sources  

1. Neural SPH: Improved Neural Modeling of Lagrangian Fluid Dynamics [23][]  
2. Graph Neural Network-Accelerated Lagrangian Fluid Simulation [25][]  
3. Literature Review – Neural SPH [26][]  
4. Equi-Euler GraphNet: Dual Force and Trajectory Prediction [30][]  
5. Physics Simulation with Graph Neural Networks Targeting Mobile [5][]  
6. Explicit Time Integration – Physics-Based Simulation (Euler methods) [59][]  
7. Euler Integration Method for ODEs [60][]  

[5]: https://community.arm.com/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/physics-simulation-graph-neural-networks-targeting-mobile
[23]: https://arxiv.org/html/2402.06275v1
[25]: https://www.researchgate.net/publication/358604218_Graph_neural_network-accelerated_Lagrangian_fluid_simulation
[26]: https://www.themoonlight.io/en/review/neural-sph-improved-neural-modeling-of-lagrangian-fluid-dynamics
[30]: https://arxiv.org/html/2504.13768v1
[59]: https://phys-sim-book.github.io/lec1.4-explicit_time_integration.html
[60]: https://x-engineer.org/euler-integration/