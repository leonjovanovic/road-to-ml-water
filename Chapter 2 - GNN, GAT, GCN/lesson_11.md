# Lesson 11 – Updating Node States & the **Encode-Process-Decode** Pipeline  
_Graph Neural Networks for Physical Systems_

---

## 1. Where We Are in the Message-Passing Loop  

Recall the generic *Message Passing Neural Network* (MPNN) cycle carried out at every layer *ℓ* of a GNN:

1. **Message** generation  
2. **Aggregation** of incoming messages  
3. **Update** of each node’s own hidden state  

So far we have already discussed steps 1 and 2.  
Today we focus on **step 3** and on the larger **encode-process-decode** architecture in which that step usually lives.

---

## 2. The Update Function – Formal Definition  

For a node *v* let  

*  `h_v^(ℓ-1)` be its hidden feature vector before layer ℓ  
*  `m_v^(ℓ)`   be the aggregated message coming from its neighbours at layer ℓ  

The **update** operation is a learnable map  

\[
h_v^{(\ell)} = \operatorname{UPDATE}^{(\ell)}\!\bigl( h_v^{(\ell-1)},\; m_v^{(\ell)} \bigr).
\]

This single line hides several design choices that strongly influence the expressive power, training stability and physical interpretability of the network.

---

## 3. Popular Update Mechanisms  

| Mechanism | Equation | When to Use | Quick Analogy |
|-----------|-----------|------------|---------------|
| **Linear combination** | \(h_v^{(\ell)} = W m_v^{(\ell)} + b\) | Very shallow models or when you want a simple linear transform. | Receiving a summary email and forwarding it unchanged except for a template. |
| **MLP / non-linear** | \(h_v^{(\ell)} = \sigma\!\bigl(W_1 m_v^{(\ell)} + W_2 h_v^{(\ell-1)} + b\bigr)\) | Default choice; adds non-linearity to capture complex interactions. | Reading the summary, mixing it with your past experience and writing a new plan. |
| **Gated Recurrent Unit (GRU)** | Uses reset and update gates, e.g.  \(\tilde h = \tanh\bigl(W_m m_v^{(\ell)} + r \odot U h_v^{(\ell-1)}\bigr)\) with  \(h_v^{(\ell)} = (1-z)\odot h_v^{(\ell-1)} + z\odot \tilde h\). | Needed when information must be remembered or forgotten selectively, e.g. long multi-step simulations. | A project manager who decides what to keep from last week and what new info to incorporate. |

### Why gating helps in physics  
Particle interactions can have both *slow* (e.g. pressure) and *fast* (e.g. collisions) components.  
GRU/LSTM-style updates give each node an internal memory that decides whether to preserve a long-term trend or react instantly to a sudden force spike.

---

## 4. Putting the Update in Context – Encode-Process-Decode

Most successful GNN-based fluid or rigid-body simulators adopt the following macro-architecture:

```
Raw Particle Data ─▶ Encoder ─▶ Processor (GNN core) ─▶ Decoder ─▶ Physical Output
                    (h0, e0)        (K × message-passing)       (e.g. acceleration)
```

### 4.1 Encoder  
* Maps raw positions, velocities, types, … into **initial node/edge embeddings** \(h_v^{(0)}, e_{uv}^{(0)}\).  
* Often a small MLP per node/edge.  
* Think of it as translating different measurement units into a common internal language.

### 4.2 Processor = Repeated Update Cycle  
* Consists of *K* identical (or stacked) GNN layers.  
* Each layer executes _message → aggregation → update_.  
* Weight sharing across the *K* iterations makes the processor **recurrent in space**: exactly the same interaction rule is applied multiple times, letting information propagate K-hops.

> **Real-life analogy**  
> Imagine a group of engineers (particles) exchanging design notes.  
> After each exchange round they revise their own blueprint (update) and send out an updated note.  
> Repeating the same protocol K times spreads the effect of a single comment throughout the entire team.

### 4.3 Decoder  
* Reads the final node embeddings \(h_v^{(K)}\) and produces physically meaningful quantities  
  – typically **accelerations** \(\mathbf{a}_v\) or **forces**.  
* The decoder is *task specific*: a regression head for accelerations differs from a classifier head used in fracture prediction.

### 4.4 Benefits of the Modular Split  

| Benefit | Reason |
|---------|--------|
| **Interpretability** | You can inspect encoder outputs, latent dynamics and final predictions separately. |
| **Re-usability / Transfer learning** | A processor trained on one fluid can be paired with a new encoder to handle different input formats. |
| **Stability tricks in one place** | Residual connections or normalization layers can be added inside the processor without touching the rest. |

---

## 5. Practical Tips & Gotchas  

1. **Residual (skip) connections**  
   Add \(h_v^{(\ell)} \;{+}= h_v^{(\ell-1)}\) to mitigate *over-smoothing* and preserve local detail.

2. **Shared vs unshared weights**  
   *Shared* (a single set reused K times) enforces the same physics rule repeatedly – closer to classical time-integration schemes.  
   *Unshared* increases capacity but risks overfitting.

3. **Normalisation inside UPDATE**  
   LayerNorm or BatchNorm on \(m_v^{(\ell)}\) often stabilises training, especially when node degrees vary wildly.

4. **Physical constraints**  
   Add loss terms or projection layers (e.g. to conserve total momentum) after the decoder if required by the application.

---

## 6. Worked-Out Mini-Example  

Imagine 2500 SPH particles representing a water drop impacting a surface.

1. **Encoder**  
   * Position \((x,y,z)\), velocity \((v_x,v_y,v_z)\) and mass are fed through a 2-layer MLP, producing \(h_v^{(0)}\in\mathbb{R}^{64}\).

2. **Processor**  
   * K = 10 layers, each  
     * message: edge-MLP using relative displacement,  
     * aggregation: **sum**,  
     * update: GRU with hidden size = 64.  
   * Weight matrix **shared** across the 10 iterations.

3. **Decoder**  
   * Single linear layer to \(\mathbf{a}_v\in\mathbb{R}^3\).

4. **Integration** (external)  
   * Symplectic Euler with Δt = 2 ms updates positions & velocities.

Despite its simplicity, this setup achieves ≈40× speed-up over a classical solver on the same hardware while preserving splash dynamics within 5 % error over 1000 steps [5].

---

## 7. Key Take-Aways  

* The **update function** is where each node fuses _its past_ with _newly aggregated evidence_.  
* Choosing between linear, non-linear or gated updates trades simplicity for long-range temporal reasoning.  
* The **encode-process-decode** layout isolates concerns and mirrors established simulation pipelines (input parsing → timestep loop → output).  
* Proper design of the update mechanism and processor depth is critical for capturing both **local fast interactions** and **global slow trends** in physical systems.

---

## References  

[5] <https://community.arm.com/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/physics-simulation-graph-neural-networks-targeting-mobile>  
[14] <https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book-Chapter_5-GNNs.pdf>  
[22] <https://karthick.ai/blog/2024/Graph-Neural-Network/>  
[30] <https://arxiv.org/abs/2504.13768>
