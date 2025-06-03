# Lesson 14 – Going Deeper: Layer Stacking, Receptive Field & Over-Smoothing Challenges  


## 1  Why add more GNN layers?  

In a Graph Neural Network one **message–passing layer** lets every node consult its *immediate* neighbours.  
Stacking layers means repeating this procedure so that information can travel farther:

<div align="center">

$$
\underbrace{\text{1 layer}}_{\text{1-hop}}
\; \Longrightarrow \;
\underbrace{\text{K layers}}_{\text{K-hop receptive field}}
$$
</div>

After **K layers** the embedding of node \(v\) has access to every node that is at most \(K\) edges away:

<div align="center">

$$
\boxed{\;h^{(K)}_v 
=\; \text{AGGREGATE}\Bigl(\!\bigl\{h^{(0)}_u\;|\;\text{dist}(u,v)\le K\bigr\}\Bigr)}
$$
</div>

*Real-life analogy* – Think of gossip in a city. One person tells their friends (1 hop), after two phone calls friends-of-friends know (2 hops) and so on. The deeper the chain the wider the reach.

For physics this is indispensable:  
• A local **pressure wave** propagates through fluid particles several neighbourhoods away.  
• A deformable solid needs distant nodes to “feel” global constraints such as fixed boundary particles.

---

## 2  Formal reminder: the update rule through depth  

A generic message-passing layer can be written [14]:

<div align="center">

\[
m^{(l)}_v \;=\;
\text{AGG}\Bigl(\{\,\phi_m(h^{(l-1)}_v,h^{(l-1)}_u,e_{uv})\;|\;u\in\mathcal N(v)\}\Bigr)
\]

\[
h^{(l)}_v \;=\;
\phi_u\!\bigl(h^{(l-1)}_v , m^{(l)}_v\bigr)
\]
</div>

where  
* \(h^{(l)}_v\) – embedding of node \(v\) after \(l\) layers,  
* \(\phi_m,\phi_u\) – learnable functions (usually small MLPs),  
* \(\text{AGG}\) – permutation-invariant aggregator (sum/mean/max/attention).

Iterating the pair \(\bigl(m^{(l)}_v, h^{(l)}_v\bigr)\) **K times** stacks K layers.

---

## 3  Benefits & first warning signs  

| Depth | What you gain | What can go wrong |
|-------|---------------|-------------------|
| Shallow (1–2) | Only very local effects | Miss long-range physics |
| Medium (3–6) | Captures most interactions in typical particle radius | Small risk of over-smoothing |
| Deep (7 +) | Global context, entire fluid domain | Over-smoothing, exploding cost |

### Computational cost  

Each extra layer adds:
* A new parameter matrix (memory).  
* One more sparse matrix multiplication (runtime).  

> On an SPH fluid with 100 k particles, moving from 3➜8 layers roughly triples GPU memory-time footprint [5].

---

## 4  Over-smoothing – when everything looks the same  

**Definition** – After many layers node embeddings converge to an almost identical vector, losing discrimination power [16].

Mathematically, repeated aggregation is similar to applying a **random-walk Markov operator**. Its powers push all rows towards the dominant eigenvector, hence features “wash out”.

*Coffee analogy* – Keep stirring milk into an espresso: eventually colour differences disappear → the embeddings are identical beige.

### Symptoms
* Validation accuracy plateaus then drops.  
* Cosine similarity between distant node embeddings → 1.  
* Gradients at early layers vanish.

---

## 5  Over-squashing (brief mention)  

While over-smoothing makes features similar, **over-squashing** happens when *too much* information must be compressed into a *fixed-size* vector that travels through a narrow graph bottleneck [44].  
Both phenomena are amplified by depth but stem from different causes.

---

## 6  Mitigation strategies  

| Technique | Intuition | Equation / Description |
|-----------|-----------|------------------------|
| **Residual / skip connections** (“ResGNN”) | Give later layers direct access to earlier, more local features [16]. | $$h^{(l)}_v = \phi_u(h^{(l-1)}_v,m^{(l)}_v) \;+\; h^{(l-1)}_v$$ |
| **Dense or Jumping Knowledge** | Concatenate embeddings from several layers and let model choose best scale. | $$h^{\text{out}}_v = \text{JK}\bigl(h^{(1)}_v,\dots,h^{(K)}_v\bigr)$$ |
| **Normalization (Batch/Layer)** | Prevents numerical explosion, stabilises gradient flow. | apply to \(h^{(l)}\) before non-linearity |
| **Depth-adaptive architectures** | Use fewer layers but enlarge receptive field with *dilated* neighbourhoods or *hierarchical pooling*. ||
| **Graph rewiring / positional encodings** | Add virtual edges or distance encodings so fewer layers are needed to pass distant info [44]. ||

> In practice, most fluid-simulation papers settle on **3–6 GNN layers with residual links** – deep enough to span several smoothing radii yet shallow enough to avoid smoothing [5].

---

## 7  Practical checklist for your own models  

1. **Measure graph diameter**. For many SPH scenes: diameter ≈ 10–15 hops. Aim for K ≈ ⅓–½ of that.  
2. **Plot embedding variance vs. depth** during training; collapsing variance signals over-smoothing soon.  
3. **Add residual connections** by default. They cost nothing but help greatly.  
4. **Benchmark depth**: run 2, 4, 6 layers; plot rollout error vs. wall-clock time. Select a knee point.  
5. **Remember neighbour growth**: A K-hop neighbourhood scales roughly as \(O(\text{degree}^K)\). Use sparse libraries (PyG / DGL) so cost does not explode.

---

## 8  Summary  

Depth lets a GNN *see farther*, exactly what you need for long-range physical effects.  
Yet each layer also blurs information. Going too deep leads to **over-smoothing**; information-rich but undifferentiated embeddings hurt performance.  

Mitigate with residual connections, smart aggregators, and by choosing a depth that matches *your* graph’s physical scope. The goal is a **Goldilocks depth** – *just right* for the scale of interactions you want to capture.

---

## References  

[5] – Physics Simulation With Graph Neural Networks Targeting Mobile. <https://community.arm.com/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/physics-simulation-graph-neural-networks-targeting-mobile>  

[14] – The Graph Neural Network Model. <https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book-Chapter_5-GNNs.pdf>  

[16] – Introduction to Graph Machine Learning. <https://huggingface.co/blog/intro-graphml>  

[44] – Math Behind Graph Neural Networks. <https://rish-16.github.io/posts/gnn-math/>  
