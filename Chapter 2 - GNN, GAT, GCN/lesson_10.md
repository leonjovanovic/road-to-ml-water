
# Lesson 10 &nbsp;–&nbsp; Aggregation Choices in Graph Neural Networks  
*Sum, Mean, Max & Attention: how GNNs collect information from their neighbours*

---

## 1 Why do we need an “aggregation” step?
During message passing every node *v* receives a (variable-length) multiset of messages  
$\{m_{u\!\to\!v}\;|\;u\in\mathcal N(v)\}$.  
Because a neural network layer expects a **fixed-length** vector, we must *compress* that multiset into a single representation $\mathbf m_v$.

Three hard constraints guide the design:

1. **Permutation invariance** – re-ordering neighbours must not change the result.  
2. **Variable degree** – nodes can have 1 or 10 000 neighbours; the operator must scale.  
3. **Differentiability** – the function must be usable inside back-propagation.

An *aggregation* operator $\text{AGG}(\,\cdot\,)$ that satisfies 1-3 is therefore the linchpin of every GNN layer [14].

---

## 2 Classical invariant operators

### 2.1 Sum aggregation  

$$
\boxed{\;\mathbf m_v = \sum_{u\in\mathcal N(v)} \mathbf h_u\;}
$$  

*Pros*  
* • exact “addition” of neighbour evidence – useful when forces, charges, fluxes **add up** (e.g. total incoming momentum).  
* • implementation is a single sparse-matrix multiplication; fast and hardware friendly.  

*Cons*  
* • high-degree nodes may dominate (values can explode) unless later normalised [22].

---

### 2.2 Mean aggregation  

$$
\boxed{\;\mathbf m_v = \tfrac1{|\mathcal N(v)|}\sum_{u\in\mathcal N(v)} \mathbf h_u\;}
$$

Effectively a **normalised sum** (used in vanilla GCN).  
Keeps magnitudes stable irrespective of degree; well-behaved on social or citation graphs where information should be “shared” rather than accumulated [22].

---

### 2.3 Max pooling  

$$
\boxed{\;[\mathbf m_v]_k = \max_{u\in\mathcal N(v)} [\mathbf h_u]_k\;}
$$

Selects the element-wise strongest signal.  
Analogy: instead of *asking friends for an average opinion*, you listen to the **loudest** voice in each topic.  
Great for picking up sharp, localised cues (e.g. the most highly stressed spring in a mechanical lattice).

---

## 3 Learnable, **attention-based** aggregation  

Classical operators treat all neighbours equally; physical interactions are rarely that democratic.  
Graph Attention Networks (GAT) introduce *data-driven weights* [21]:

1. **Linear projection**

   $$
   \mathbf g_i = \mathbf W\,\mathbf h_i
   $$

2. **Edge score**

   $$
   e_{ij} = \text{LeakyReLU}\big(\mathbf a^\top[\mathbf g_i\;\|\;\mathbf g_j]\big)
   $$

3. **Normalise**

   $$
   \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k\in\mathcal N(i)}\exp(e_{ik})}
   $$

4. **Weighted sum**

   $$
   \boxed{\;\mathbf m_i = \sum_{j\in\mathcal N(i)} \alpha_{ij}\,\mathbf g_j\;}
   $$

Several *heads* can be run in parallel; their outputs are concatenated (or averaged) prior to the update.

*Physical interpretation* – the network *learns* which neighbouring particle exerts more influence; comparable to an adaptive kernel width in SPH.  
Empirically improves splash or collision prediction where a few close-by particles dominate the dynamics [55].

---

## 4 Choosing an operator for physical simulation

| Phenomenon | Typical operator | Rationale |
|------------|-----------------|-----------|
| Mass / charge conservation | **Sum** | globally additive quantity |
| Diffusive processes (heat) | **Mean** | information spreads evenly |
| Contact or fracture detection | **Max** | look for extreme stress |
| Highly anisotropic flows (jet, splash) | **Attention** | focus on direction-dependent forces |

> Analogy – Think of aggregation like *budget reports* in a company:  
> • **Sum**: total sales across all branches.  
> • **Mean**: average salary per employee (branch size shouldn’t skew).  
> • **Max**: highest single expense flagging possible fraud.  
> • **Attention**: the CFO weighs certain branch reports more because of strategic importance.

---

## 5 Implementation tips (PyTorch Geometric)

```python
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

# sum / degree-normalised mean
conv1 = GCNConv(in_channels, hidden)

# mean / max / concatenated aggregators
conv2 = SAGEConv(hidden, hidden, aggr='mean')   # 'sum' or 'max' also possible

# multi-head attention (8 heads, outputs concatenated)
conv3 = GATConv(hidden, hidden // 8, heads=8)
```

All operators return the same tensor shape, so they can be swapped during ablation studies.

*Efficiency caveat* – attention requires edge-wise soft-max ⇒ memory grows with *#edges*.  
For very dense particle neighbourhoods (e.g. granular matter) you might combine *sum* for long-range forces and *attention* inside a restricted radius.

---

## 6 Key take-aways

* Aggregation turns a *set* of neighbour messages into a fixed vector while preserving permutation invariance.  
* **Sum, mean, max** are parameter-free, cheap, and often sufficient.  
* **Attention** makes the weighting learnable and excels when influence is heterogeneous or directional.  
* The operator you pick implicitly sets an *inductive bias* – align it with the physics you want to capture.  
* Always monitor gradient magnitudes; high-degree graphs may need normalisation or residual tricks.

---

## Sources  

1. The Graph Neural Network Model [14]  
2. Graph Neural Network – In a Nutshell [22]  
3. Graph Attention Networks (GAT) tutorial [21]  
4. Physics Simulation with Graph Neural Networks (mobile) [5]  
5. Graph Attention vs GCN comparison [55]

[14]: https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book-Chapter_5-GNNs.pdf  
[22]: https://karthick.ai/blog/2024/Graph-Neural-Network/  
[21]: https://www.baeldung.com/cs/graph-attention-networks  
[5]:  https://community.arm.com/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/physics-simulation-graph-neural-networks-targeting-mobile  
[55]: https://towardsdatascience.com/graph-neural-networks-part-2-graph-attention-networks-vs-gcns-029efd7a1d92
