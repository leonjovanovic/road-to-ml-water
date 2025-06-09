# Lesson 2 — Shortcomings of MLPs & CNNs on Irregular, Interacting Systems
*(course: “Graph Neural Networks for Physical Systems”)*

---

## 1 Why Talk About the “Old” Models First?
You already wield MLPs and CNNs with ease, so understanding **their precise failure modes** on particle‐based physics will clarify **why Graph Neural Networks (GNNs) are needed at all**. It is not about bashing classics; it is about matching inductive bias to problem structure.

---

## 2 MLPs: Fixed-Size Vectors in a Variable-Size World
### 2.1 Architectural Assumption
An MLP consumes an input vector
$$
\mathbf{x}\in\mathbb{R}^{n}
$$
whose dimensionality *n* is fixed at design time. Every hidden layer is another affine map followed by a non-linearity.

### 2.2 What Breaks for Particles?
| Challenge | Consequence with MLPs |
|-----------|-----------------------|
| Number of particles changes frame-to-frame | Need to **pad, truncate, or rasterize** → wasted memory or lost data |
| Physics lives in **relations** (pairwise forces, constraints) | Flat vector provides **no native notion of neighbourhood** |
| Particle ordering is arbitrary | MLP output **changes when you shuffle inputs** |

### 2.3 “But MLPs Are Universal Approximators!”
True—given enough width they approximate any continuous map [1]. Yet approximation theorems ignore *data encoding*. Converting a set of 20 000 unordered particles into a fixed vector destroys relational structure, so the network wastes parameters simply reinventing symmetry.

Theoretical error decreases like
$$
\varepsilon(N)=\mathcal{O}\!\bigl(N^{-1}\bigr)
$$
with *N* hidden units [2], but only **after** the input has been embedded sensibly—something the vanilla MLP cannot enforce.

---

## 3 CNNs: Grids, Kernels and the Tyranny of Regularity
### 3.1 Translation Equivariance Recap
For a convolution layer
$$
f\!\bigl(T_{\delta}\mathbf{x}\bigr)=T_{\delta}\,f(\mathbf{x})
$$
where *T*δ shifts the image by δ. Perfect for pixels and voxels [3].

### 3.2 Irregular Domains Break the Assumption
Particles do **not** live on grids. Mapping them onto one introduces:

* **Aliasing** – two particles in the same voxel merge.
* **Scale explosion** – halving voxel size multiplies memory by 2³ in 3-D.
* **Orientation mismatch** – rotate the cloud and the grid‐based representation changes even though physics did not.

### 3.3 Pooling Kills Precision
Pooling is ideal for “cat vs. dog” but catastrophic for **pressure waves**, **shock fronts** and **free surfaces** that require sub-voxel fidelity [4].

### 3.4 Real-World Analogy – “Birds on a Chessboard”
Imagine forcing a flock of birds to perch on a chessboard:

* Two birds on one square collapse into one “super-bird”.
* Empty squares waste space.
* A 45° rotation alters which birds occupy which squares, although the flock itself is unchanged.

That is exactly what rasterisation does to particles.

---

## 4 Physical Consistency & Symmetry
Physics respects permutation of particle labels. MLPs/CNNs do not:

* Training must expose **every possible ordering** → data blow-up.
* Extrapolation to *more* or *fewer* particles is brittle.
* Conservation laws require **manual loss terms or post-processing** [5], [6].

Patching these gaps (heavy augmentation, custom layers, hand-crafted features) adds complexity without fully solving the root problem [7].

---

## 5 Case Study – CNN Forecasts a Splash … Poorly
1. **Input**: 20 k SPH particles (positions, velocities, densities).
2. **Hack**: Rasterise densities onto a 128 × 128 grid; feed to a U-Net.
3. **Output**: Next‐step density image → de-rasterise back to particles.

Observed issues:

* **Satellite droplets vanish** (sub-voxel).
* **Momentum leaks** through zero padding.
* Doubling resolution multiplies memory by 4× in 2-D, 8× in 3-D.

The culprit is the mismatch between continuous, relational data and grid-centric inductive bias.

---

## 6 When Are MLPs or CNNs Still Appropriate?
* **Eulerian CFD** (flow variables already on grids).
* **Rendering / visual analytics** of particle data.
* **Hybrid models** where an image branch handles fixed boundaries while a GNN tracks particles.

But *for unstructured, dynamic state itself*, we need an architecture whose native language is “entities + relations”.

---

## 7 Key Take-Aways
1. **MLPs require fixed-size, ordered inputs** → unsuitable for variable particle counts.
2. **CNNs assume regular grids and locality** → break on irregular geometries & rotations.
3. Neither architecture encodes **permutation symmetry** or **relational reasoning**.
4. Work-arounds exist but are messy, data-hungry, and often physically inconsistent.

The natural successor is the **Graph Neural Network**, built from the ground up around sets and their interactions. In the next lesson we will see how GNNs innately respect permutation symmetry and relational structure.

---

[1]: https://arxiv.org/html/2504.11397v1
[2]: https://ar5iv.labs.arxiv.org/html/1805.00915
[3]: https://blog.paperspace.com/pooling-and-translation-invariance-in-convolutional-neural-networks/
[4]: https://fabianfuchsml.github.io/equivariance1of2/
[5]: https://proceedings.neurips.cc/paper_files/paper/2023/file/6cde6435e111671b04f4574006cf3c47-Paper-Conference.pdf
[6]: https://www.mdpi.com/2673-2688/5/3/74
[7]: https://arxiv.org/html/2504.08766v1