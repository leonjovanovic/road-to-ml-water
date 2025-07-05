<!-- GNN Roadmap -->

# 3-Phase GNN Roadmap


## PHASE 0 Bootstrap (½ day)

**Purpose:** Get a stable playground and learn what a “graph object” really looks like.

### Tasks
- **1.** Create a Conda environment or `requirements.txt`; install PyTorch + PyTorch Geometric (PyG) (or DGL).  
- **2.** Notebook that:
  - Loads the Cora dataset.
  - Prints tensor shapes.
  - Draws the graph with NetworkX.
- **3.** Write a two‑sentence takeaway.  
  _Example:_  
  > “I didn’t know `edge_index` was just a `2×E` tensor!”

### Key Sources
- **PyTorch Geometric Installation**  
  https://pytorch-geometric.readthedocs.io/en/stable/notes/installation.html :contentReference[oaicite:0]{index=0}  
- **PyG Data API (`Data` object)**  
  https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.data.Data.html :contentReference[oaicite:1]{index=1}  
- **NetworkX Drawing Tutorial**  
  https://networkx.org/documentation/stable/tutorial.html#drawing-graphs :contentReference[oaicite:2]{index=2}  

---

## PHASE 1 Core Supervised GNNs (≈3 days)

**Big idea:** Train out‑of‑the‑box layers on three supervision levels: node, graph, edge.

### 1. Node‑level
- Models: GCN, GIN, GAT on Cora.  
- Output: CSV of accuracies & runtimes.

### 2. Graph‑level
- Model: GIN + global pooling on MUTAG.  
- Task: Regression on QM9 (choose one property).

### 3. Edge‑level
- Task: Link prediction on PPI with positive & negative sampling.

### Deliverables
- `train.py` script with `--task {node,graph,edge}` flag  
- `results.md` table (accuracy, ROC‑AUC, comments)  
- Optional: W&B project link if you log runs  

### Key Sources
- **PyG Official Examples**  
  https://github.com/pyg-team/pytorch_geometric/tree/master/examples :contentReference[oaicite:3]{index=3}  
- **GCN Paper (Kipf & Welling 2017)**  
  https://arxiv.org/abs/1609.02907 :contentReference[oaicite:4]{index=4}  
- **GIN Paper (Xu et al. 2019)**  
  https://arxiv.org/abs/1810.00826 :contentReference[oaicite:5]{index=5}  
- **DGL MUTAG Tutorial**  
  https://docs.dgl.ai/en/latest/tutorials/models/1_gnn/4_mutag.html :contentReference[oaicite:6]{index=6}  

---

## PHASE 2 Beyond the Black Box (≈4–5 days)

**Big idea:** Open the hood, understand representations, run bigger sweeps, finish with a mini‑project.

### 1. Custom GCN Layer
- Implement your own GCN layer (dense first, then sparse).  
- Benchmark against `GCNConv`.

### 2. Embedding Visualization
- Visualize learned embeddings with t‑SNE / UMAP.  
- Inspect GAT attention coefficients `αᵢⱼ`.

### 3. Hyperparameter Sweeps
- Create a `sweep.yaml` for W&B (lr, hidden_dim, layers).  
- Run a sweep to find optimal settings.

### 4. Capstone (choose ONE)
1. **MovieLens 100K Recommendation**  
   Starter: [`examples/gcn_recommender.py`](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn_recommender.py)  
2. **METR‑LA Traffic Prediction**  
   Starter: [`examples/stgcn_metrla.py`](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/stgcn_metrla.py)  
3. **ModelNet10 Point‑Cloud Classification**  
   Starter: [`examples/dynamic_edge_conv_modelnet.py`](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/dynamic_edge_conv_modelnet.py)  

> **Deliverable:** 2‑page report (Problem → Model → Results → Error analysis).

### Key Sources
- **PyTorch Sparse Documentation**  
  https://pytorch.org/docs/stable/sparse.html :contentReference[oaicite:7]{index=7}  
- **scikit‑learn t‑SNE API**  
  https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html :contentReference[oaicite:8]{index=8}  
- **UMAP Package**  
  https://umap-learn.readthedocs.io :contentReference[oaicite:9]{index=9}  
- **W&B Sweeps Quickstart**  
  https://docs.wandb.ai/guides/sweeps/walkthrough :contentReference[oaicite:10]{index=10}  
- **PyG Capstone Starters (GitHub)**  
  https://github.com/pyg-team/pytorch_geometric/tree/master/examples :contentReference[oaicite:11]{index=11}  

---

## Suggested Timeline

| Day      | Phase                                    |
|:--------:|:----------------------------------------:|
| **1**    | Phase 0 – Bootstrap                      |
| **2–4**  | Phase 1 – Core Supervised GNNs           |
| **5–9**  | Phase 2 – Custom Layers & Visualization  |
| **10–14**| Phase 2 Capstone Project                 |

---

## General Tips

- **Reproducibility:** `torch.manual_seed(42)` (graph splits are stochastic).  
- **Debugging:** Test on CPU with a 50‑node subgraph (`k_hop_subgraph`) before GPU.  
- **Persistence:** Save raw logs; Colab/Replit VMs may time out.  
- **Scratchpad:** Maintain `scratch.md` with useful code snippets (e.g., t‑SNE plotting).

---

Happy graphing!
