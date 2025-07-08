from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score
)

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_add_pool
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset, QM9

from node_level_train import plot_losses


class GIN(nn.Module):
    """
    A Graph Isomorphism Network (GIN) model that consists of multiple layers of GINConv.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        output_channels: int,
        dropout_rate: float = 0.5,
    ):
        super(GIN, self).__init__()
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList()

        layer_first = GINConv(
            self.make_mlp(in_channels, hidden_channels[0]), train_eps=True
        )
        self.layers.append(layer_first)

        for i in range(len(hidden_channels) - 1):
            self.layers.append(
                GINConv(
                    self.make_mlp(hidden_channels[i], hidden_channels[i + 1]),
                    train_eps=True,
                )
            )
        self.head = nn.Linear(hidden_channels[-1], output_channels)

    def make_mlp(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        use_bn: bool = False,
    ) -> nn.Sequential:
        if hidden_dim is None:
            hidden_dim = out_dim
        layers = [nn.Linear(in_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers += [nn.ReLU(), nn.Linear(hidden_dim, out_dim)]
        return nn.Sequential(*layers)

    def forward(self, data: Data, regression: bool = False) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = global_add_pool(x, batch)

        return self.head(x)


def train(
    model: GIN, train_loader: DataLoader, val_loader: DataLoader, hyperparameters: dict
) -> tuple[list[float], list[float]]:

    eps_params = [p for n, p in model.named_parameters() if n.endswith("eps")]
    other_params = [p for n, p in model.named_parameters() if not n.endswith("eps")]

    optimizer = Adam(
        [
            {
                "params": other_params,
                "lr": hyperparameters["learning_rate"],
                "weight_decay": 0,
            },
            {
                "params": eps_params,
                "lr": hyperparameters["learning_rate"] * 0.1,
                "weight_decay": 5e-4,
            },
        ]
    )

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience = hyperparameters.get("patience", 10)
    patience_counter = 0

    for epoch in tqdm(range(hyperparameters["epochs"])):
        model.train()
        train_loss_sum, n_graphs = 0.0, 0
        for train_batch in train_loader:
            optimizer.zero_grad()
            out = model(train_batch)
            if hyperparameters["head_type"] == "regression":
                prediction = out.squeeze()
                target = train_batch.y[:, hyperparameters["target_property_index"]].float()
                train_loss = F.mse_loss(prediction, target)
            else:
                train_loss = F.cross_entropy(out, train_batch.y)
            train_loss.backward()
            optimizer.step()
            train_loss_sum += train_loss.item() * len(train_batch)  # weight by size
            n_graphs += len(train_batch)

        train_losses.append(train_loss_sum / n_graphs)

        model.eval()
        val_loss_sum, n_graphs = 0.0, 0
        for val_batch in val_loader:
            with torch.no_grad():
                out = model(val_batch)
                if hyperparameters["head_type"] == "regression":
                    prediction = out.squeeze()
                    target = val_batch.y[:, hyperparameters["target_property_index"]].float()
                    val_loss = F.mse_loss(prediction, target)
                else:
                    val_loss = F.cross_entropy(out, val_batch.y)
                
                val_loss_sum += val_loss.item() * len(val_batch)  # weight by size
                n_graphs += len(val_batch)

        val_loss = val_loss_sum / n_graphs
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        val_losses.append(val_loss)
        if epoch % hyperparameters["log_interval"] == 0:
            print(
                f"Epoch: {epoch}, Train Loss: {train_loss.item()}, Val Loss: {val_loss}"
            )

    model.load_state_dict(best_model_state)
    print("Training complete. Best validation loss:", best_val_loss)
    return train_losses, val_losses


def evaluate_classification(model: GIN, test_loader: DataLoader):
    model.eval()
    outs, labels = [], []
    for test_batch in test_loader:
        with torch.no_grad():
            out: torch.Tensor = model(test_batch)
            outs.append(out)
            labels.append(test_batch.y)

    outs = torch.cat(outs, dim=0)
    labels = torch.cat(labels, dim=0).cpu().numpy()
    predicted_classes = outs.argmax(dim=1).cpu().numpy()

    report = classification_report(labels, predicted_classes, digits=3)
    print("Classification Report:\n", report)

    matrix = confusion_matrix(labels, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot(cmap="Blues")
    plt.show()


def evaluate_regression(model: GIN, test_loader: DataLoader, hyperparameters: dict):
    model.eval()
    outs, labels = [], []

    for test_batch in test_loader:
        with torch.no_grad():
            out: torch.Tensor = model(test_batch)  # shape: [batch_size, 1]
            outs.append(out.view(-1))              # flatten to [batch_size]
            labels.append(test_batch.y[:, hyperparameters["target_property_index"]])

    # Concatenate predictions and labels
    preds = torch.cat(outs, dim=0).cpu().numpy()
    targets = torch.cat(labels, dim=0).cpu().numpy()

    # Compute regression metrics
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    print(f"Regression Evaluation:\nMAE: {mae:.4f}\nR² Score: {r2:.4f}")


def load_dataset(dataset_name: str, device: str) -> tuple[QM9 | TUDataset, list[Data]]:
    if dataset_name == "QM9":
        dataset = QM9(root="data/QM9")
        print(f"QM9: {len(dataset)} molecules")

    elif dataset_name == "MUTAG":
        dataset = TUDataset(root="data", name="MUTAG")
        print(f"MUTAG: {len(dataset)} graphs")

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_list = [data.to(device) for data in dataset]
    return dataset, dataset_list


def main(hyperparameters: dict):

    dataset, dataset_list = load_dataset(
        hyperparameters["dataset"], hyperparameters["device"]
    )

    model = GIN(
        in_channels=dataset[0].num_features,
        hidden_channels=hyperparameters["layers"],
        output_channels=(
            1 if hyperparameters["head_type"] == "regression" else dataset.num_classes
        ),
        dropout_rate=hyperparameters["dropout"],
    )
    model = model.to(hyperparameters["device"])

    train_set, val_set, test_set = random_split(dataset_list, hyperparameters["split"])
    train_loader = DataLoader(
        train_set, batch_size=hyperparameters["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    train_losses, val_losses = train(model, train_loader, val_loader, hyperparameters)

    plot_losses(train_losses, val_losses)

    if hyperparameters["head_type"] == "classification":
        evaluate_classification(model, test_loader)
    else:
        evaluate_regression(model, test_loader, hyperparameters)
