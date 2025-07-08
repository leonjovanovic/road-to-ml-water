import random
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch import Tensor
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GINConv, GATConv, GATv2Conv, MessagePassing
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from tqdm import tqdm


torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


class GNN(nn.Module):
    """
    A Graph Neural Network (GNN) model that consists of multiple layers of message passing.
    IMPORTANT: Use it only for GCN or GAT layers, not for GINConv.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        output_channels: int,
        layer_class: MessagePassing,
        dropout_rate: float = 0.5,
    ):
        super(GNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.layers = nn.ModuleList()

        layer_first = layer_class(in_channels, hidden_channels[0])
        self.layers.append(layer_first)

        for i in range(len(hidden_channels) - 1):
            self.layers.append(layer_class(hidden_channels[i], hidden_channels[i + 1]))

        layer_last = layer_class(hidden_channels[-1], output_channels)
        self.layers.append(layer_last)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x


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

        layer_last = GINConv(
            self.make_mlp(hidden_channels[-1], output_channels), train_eps=True
        )
        self.layers.append(layer_last)

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

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x


def train(
    model: GNN | GIN, data: Data, hyperparameters: dict
) -> tuple[list[float], list[float]]:

    eps_params = [p for n, p in model.named_parameters() if n.endswith("eps")]
    other_params = [p for n, p in model.named_parameters() if not n.endswith("eps")]

    wd = 0 if model is GIN else 5e-4
    wd_esp = 5e-4
    optimizer = Adam(
        [
            {
                "params": other_params,
                "lr": hyperparameters["learning_rate"],
                "weight_decay": wd,
            },
            {
                "params": eps_params,
                "lr": hyperparameters["learning_rate"] * 0.1,
                "weight_decay": wd_esp,
            },
        ]
    )

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience = hyperparameters.get("patience", 10)
    patience_counter = 0

    for epoch in tqdm(range(hyperparameters["epochs"])):
        model.train()
        optimizer.zero_grad()
        out = model(data)

        train_loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        train_losses.append(train_loss.item())
        train_loss.backward()
        optimizer.step()

        model.eval()
        out = model(data)
        val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        val_losses.append(val_loss.item())
        if epoch % hyperparameters["log_interval"] == 0:
            print(
                f"Epoch: {epoch}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}"
            )

    model.load_state_dict(best_model_state)
    print("Training complete. Best validation loss:", best_val_loss)
    return train_losses, val_losses


def plot_losses(train_losses: list[float], val_losses: list[float]):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Losses")
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate(model: GNN, data: Data):
    model.eval()
    with torch.no_grad():
        out: Tensor = model(data)

    predicted_classes = out.argmax(dim=1)[data.test_mask].cpu().numpy()
    labels = data.y[data.test_mask].cpu().numpy()

    report = classification_report(labels, predicted_classes, digits=3)
    print("Classification Report:\n", report)

    matrix = confusion_matrix(labels, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot(cmap="Blues")
    plt.show()


def main(layer_class: MessagePassing, hyperparameters: dict):

    cora_dataset = Planetoid(root="data", name="Cora")
    data = cora_dataset[0]
    data = data.to(hyperparameters["device"])

    if layer_class in [GCNConv, GATConv, GATv2Conv]:
        model = GNN(
            in_channels=data.num_features,
            hidden_channels=hyperparameters["layers"],
            output_channels=cora_dataset.num_classes,
            layer_class=layer_class,
            dropout_rate=hyperparameters.get("dropout", 0.5),
        )
    else:
        model = GIN(
            in_channels=data.num_features,
            hidden_channels=hyperparameters["layers"],
            output_channels=cora_dataset.num_classes,
            dropout_rate=hyperparameters.get("dropout", 0.5),
        )
    model.to(hyperparameters["device"])

    train_losses, val_losses = train(model, data, hyperparameters)

    plot_losses(train_losses, val_losses)

    evaluate(model, data)
