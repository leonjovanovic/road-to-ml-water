from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch import Tensor
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from tqdm import tqdm


class GCNModel(nn.Module):
    def __init__(
        self, in_channels: int, hidden_channels: list[int], output_channels: int
    ):
        super(GCNModel, self).__init__()
        self.layers = nn.ModuleList()

        layer_first = GCNConv(in_channels, hidden_channels[0])
        self.layers.append(layer_first)

        for i in range(len(hidden_channels) - 1):
            self.layers.append(GCNConv(hidden_channels[i], hidden_channels[i + 1]))

        layer_last = GCNConv(hidden_channels[-1], output_channels)
        self.layers.append(layer_last)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x)
        x = self.layers[-1](x, edge_index)
        return x


def train(
    model: GCNModel, data: Data, hyperparameters: dict
) -> tuple[list[float], list[float]]:
    optimizer = Adam(
        model.parameters(),
        lr=hyperparameters["learning_rate"],
        weight_decay=hyperparameters["weight_decay"],
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


def evaluate(model: GCNModel, data: Data):
    model.eval()
    out: Tensor = model(data)

    predicted_classes = out.argmax(dim=1)[data.test_mask].cpu().numpy()
    labels = data.y[data.test_mask].cpu().numpy()

    report = classification_report(labels, predicted_classes, digits=3)
    print("Classification Report:\n", report)

    matrix = confusion_matrix(labels, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot(cmap="Blues")
    plt.show()


def main(hyperparameters: dict):
    cora_dataset = Planetoid(root="data", name="Cora")
    data = cora_dataset[0]
    data.to(hyperparameters["device"])

    model = GCNModel(
        in_channels=data.num_features,
        hidden_channels=hyperparameters["layers"],
        output_channels=cora_dataset.num_classes,
    )
    model.to(hyperparameters["device"])

    train_losses, val_losses = train(model, data, hyperparameters)

    plot_losses(train_losses, val_losses)

    evaluate(model, data)


if __name__ == "__main__":
    hyperparameters = {
        "learning_rate": 0.01,
        "weight_decay": 5e-4,
        "epochs": 200,
        "log_interval": 10,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    main(hyperparameters)
