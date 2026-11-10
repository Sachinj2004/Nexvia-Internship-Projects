"""
Real-Time Financial Fraud Detection Platform (AIML-01)
PyTorch Geometric Graph Neural Network Engine: src/gnn.py

Implements a high-performance GraphSAGE (Graph Sample and Aggregate) model
for credit card transaction fraud classification using sparse similarity graphs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GNN_MODEL_PATH = PROJECT_ROOT / "models" / "gnn_model.pt"
DEFAULT_GNN_SCALER_PATH = PROJECT_ROOT / "models" / "gnn_scaler.joblib"
DEFAULT_ANCHORS_PATH = PROJECT_ROOT / "models" / "gnn_anchors.parquet"

EXPECTED_FEATURES: List[str] = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]


# -----------------------------------------------------------------------------
# Graph Construction Utilities
# -----------------------------------------------------------------------------
def build_knn_graph(
    features: np.ndarray,
    k: int = 5,
    metric: str = "euclidean"
) -> torch.Tensor:
    """
    Construct a sparse k-Nearest Neighbors transaction similarity graph.
    Prevents self-loops and produces bidirectional undirected edges.

    Args:
        features: Scaled feature array of shape (N, 30).
        k: Number of nearest neighbors to connect per node.
        metric: Distance metric for nearest neighbors calculation.

    Returns:
        torch.Tensor: Edge index tensor of shape (2, E).
    """
    n_samples = len(features)
    if n_samples <= 1:
        # Edge case: single node has no neighbors
        return torch.empty((2, 0), dtype=torch.long)

    effective_k = min(k + 1, n_samples)
    nn_search = NearestNeighbors(n_neighbors=effective_k, metric=metric, n_jobs=-1)
    nn_search.fit(features)
    indices = nn_search.kneighbors(features, return_distance=False)

    source_nodes = []
    target_nodes = []

    for src_idx in range(n_samples):
        for neighbor_idx in indices[src_idx]:
            if src_idx != neighbor_idx:  # exclude self-loops
                source_nodes.append(src_idx)
                target_nodes.append(neighbor_idx)
                # Ensure bidirectional symmetry
                source_nodes.append(neighbor_idx)
                target_nodes.append(src_idx)

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    # Remove duplicate edges
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


def create_pyg_data(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    k: int = 5
) -> Data:
    """
    Pack numerical features and sparse k-NN edges into a PyTorch Geometric Data object.
    """
    edge_index = build_knn_graph(features, k=k)
    x_tensor = torch.tensor(features, dtype=torch.float32)

    if labels is not None:
        y_tensor = torch.tensor(labels, dtype=torch.float32)
        return Data(x=x_tensor, edge_index=edge_index, y=y_tensor)

    return Data(x=x_tensor, edge_index=edge_index)


# -----------------------------------------------------------------------------
# Model Architecture: Two-Layer GraphSAGE
# -----------------------------------------------------------------------------
class FraudGraphSAGE(nn.Module):
    """
    Two-layer inductive GraphSAGE network for transaction fraud detection.
    Aggregates neighborhood feature context to identify coordinated fraudulent patterns.
    """

    def __init__(
        self,
        in_channels: int = 30,
        hidden_channels: int = 64,
        out_channels: int = 1,
        dropout: float = 0.3
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr="mean")
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing raw logit fraud scores.
        """
        # If no edges exist (e.g. isolated single node), SAGEConv gracefully uses self-node representation
        h = self.conv1(x, edge_index)
        # Apply batch norm if batch has more than 1 sample
        if h.size(0) > 1:
            h = self.bn1(h)
        h = F.leaky_relu(h, negative_slope=0.1)
        h = self.dropout(h)
        logits = self.conv2(h, edge_index)
        return logits.squeeze(-1)

    def predict_probabilities(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute calibrated probabilities via sigmoid activation."""
        logits = self.forward(x, edge_index)
        return torch.sigmoid(logits)


# -----------------------------------------------------------------------------
# OOP Predictor Interface
# -----------------------------------------------------------------------------
class GNNFraudPredictor:
    """
    High-level scoring interface utilizing the trained PyTorch Geometric model.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        anchors_path: Optional[Path] = None,
        device: Optional[str] = None,
        default_threshold: float = 0.5
    ):
        self.model_path = model_path or DEFAULT_GNN_MODEL_PATH
        self.scaler_path = scaler_path or DEFAULT_GNN_SCALER_PATH
        self.anchors_path = anchors_path or DEFAULT_ANCHORS_PATH
        self.default_threshold = default_threshold

        # Device selection: Apple Silicon MPS if available, otherwise CPU
        if device:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model: Optional[FraudGraphSAGE] = None
        self.scaler = None
        self.anchor_features: Optional[np.ndarray] = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load fitted StandardScaler and serialized PyTorch model weights."""
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"GNN scaler not found at: {self.scaler_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"GNN model checkpoint not found at: {self.model_path}")

        self.scaler = joblib.load(self.scaler_path)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model = FraudGraphSAGE(
            in_channels=checkpoint.get("in_channels", 30),
            hidden_channels=checkpoint.get("hidden_channels", 64),
            out_channels=checkpoint.get("out_channels", 1),
            dropout=0.0  # Zero dropout during inference
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Load reference anchors if available for single-transaction neighborhood augmentation
        if self.anchors_path.exists():
            try:
                anchors_df = pd.read_parquet(self.anchors_path)
                clean_anchors = anchors_df[EXPECTED_FEATURES].values
                self.anchor_features = self.scaler.transform(clean_anchors)
            except Exception:
                self.anchor_features = None

    def predict_batch(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        threshold: Optional[float] = None,
        k: int = 5
    ) -> pd.DataFrame:
        """
        Score a batch DataFrame or list of transaction payloads using GraphSAGE.
        """
        t = threshold if threshold is not None else self.default_threshold

        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")

        missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Batch data missing required features: {missing}")

        features_raw = df[EXPECTED_FEATURES].astype(float).values
        features_scaled = self.scaler.transform(features_raw)

        pyg_data = create_pyg_data(features_scaled, k=k)
        x = pyg_data.x.to(self.device)
        edge_index = pyg_data.edge_index.to(self.device)

        with torch.no_grad():
            probs_tensor = self.model.predict_probabilities(x, edge_index)
            # Handle 0-d scalar output if single row
            if probs_tensor.dim() == 0:
                probs = np.array([probs_tensor.cpu().item()])
            else:
                probs = probs_tensor.cpu().numpy()

        result_df = df.copy()
        result_df["fraud_probability"] = np.round(probs, 6)
        result_df["predicted_class"] = (probs >= t).astype(int)
        result_df["risk_label"] = np.where(result_df["predicted_class"] == 1, "Fraud", "Legitimate")
        result_df["threshold_used"] = float(t)
        result_df["status"] = np.where(result_df["predicted_class"] == 1, "FLAGGED_FOR_REVIEW", "APPROVED")

        return result_df

    def predict_single(
        self,
        transaction: Dict[str, Any],
        threshold: Optional[float] = None,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Score an individual transaction.
        Augments graph neighborhood using cached reference anchor nodes.
        """
        t = threshold if threshold is not None else self.default_threshold

        missing = [c for c in EXPECTED_FEATURES if c not in transaction]
        if missing:
            raise ValueError(f"Transaction missing required features: {missing}")

        single_raw = np.array([[float(transaction[c]) for c in EXPECTED_FEATURES]])
        single_scaled = self.scaler.transform(single_raw)

        # If reference anchors exist, augment neighborhood to allow message passing
        if self.anchor_features is not None and len(self.anchor_features) > 0:
            combined_features = np.vstack([single_scaled, self.anchor_features])
            pyg_data = create_pyg_data(combined_features, k=k)
            x = pyg_data.x.to(self.device)
            edge_index = pyg_data.edge_index.to(self.device)

            with torch.no_grad():
                probs = self.model.predict_probabilities(x, edge_index).cpu().numpy()
            prob = float(probs[0])
        else:
            # Standalone inference
            pyg_data = create_pyg_data(single_scaled, k=1)
            x = pyg_data.x.to(self.device)
            edge_index = pyg_data.edge_index.to(self.device)

            with torch.no_grad():
                prob_tensor = self.model.predict_probabilities(x, edge_index)
                prob = float(prob_tensor.cpu().item() if prob_tensor.dim() == 0 else prob_tensor.cpu().numpy()[0])

        is_fraud = int(prob >= t)
        return {
            "fraud_probability": round(prob, 6),
            "predicted_class": is_fraud,
            "risk_label": "Fraud" if is_fraud == 1 else "Legitimate",
            "threshold_used": float(t),
            "status": "FLAGGED_FOR_REVIEW" if is_fraud == 1 else "APPROVED"
        }
