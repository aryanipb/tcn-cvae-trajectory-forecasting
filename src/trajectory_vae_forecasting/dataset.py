from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class GraphTrajectoryDataset(Dataset):
    def __init__(self, pt_path: str | Path, max_samples: Optional[int] = None):
        path = Path(pt_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        graphs = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(graphs, list):
            raise TypeError(f"Expected a list of Data objects in {path}, got {type(graphs)}")
        if not graphs:
            raise ValueError(f"Dataset is empty: {path}")

        for idx, graph in enumerate(graphs[:5]):
            if not isinstance(graph, Data):
                raise TypeError(f"Item {idx} is not torch_geometric.data.Data")

        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be > 0")
            graphs = graphs[: max_samples]

        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> Data:
        return self.graphs[index]
