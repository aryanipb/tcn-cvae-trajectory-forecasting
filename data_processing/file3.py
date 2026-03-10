import torch
from pathlib import Path
import os

INPUT_ROOT = "/home/aryan/work/startup/datasetpredmodel/av2_graphs"
OUTPUT_ROOT = "/home/aryan/work/startup/datasetpredmodel"


files = [f for f in os.listdir(INPUT_ROOT) if f.endswith(".pt")]
graphs = []
for file in files:
    graph = torch.load(Path(INPUT_ROOT) / file, map_location="cpu", weights_only=False)
    graphs.append(graph)

torch.save(graphs, Path(OUTPUT_ROOT) / "ak9val.pt")
