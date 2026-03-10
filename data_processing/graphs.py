import os
import pandas as pd
import multiprocessing as mp
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import cKDTree

parent_directory = "/home/aryan/work/startup/datasetpredmodel/k9av2"
progress_file = "/home/aryan/work/startup/datasetpredmodel/progressgraphs.txt"
torch.set_printoptions(profile="full")
k = 9

def load_progress():
    if not os.path.exists(progress_file):
        return set()
    with open(progress_file, "r") as f:
        return set(line.strip() for line in f if line.strip())

processed_set = load_progress()

def mark_done(dir_path):
    if dir_path in processed_set:
        return
    with open(progress_file, "a") as f:
        f.write(dir_path + "\n")
    processed_set.add(dir_path)

def f(dir_path):
    parquet_file = dir_path
    filename = os.path.splitext(os.path.basename(parquet_file))[0]
    df = pd.read_parquet(parquet_file)
    focal = str(df['focal_track_id'].iloc[0])
    ego_features = df[df['track_id'] == focal].sort_values(by=['timestep'])

    b = df.groupby('timestep')
    events = [g for _, g in b]
    if (len(events)!=110):
        print(f"Expected 110 timesteps in file {filename}, but got {len(events)}. Skipping.")
    input_events = events[:50]

    temporal_features = np.zeros((50, k, 5), dtype=np.float32)
    spatial_features = np.zeros((50, k, 4), dtype=np.float32)
    edge_index = np.zeros((50, k), dtype=np.int64)

    new_ego_features = ego_features[['position_x', 'position_y', 'velocity', 'acceleration', 'heading']]
    t0_ego = new_ego_features.iloc[0]
    for timestep, input_event in enumerate(input_events):
        ego_row = ego_features.loc[ego_features['timestep'] == input_event['timestep'].iloc[0]]
        if ego_row.empty:
            print(f"No ego data for timestep {input_event['timestep'].iloc[0]} in file {filename}")
            return None

        ego_feature = ego_row.iloc[0]

        others = input_event[input_event['track_id'] != focal]
        coords = others[['position_x', 'position_y']].to_numpy()

        tree = cKDTree(coords)
        dists, idxs = tree.query([ego_feature['position_x'], ego_feature['position_y']], k=k)

        selected = others.iloc[idxs].reset_index(drop=True)

        edge_index[timestep] = idxs

        temporal_features[timestep, :, 0] = selected['position_x'].to_numpy() - t0_ego['position_x']
        temporal_features[timestep, :, 1] = selected['position_y'].to_numpy() - t0_ego['position_y']
        temporal_features[timestep, :, 2] = selected['velocity'].to_numpy() - t0_ego['velocity']
        temporal_features[timestep, :, 3] = selected['acceleration'].to_numpy() - t0_ego['acceleration']
        temporal_features[timestep, :, 4] = selected['heading'].to_numpy() - t0_ego['heading']

        spatial_features[timestep, :, 0] = dists
        spatial_features[timestep, :, 1] = np.sqrt((selected['velocity_x'] - ego_feature['velocity_x'])**2 + (selected['velocity_y'] - ego_feature['velocity_y'])**2)
        spatial_features[timestep, :, 2] = np.sqrt((selected['acceleration_x'] - ego_feature['acceleration_x'])**2 + (selected['acceleration_y'] - ego_feature['acceleration_y'])**2)
        spatial_features[timestep, :, 3] = np.arctan2(np.sin(ego_feature['heading'] - selected['heading']), np.cos(ego_feature['heading'] - selected['heading']))

    temporal_features = torch.tensor(temporal_features, dtype=torch.float32)
    spatial_features = torch.tensor(spatial_features, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    t0_ego = torch.tensor(t0_ego.values, dtype=torch.float32)

    target = ego_features[['position_x', 'position_y', 'velocity', 'acceleration', 'heading']].iloc[50:90]
    target = torch.tensor(target.values, dtype=torch.float32)
    target = target - t0_ego

    ego_features1 = ego_features[['position_x', 'position_y', 'velocity', 'acceleration', 'heading']].iloc[:50]
    ego_features1 = torch.tensor(ego_features1.values, dtype=torch.float32)
    ego_features1 = ego_features1 - t0_ego

    out_file = os.path.join('/home/aryan/work/startup/datasetpredmodel/av2_graphs', filename + ".pt")
    data = Data(x=temporal_features, edge_attr=spatial_features, edge_index=edge_index, y=target, ego_features=ego_features1, t0_ego=t0_ego)
    torch.save(data, out_file)
    return dir_path

if __name__ == "__main__":
    dirs = [
    os.path.join(parent_directory, f)
    for f in os.listdir(parent_directory)
    if f.endswith(".parquet")]


    to_process = [d for d in dirs if d not in processed_set]

    if not to_process:
        print("Nothing to process; all files listed in progress.txt are done.")
    else:
        pool = mp.Pool(processes=6)
        async_results = []
        try:
            for d in to_process:
                r = pool.apply_async(f, args=(d,), callback=lambda res, d=d: mark_done(d))
                async_results.append(r)
            pool.close()
            for r in async_results:
                r.wait()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
    print("Processing complete.")
