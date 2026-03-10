import os
import pandas as pd
import multiprocessing as mp
import numpy as np

parent_directory = "/home/aryan/work/startup/datasetpredmodel/av2"
progress_file = "/home/aryan/work/startup/datasetpredmodel/progresss0.txt"

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
    files = [f for f in os.listdir(dir_path) if f.endswith(".parquet")]
    if not files:
        return None

    parquet_file = os.path.join(dir_path, files[0])
    filename = os.path.splitext(os.path.basename(parquet_file))[0]
    df = pd.read_parquet(parquet_file)
    a  = df.groupby('track_id')
    result = []
    for _, group in a:
        group = group.sort_values(by="timestep")

        dt_step = group["timestep"].diff()
        dt_real = dt_step * 0.1

        valid = (dt_step == 1)

        acc_x = np.zeros(len(group), dtype=np.float32)
        acc_y = np.zeros(len(group), dtype=np.float32)

        dvx = group["velocity_x"].diff()
        dvy = group["velocity_y"].diff()

        acc_x[valid] = dvx[valid] / dt_real[valid]
        acc_y[valid] = dvy[valid] / dt_real[valid]

        group["acceleration_x"] = acc_x
        group["acceleration_y"] = acc_y
        result.append(group)

    df = pd.concat(result, ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    df['acceleration'] = np.sqrt(df['acceleration_x']**2 + df['acceleration_y']**2)
    df['velocity'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
    df = df[(df['object_type'] == 'vehicle') | (df['object_type'] == 'cyclist') | (df['object_type'] == 'bus') | (df['object_type'] == 'motorcyclist')]
    df.drop(columns=['observed', 'object_type', 'object_category', 'num_timestamps', 'city', 'start_timestamp', 'end_timestamp', 'scenario_id'], inplace=True)
    df.to_parquet(f"/home/aryan/work/startup/datasetpredmodel/av2parquets/{filename}.parquet")
    return dir_path

if __name__ == "__main__":
    dirs = [
    os.path.join(parent_directory, f)
    for f in os.listdir(parent_directory)
    if os.path.isdir(os.path.join(parent_directory, f))]


    to_process = [d for d in dirs if d not in processed_set]

    if not to_process:
        print("Nothing to process; all directories listed in progress.txt are done.")
    else:
        pool = mp.Pool(processes=9)
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
