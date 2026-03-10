import os
import pandas as pd
import multiprocessing as mp
import shutil

# Constants
PARENT_DIR = "/home/aryan/work/startup/datasetpredmodel/av2parquets"
PROGRESS_FILE = "/home/aryan/work/startup/datasetpredmodel/progress1.txt"
DEST_DIR = "/home/aryan/work/startup/datasetpredmodel/k9av2/"

def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())

def worker_task(dir_path):
    """
    The actual work happens here. Note: We don't update the
    progress file inside the worker to avoid race conditions.
    """
    try:
        df = pd.read_parquet(dir_path)
        filename = os.path.basename(dir_path)
        a = df.groupby('timestep')
        results = []
        for _,group in a:
            result = group['track_id'].nunique()
            results.append(result)
        file_min = min(results)
        if file_min >= 9:
            dest_path = os.path.join(DEST_DIR, filename)
            shutil.copy2(dir_path, dest_path)
            return dir_path, True # Signal success
        return dir_path, False # Signal skipped but done
    except Exception as e:
        print(f"Error processing {dir_path}: {e}")
        return dir_path, None # Signal failure

def main():
    # 1. Setup
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    processed_set = load_progress()

    # 2. Filter tasks
    all_files = [os.path.join(PARENT_DIR, f) for f in os.listdir(PARENT_DIR) if f.endswith(".parquet")]
    to_process = [d for d in all_files if d not in processed_set]

    if not to_process:
        print("Everything is already processed.")
        return

    # 3. Process with Pool
    # Using imap_unordered is often more memory efficient for large lists
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # We open the file once and use it to log as tasks finish
        with open(PROGRESS_FILE, "a", buffering=1) as log_file:
            for result_path, success in pool.imap_unordered(worker_task, to_process):
                if success is not None:
                    log_file.write(result_path + "\n")
                    print(f"Done: {os.path.basename(result_path)}")

if __name__ == "__main__":
    main()
