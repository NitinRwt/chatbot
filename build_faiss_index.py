import json
import numpy as np
import faiss
import os

DATA_DIR = "data"
FAISS_DIR = "faiss_index"

# Input embedding files
EMB_PROFILE_PATH = os.path.join(DATA_DIR, "embeddings_profiles.jsonl")
EMB_JOB_PATH = os.path.join(DATA_DIR, "embeddings_jobs.jsonl")

# Output index files & id maps
if not os.path.isdir(FAISS_DIR):
    os.makedirs(FAISS_DIR)

PROFILE_INDEX_FILE = os.path.join(FAISS_DIR, "profiles_index.bin")
JOB_INDEX_FILE = os.path.join(FAISS_DIR, "jobs_index.bin")
PROFILE_ID_MAP = os.path.join(FAISS_DIR, "profile_id_map.json")
JOB_ID_MAP = os.path.join(FAISS_DIR, "job_id_map.json")

def build_index(emb_path: str, index_file: str, id_map_file: str):
    embeddings = []
    id_list = []
    # 1. Read each embedding
    with open(emb_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            id_list.append(obj["id"])
            embeddings.append(obj["embedding"])
    # 2. Convert to numpy matrix
    vectors = np.array(embeddings, dtype="float32")
    dim = vectors.shape[1]  # 1536

    # 3. Choose a FAISS index type. For small data, we can use IndexFlatL2 (exact search).
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)  # add all vectors

    # 4. Save index and id_map
    faiss.write_index(index, index_file)
    with open(id_map_file, "w", encoding="utf-8") as f_map:
        json.dump(id_list, f_map)

    print(f"Built index: {index_file} (n_items={len(id_list)})")

if __name__ == "__main__":
    build_index(EMB_PROFILE_PATH, PROFILE_INDEX_FILE, PROFILE_ID_MAP)
    build_index(EMB_JOB_PATH, JOB_INDEX_FILE, JOB_ID_MAP)
    print("FAISS index building complete.")
