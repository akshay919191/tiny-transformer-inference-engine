# prepare_data.py
import numpy as np
import tiktoken
import os

tok = tiktoken.get_encoding("gpt2")

def tokenize_file(input_path, output_path, chunk_size=5_000_000):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    total_tokens = 0
    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "wb") as f_out:
        while True:
            chunk = f_in.read(chunk_size)
            if not chunk:
                break
            ids = tok.encode_ordinary(chunk)
            arr = np.array(ids, dtype=np.uint16)
            arr.tofile(f_out)
            total_tokens += len(ids)
            print(f"[{output_path}] Processed {total_tokens:,} tokens...")
    print(f"[{output_path}] Done. Total: {total_tokens:,}")
    return total_tokens

if __name__ == "__main__":
    tokenize_file("data/TinyStories-train.txt", "data/train.bin")
    tokenize_file("data/TinyStories-valid.txt", "data/val.bin")
