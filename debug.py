import torch
from tests.make_golden import load_model, generate_greedy_golden   # use the real name of your golden script

dev = "cuda"
golden = torch.load("tests/golden.pt")
model, rt = load_model("checkpoints/ckpt_step20000.pt", dev, backend="pytorch")
print("dtype:", next(model.parameters()).dtype)

for i, e in enumerate(golden):
    ref = generate_greedy_golden(model, rt, dev, e["prompt_ids"].tolist(), 50).tolist()
    g = e["generated_ids"].tolist()
    bad = next((j for j, (a, b) in enumerate(zip(ref, g)) if a != b), None)
    print(i, "prompt_len", len(e["prompt_ids"]), "OK" if bad is None else f"golden differs at index {bad}")