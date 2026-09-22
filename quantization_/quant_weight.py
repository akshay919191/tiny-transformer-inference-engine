"""
in this model we have fp32 , that' why we will use 4 to calculate size
"""

from .quant_int8 import quantize_
from models.transformer_block import Transformer
from models.model_config import ModelConfig
from extracter import unwrapped_state_dict , ckpt
from dataclasses import fields
from train import args
import torch
import torch.nn as nn

run_time = ModelConfig(**{f.name: getattr(args, f.name) for f in fields(ModelConfig)})

model = Transformer(run_time , attn_type = "mqa" , backend = "cuda")
model.load_state_dict(unwrapped_state_dict)

params = 0
params += sum(p.numel() for p in model.parameters())
print(f"no of params: {params}")


"""
quantization:
    : only weights for now
"""

CKPT_PATH = "checkpoints/ckpt_20000.pt"
CKPT_PATH_QUANT = "checkpoints/ckpt_int8.pt"
QUANTIZE_PER = "channel"
GROUP_SIZE = None

module_type_by_weight_name = {}
tied_weight_names = {} 


for mod_name, module in model.named_modules():
    if isinstance(module, (nn.Linear, nn.Embedding)):
        weight_name = f"{mod_name}.weight"
        mtype = "linear" if isinstance(module, nn.Linear) else "embedding"
        module_type_by_weight_name[weight_name] = mtype

        ptr = module.weight.data_ptr()
        if ptr in tied_weight_names:
            print(f"[tie detected] {weight_name} shares storage with {tied_weight_names[ptr]} "
                  f"-> will NOT be quantized separately")
        else:
            tied_weight_names[ptr] = weight_name

quantized_state = {}
quant_metadata  = {}

for name , tensor in unwrapped_state_dict.items():
    mtype = module_type_by_weight_name.get(name)

    is_tied_duplicate = (
        name in unwrapped_state_dict
        and tensor.data_ptr() in tied_weight_names
        and tied_weight_names[tensor.data_ptr()] != name
    )

    if tensor.dim() == 2 and mtype in ("linear", "embedding") and not is_tied_duplicate:
        w = tensor.float()
        q , scale , pad = quantize_(w , 
                          quantize_per_ = QUANTIZE_PER ,
                          group_size = GROUP_SIZE)
        quantized_state[f"{name}.qweight"] = q
        quantized_state[f"{name}.scale"] = scale

        quant_metadata[name] = {
            "quantize_per_": QUANTIZE_PER,
            "group_size": GROUP_SIZE,
            "pad": pad,
            "orig_shape": list(w.shape),
            "module_type": mtype,
        }

    else:
        quantized_state[name] = tensor

orig_bytes = sum(t.numel() * 4 for t in unwrapped_state_dict.values())
new_bytes = sum(t.numel() * (1 if t.dtype == torch.int8 else 4) for t in quantized_state.values())

print(f"Original: {orig_bytes/1e6:.2f} MB")
print(f"Quantized: {new_bytes/1e6:.2f} MB")
print(f"Compression: {orig_bytes/new_bytes:.2f}x")

torch.save({
    "model": quantized_state,
    "quant_metadata": quant_metadata,
    "model_config": ckpt["model_config"],
    "train_config": ckpt.get("train_config", {}),
}, CKPT_PATH_QUANT)

print(f"Saved : {CKPT_PATH_QUANT}")