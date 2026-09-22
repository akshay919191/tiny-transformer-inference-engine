import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_path = "checkpoints/ckpt_step20000.pt"
ckpt = torch.load(ckpt_path , map_location = DEVICE)

state_dict = ckpt["model"]

## torch.compile name fixing
unwrapped_state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
    }
