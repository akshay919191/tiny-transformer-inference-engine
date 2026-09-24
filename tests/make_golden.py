import os
import argparse
import torch
import torch.nn.functional as F
import tiktoken

from models.transformer_block import Transformer
from models.model_config import ModelConfig
from kv_cache import KVCache_kv
from kernels.capability import resolve_backend

enc = tiktoken.get_encoding("gpt2")


def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y")


def load_model(ckpt_path, device, attn_type=None, backend=None, causal=None):
    # 1. Use your custom class method to handle the read-only properties safely
    run_time = ModelConfig.from_checkpoint(ckpt_path)
    
    # Reload checkpoint metadata container for secondary runtime properties
    ckpt = torch.load(ckpt_path, map_location=device)
    train_cfg = ckpt.get("train_config", {})

    # Apply manual configuration overrides if specified at runtime
    resolved_attn_type = attn_type if attn_type is not None else train_cfg.get("attn_type", "mqa")
    requested_backend = backend if backend is not None else train_cfg.get("backend", "pytorch")

    resolved_backend = resolve_backend(requested_backend, run_time, resolved_attn_type)

    if causal is not None:
        run_time.causal = causal

    print(f"[DEBUG] Loading model with attn_type={resolved_attn_type}, backend={resolved_backend}, causal={run_time.causal}")

    model = Transformer(run_time, attn_type=resolved_attn_type, backend=resolved_backend)
    
    state_dict = ckpt["model"]
    unwrapped_state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
    }
    model.load_state_dict(unwrapped_state_dict)

    # Force float32 evaluation to maintain precise golden tracking parameters
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    return model, run_time


def prefill(model, tokens, kv_cache):
    with torch.no_grad():
        logits = model(tokens, kv_cache=kv_cache)
    return logits[:, -1, :]


def decode_one(model, next_token, kv_cache):
    with torch.no_grad():
        logits = model(next_token, kv_cache=kv_cache)
    return logits[:, -1, :]


def generate_greedy_golden(model, run_time, device, prompt_ids_list, max_new_tokens=50):
    ids = torch.tensor(prompt_ids_list, dtype=torch.long, device=device).unsqueeze(0)

    kv_heads = getattr(run_time, "num_kv_heads", run_time.num_heads)
    head_dim = run_time.head_dim 
    model_dtype = next(model.parameters()).dtype

    # FIX: Convert the shape structure into a tuple of primitive Python integers 
    batch_size = ids.shape[0]

    kv_cache = KVCache_kv(
        num_layers=run_time.num_layers,
        batch_size=batch_size,
        num_heads=kv_heads,
        max_seq_len=run_time.max_seq_len,
        head_dim=head_dim,
        dtype=model_dtype,  
        device=device,
    )

    generated_ids = []

    logits = prefill(model, ids, kv_cache)
    next_id = torch.argmax(logits, dim=-1, keepdim=True)
    generated_ids.append(next_id.item())

    for _ in range(max_new_tokens - 1):
        logits = decode_one(model, next_id, kv_cache)
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        generated_ids.append(next_id.item())

    return torch.tensor(generated_ids, dtype=torch.long, device="cpu")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt_step20000.pt")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default="tests/golden.pt")
    p.add_argument("--attn_type", type=str, default=None, choices=[None, "mqa", "mha"])
    p.add_argument("--backend", type=str, default=None, choices=[None, "cuda", "pytorch", "auto"])
    p.add_argument("--causal", type=str2bool, default=None)
    args = p.parse_args()

    model, run_time = load_model(
        args.ckpt,
        args.device,
        attn_type=args.attn_type,
        backend=args.backend,
        causal=args.causal,
    )

    # Paste your own raw strings into these variables
    p1 = "One"

    p15 = "Once upon a time, there was a little boy named Tim who liked big hats."

    p16 = "One sunny morning, a small girl named Lily went outside to play with her friendly dog."

    p17 = "Tom found a shiny green apple on the kitchen table and ate it up because he was hungry."

    p100 = ("Once upon a time, there was a small bird named Pip. Pip lived in a big green tree. "
            "Every day, he liked to fly high in the blue sky. One morning, he saw a big cat on the grass. "
            "The cat was sleeping. Pip wanted to find some food, so he flew down very quietly. He found a nice, "
            "juicy worm near a red flower. Pip was very happy. He ate the worm and flew back up into his safe tree. "
            "He sang a happy song until the sun went down behind the hills.")
            
    p150 = ("Once upon a time, in a small village, lived a happy little girl named Mia. Mia loved to explore the big forest "
            "behind her house. She always walked with her favorite teddy bear, Ben. One day, while walking under the tall trees, "
            "Mia saw something shiny hidden in the soft green grass. She ran over and found a small silver key. 'Look, Ben!' "
            "she said. Mia wondered what the key could open. She looked around and saw an old wooden box sitting next to a big stone. "
            "Mia ran to the box, put the key inside the lock, and turned it slowly. Click! The box opened up. Inside the box, "
            "there was a beautiful yellow ball that could glow in the dark. Mia laughed with joy and spent the rest of the day "
            "playing with her new glowing ball until her mom called her home for dinner.")

    p200 = ("Once upon a time, there was a kind old man named Leo who lived in a tiny house near a big blue lake. Leo loved "
            "to catch fish, but he never kept them. He always put them back into the clean water. One bright afternoon, Leo sat on "
            "his wooden chair and threw his long fishing line into the water. Suddenly, something pulled the line very hard! "
            "Leo stood up and pulled with all his strength. Out of the lake jumped a big fish with bright pink scales that sparkled "
            "in the bright sunshine. The pink fish looked at Leo and said, 'Please let me go, kind man.' Leo smiled and gently "
            "took the hook out. He placed the pink fish back into the calm lake. Before swimming away, the fish flipped its tail "
            "and left a tiny golden stone on the grass. Leo picked up the stone and realized it made his hands feel warm and safe. "
            "Leo carried the special stone back inside his house and placed it safely on his table, happy to have a magical new friend.")

    p250 = ("Once upon a time, a little puppy named Max lived on a big farm with lots of animals. Max was very friendly, but he was "
            "also very clumsy. He always bumped into things and tripped over his own long ears. One morning, the farmer left the big front gate "
            "open by mistake. Max was curious, so he trotted out into the wide green meadow to explore. He chased a yellow butterfly and ran "
            "around in circles until he was far away from the farm. Suddenly, the sky grew dark and heavy gray clouds appeared. Big drops of rain "
            "started falling from the sky. Max was scared and did not know how to get back home. He crawled under a large green leaf to stay dry "
            "and started to cry. Just then, a friendly cow named Daisy walked past. Daisy saw the sad little puppy and walked over. 'Don't cry, Max,' "
            "Daisy said softly. 'Follow me!' Max hopped out from under the leaf and walked safely right behind Daisy. She guided him all the way "
            "back through the wet grass straight to the warm barn. Max was so happy to be home that he barked joyfully and curled up to sleep.")

    p350 = ("Once upon a time, in a magical garden full of tall flowers, lived a tiny green frog named Sam. Sam loved to jump from one big wet lily pad "
            "to another all day long. But Sam had a secret dream. He did not want to just hop on the ground; he wanted to fly high in the air like the "
            "beautiful butterflies. Every afternoon, Sam watched the colorful bugs fly around the red roses and wished he could join them. One day, a wise "
            "old owl named Oliver flew down and sat on a branch above Sam. 'Why do you look so sad, little frog?' Oliver asked. Sam looked up and said, "
            "'I want to see the world from high up in the sky, but frogs can only hop.' Oliver smiled gently. 'You do not need wings to see the world from "
            "above, Sam. Climb onto my back!' Sam was very excited. He took a giant leap and landed safely right between the owl's soft feathers. Oliver flapped "
            "his big wings and soared high up into the fresh air. Sam looked down and gasped with delight. He could see the whole garden, the blue river, "
            "and the tiny houses far away. It was the happiest day of his life, and he realized that sometimes, asking for help can make your biggest "
            "dreams come true.")

    p450 = ("Once upon a time, in a deep forest where the trees whispered secrets to the wind, lived a small squirrel named Nutty. Nutty was very good at "
            "gathering brown acorns for the cold winter months, but he was also very forgetful. He hid his acorns in many different places under the ground "
            "but could never remember where he put them. One chilly morning, Nutty woke up and felt his tummy rumble. He went outside to find one of his food "
            "spots, but the ground was completely covered in a blanket of soft white snow. He dug in the snow near a rock, but found nothing. He dug near an "
            "old log, but still found nothing. Nutty sat on a branch and started to look very sad because his tummy was so empty. Seeing this, a kind red bird "
            "named Ruby flew down and landed right next to him. 'What is the matter, Nutty?' Ruby asked. Nutty sighed and said, 'I hid all my winter acorns, "
            "but the white snow hid them from me, and I am so hungry.' Ruby chirped happily. 'Do not worry, my friend! I fly high in the sky and can see things "
            "from above.' Ruby flew up and looked down at the white snow. With her sharp eyes, she saw a tiny brown top sticking out near a pine tree. She "
            "flew down and scratched the snow away with her feet, revealing a big pile of clean acorns. Nutty scampered down the tree as fast as he could. "
            "He hugged Ruby and thanked her for being such an amazing friend. They sat together under the pine tree, sharing the delicious food while the winter "
            "wind blew around them, knowing they would always look out for one another.")

    prompts_pool = [p1, p15, p16, p17, p100, p150, p200, p250, p350, p450]
    print(f"\nGenerating golden samples...")
    golden_data = []

    for i, prompt in enumerate(prompts_pool):
        prompt_ids_list = enc.encode_ordinary(prompt)
        prompt_len = len(prompt_ids_list)
        
        if prompt_len + 50 >= 512:
            print(f"⚠️ Prompt {i+1} total tokens ({prompt_len + 50}) equals/exceeds 512! Skipping.")
            continue
            
        print(f"Processing Prompt {i+1}/{len(prompts_pool)} | Length: {prompt_len} tokens")
        
        generated_ids = generate_greedy_golden(
            model=model,
            run_time=run_time,
            device=args.device,
            prompt_ids_list=prompt_ids_list,
            max_new_tokens=50
        )
        
        golden_data.append({
            "prompt_ids": torch.tensor(prompt_ids_list, dtype=torch.long),
            "generated_ids": generated_ids
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(golden_data, args.output)
    print(f"\nSuccessfully stored {len(golden_data)} regression profiles to '{args.output}' ✅")
