import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import argparse


from models.transformer_block import Transformer , Transformer_nocache , make_kv_cache , make_kv_cache_
from models.model_config import ModelConfig
from config import CONFIG
from kv_cache import KVCache_kv
from sampling import sample
from generation import generate , generate_nocache , test_first_decode


"""building parse for CLI type"""
parser = argparse.ArgumentParser()


parser.add_argument("--batch" , default = 2 , nargs = '?' , type = int , help = "batchsize")
parser.add_argument("--attn"  , default = "mqa" , nargs = '?' , type = str , help = "attn_type")
parser.add_argument("--promptsize" , default = 512 , nargs = '?' , type = int , help = "prompt_length")
parser.add_argument("--token" , default = 20 , nargs = '?' , type = int , help = "number_token_generation")
parser.add_argument("--backend" , default = "cuda" , nargs = '?' , type = str , help = "pytorch_vs_cuda")

args = parser.parse_args()


if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model_config = ModelConfig()
    runtime_config = CONFIG()

    print(runtime_config.casual)

    model_config.batch = args.batch

    model_cache = Transformer(
            model_config ,
            attn_type = args.attn,
            backend = args.backend
        )

    model_nocache = Transformer_nocache(
            model_config ,
            attn_type = args.attn,
            backend = args.backend
        )


    model_nocache.load_state_dict(
        model_cache.state_dict()
    )

    B = args.batch
    T = runtime_config.promptlength

    prompt_tokens = torch.randint(
        0 ,
        model_config.vocab_size,
        (B , T),
        device = device
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


    print("\n\n")
    print("########################################")
    print("#          WITHOUT KV CACHE            #")
    print("########################################")

    generated_nocache = generate_nocache(
        model_nocache,
        prompt_tokens,
        max_new_tokens=args.token,
    )


    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("\n\n")
    print("########################################")
    print("#           WITH KV CACHE               #")
    print("########################################")

    generated_cache = generate(
        model_cache,
        prompt_tokens,
        max_new_tokens=args.token,
        config=model_config,
    )


    print()
    print("Prompt shape:")
    print(prompt_tokens.shape)

    print()
    print("No-cache output:")
    print(generated_nocache.shape)

    print()
    print("KV-cache output:")
    print(generated_cache.shape)

    same = torch.equal(
        generated_nocache,
        generated_cache,
    )

    print()
    

    with torch.no_grad():

        logits_nocache = model_nocache(
            prompt_tokens
        )

        prefill_cache = make_kv_cache_(
            model_cache,
            model_config,
            batch_size=B,
            max_seq_len=T + 1,
            device=device,
        )

        logits_cache = model_cache(
            prompt_tokens,
            kv_cache=prefill_cache,
        )

        diff = (
            logits_nocache
            -
            logits_cache
        ).abs().max()

        print(
            "Prefill max diff:",
            diff.item()
        )

    test_first_decode(
        model_nocache,
        model_cache,
        prompt_tokens,
        model_config,
        attn_type = args.attn
    )