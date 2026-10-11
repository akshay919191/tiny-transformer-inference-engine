"""Compatibility helpers; use ``python inference.py`` for the command line."""

import tiktoken

from inference import generate_ids, load_model

_tokenizer = tiktoken.get_encoding("gpt2")


def generate(model, run_time, device, prompt, max_new_tokens=100,
             temperature=0.9, top_k=20, top_p=1.0):
    ids = _tokenizer.encode_ordinary(prompt)
    output = generate_ids(model, run_time, ids, max_new_tokens,
                          temperature=temperature, top_k=top_k, top_p=top_p)
    previous = ""
    for index in range(len(output)):
        decoded = _tokenizer.decode(output[:index + 1])
        yield decoded[len(previous):]
        previous = decoded


__all__ = ["load_model", "generate", "generate_ids"]
