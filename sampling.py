import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import argparse

from kernels.kernel import TopK

topk = TopK()



def smaple_temperature(
        logits,
        temp = 0.4
):
    if temp <= 0:
        raise ValueError("must be > 0")

    logits = logits / temp

    logits = nn.functional.softmax(logits , dim = -1)

    next_token = torch.multinomial(
        logits , num_samples = 1
    )

    return next_token

def apply_top_p(logits, top_p):

    if top_p >= 1.0:
        return logits

    if top_p <= 0.0:
        raise ValueError("top_p must be > 0")

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
        dim=-1,
    )

    sorted_probs = F.softmax(
        sorted_logits,
        dim=-1,
    )

    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=-1,
    )

    remove = cumulative_probs > top_p

    remove[:, 1:] = remove[:, :-1].clone()

    remove[:, 0] = False

    sorted_logits = sorted_logits.masked_fill(
        remove,
        float("-inf"),
    )

    logits = torch.full_like(
        logits,
        float("-inf"),
    )

    logits.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_logits,
    )

    return logits

def apply_top_k(logits, topk_digit):

    if topk_digit <= 0:
        return logits

    K = min(
        topk_digit,
        logits.shape[-1]
    )

    result = topk(
        logits.float().contiguous(),
        K
    )

    top_values = result[0]

    kth = top_values[:, -1].unsqueeze(-1)

    return logits.masked_fill(
        logits < kth,
        float("-inf")
    )

def sample(
    logits,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
):

    if temperature <= 0:
        raise ValueError(
            "temperature must be > 0"
        )


    logits = logits / temperature

    if top_k > 0:
        logits = apply_top_k(
            logits,
            top_k,
        )

    if top_p < 1.0:
        logits = apply_top_p(
            logits,
            top_p,
        )

    probs = F.softmax(
        logits,
        dim=-1,
    )


    next_token = torch.multinomial(
        probs,
        num_samples=1,
    )

    return next_token

