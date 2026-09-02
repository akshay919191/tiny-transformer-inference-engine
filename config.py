from models.model_config import ModelConfig
""" here we can modify the config for model and for inference too"""

from dataclasses import dataclass

@dataclass
class CONFIG:
    backend = "cuda"
    attn = "mqa"
    promptlength = 512
    max_steps = 5000