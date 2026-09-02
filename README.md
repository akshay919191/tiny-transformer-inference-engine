tiny-transformer-inference
A GPT-style decoder-only transformer built from scratch in PyTorch, with optionalTriton CUDA attention kernels. Trained on TinyStoriesusing the GPT-2 tokenizer.

Architecture
Decoder-only transformer with RoPE, RMSNorm, SwiGLU MLPs
Attention: MHA (multi-head) or MQA (multi-query, 1 shared KV head)
Two interchangeable backends: pytorch (reference) and cuda (Triton kernels)
KV cache + streaming generation (in progress)
Config	Value
vocab_size	50257 (GPT-2)
d_model	256
num_layers	8
num_heads	8
num_kv_heads	1 (MQA)
max_seq_len	512
Setup
pip install -r requirements.txt
Data
Tokenized TinyStories as uint16 binaries, produced with tiktoken (GPT-2 encoding):

data/├── train.bin└── val.bin
Training
python train.py --max_steps 5000 --lr 3e-4 --backend cuda --attn_type mqa
Flag	Default	Description
--max_steps	5000	training steps
--lr	3e-4	learning rate (AdamW)
--weight_decay	0.1	weight decay
--device	cuda	device
--backend	cuda	cuda (Triton) or pytorch
--attn_type	mqa	mqa or mha
Checkpoints are saved to checkpoints/ and store both the model config andthe training flags, so inference reproduces the exact setup.

Generation
python generation.py --ckpt checkpoints/ckpt_final.pt --prompt "Once upon a time" --token 100
Flag	Default	Description
--ckpt	checkpoints/ckpt_final.pt	checkpoint path
--prompt	"Once upon a time"	prompt text
--temperature	1.0	sampling temperature
--token	100	tokens to generate
Tokens are printed to the terminal as they are generated (streaming).

Project structure
models/        transformer blocks, attention (mqa.py), rope, rmsnorm, mlp, embeddingkernels/       Triton attention kernelsbenchmarks/    backend benchmarksconfig.py      runtime configtrain.py       training loopgeneration.py  streaming inference