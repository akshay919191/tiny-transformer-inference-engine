import torch
from serving.sequence import Sequence
from serving.batch import build_batch_step
from sampling import sample


class Scheduler:
    def __init__(self, model, pool, alloc, cfg, block_size=16,
                 temperature=1.0, top_k=32, top_p=0.9):
        self.model = model
        self.pool = pool
        self.alloc = alloc
        self.cfg = cfg
        self.block_size = block_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.device = next(model.parameters()).device

        self.waiting = []   
        self.running = []   
        self.next_seq_id = 0

    def add_request(self, prompt_ids, max_new_tokens):
        self.waiting.append({"prompt_ids": prompt_ids, "max_new_tokens": max_new_tokens,
                              "generated": 0})

    def _try_admit(self):
        while self.waiting:
            req = self.waiting[0]
            seq = Sequence(self.next_seq_id, list(req["prompt_ids"]), self.block_size)
            needed = seq.num_blocks_needed()

            reserved = len(self.running) + 1

            if self.alloc.num_free - needed < reserved:
                break

            seq.ensure_blocks(self.alloc)
            seq._meta = req
            seq._is_prefill = True
            self.running.append(seq)
            self.waiting.pop(0)
            self.next_seq_id += 1

    def step(self):
        self._try_admit()
        if not self.running:
            return []   

        entries = []
        for seq in self.running:
            if seq._is_prefill:
                entries.append((seq, seq.token_ids))       
            else:
                entries.append((seq, [seq.token_ids[-1]])) 

        batch = build_batch_step(entries, self.device)
        logits = self.model.forward_batch(batch, self.pool)

        results = []
        still_running = []
        for i, seq in enumerate(self.running):
            tok = int(sample(logits[i].unsqueeze(0), self.temperature, self.top_k, self.top_p))
            seq._is_prefill = False
            seq._meta["generated"] += 1
            done = (tok == self.cfg.eos_token_id) or (seq._meta["generated"] >= seq._meta["max_new_tokens"])

            results.append((seq.seq_id, tok, done))

            if done:
                seq.release(self.alloc)     
            else:
                seq.append_token(tok)
                seq.ensure_blocks(self.alloc)
                still_running.append(seq)

        self.running = still_running
        return results