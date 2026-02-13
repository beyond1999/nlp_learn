import math
import random
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# -----------------------------
# 1) Toy dataset: reverse a sequence
# -----------------------------
PAD = 0
SOS = 1
EOS = 2

def generate_batch(batch_size=32, min_len=3, max_len=10, vocab_size=50, device="cpu"):
    """
    Generate random sequences x (excluding special tokens),
    target y = reversed(x) with EOS.
    """
    xs, ys, x_lens, y_lens = [], [], [], []
    for _ in range(batch_size):
        L = random.randint(min_len, max_len)
        x = [random.randint(3, vocab_size - 1) for _ in range(L)]
        y = list(reversed(x)) + [EOS]
        xs.append(x)
        ys.append(y)
        x_lens.append(len(x))
        y_lens.append(len(y))

    max_x = max(x_lens)
    max_y = max(y_lens)

    x_pad = torch.full((batch_size, max_x), PAD, dtype=torch.long)
    y_pad = torch.full((batch_size, max_y), PAD, dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        x_pad[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        y_pad[i, :len(y)] = torch.tensor(y, dtype=torch.long)

    return x_pad.to(device), torch.tensor(x_lens).to(device), y_pad.to(device), torch.tensor(y_lens).to(device)

# -----------------------------
# 2) Bahdanau Attention module
# -----------------------------
class BahdanauAttention(nn.Module):
    """
    e_t^i = v^T tanh(W_s s_t + W_h h^i)
    alpha = softmax(e)
    context = sum alpha_i * h^i
    """
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.W_h = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_s = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, enc_out, enc_mask, dec_state):
        """
        enc_out: (B, T, enc_dim)
        enc_mask: (B, T) 1 for valid tokens, 0 for PAD
        dec_state: (B, dec_dim)
        """
        # Project
        # (B, T, attn_dim)
        Wh = self.W_h(enc_out)
        # (B, 1, attn_dim)
        Ws = self.W_s(dec_state).unsqueeze(1)

        # Energy: (B, T, 1) -> (B, T)
        e = self.v(torch.tanh(Wh + Ws)).squeeze(-1)

        # Mask PAD positions to -inf so softmax ignores them
        e = e.masked_fill(enc_mask == 0, float("-inf"))

        alpha = F.softmax(e, dim=-1)  # (B, T)
        # Context: (B, 1, T) @ (B, T, enc_dim) -> (B, 1, enc_dim) -> (B, enc_dim)
        context = torch.bmm(alpha.unsqueeze(1), enc_out).squeeze(1)

        return context, alpha

# -----------------------------
# 3) Encoder / Decoder with Attention
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)

    def forward(self, x, x_lens):
        # x: (B, T)
        emb = self.emb(x)  # (B, T, emb_dim)

        # pack for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(emb, x_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, h = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (B, T, 2*hid_dim)

        return out  # bidirectional outputs only (we'll not use final state here)

class AttnDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.attn = BahdanauAttention(enc_dim=enc_dim, dec_dim=dec_dim, attn_dim=attn_dim)
        self.gru = nn.GRU(emb_dim + enc_dim, dec_dim, batch_first=True)
        self.out = nn.Linear(dec_dim + enc_dim, vocab_size)

    def forward(self, enc_out, enc_mask, y_in):
        """
        Teacher forcing:
        y_in: (B, U) decoder input tokens (e.g., SOS + target[:-1])
        returns logits: (B, U, vocab)
        """
        B, U = y_in.shape
        enc_dim = enc_out.size(-1)

        emb = self.emb(y_in)  # (B, U, emb_dim)

        # initial decoder state = zeros
        dec_state = torch.zeros(1, B, self.gru.hidden_size, device=enc_out.device)  # (1,B,dec_dim)

        logits = []
        attn_weights = []

        for t in range(U):
            # Current input embedding: (B, emb_dim)
            emb_t = emb[:, t, :]

            # Current dec_state: (B, dec_dim)
            s_t = dec_state.squeeze(0)

            # Attention context: (B, enc_dim)
            c_t, alpha_t = self.attn(enc_out, enc_mask, s_t)

            # GRU input: concat(emb_t, c_t) -> (B, emb_dim+enc_dim) -> (B,1, ...)
            gru_in = torch.cat([emb_t, c_t], dim=-1).unsqueeze(1)
            out_t, dec_state = self.gru(gru_in, dec_state)  # out_t: (B,1,dec_dim)

            out_t = out_t.squeeze(1)  # (B, dec_dim)

            # output distribution uses both decoder state and context
            logit_t = self.out(torch.cat([out_t, c_t], dim=-1))  # (B, vocab)
            logits.append(logit_t.unsqueeze(1))
            attn_weights.append(alpha_t.unsqueeze(1))

        return torch.cat(logits, dim=1), torch.cat(attn_weights, dim=1)

# -----------------------------
# 4) Train
# -----------------------------
@dataclass
class Config:
    vocab_size: int = 60
    emb_dim: int = 64
    hid_dim: int = 64
    dec_dim: int = 128
    attn_dim: int = 128
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    cfg = Config()
    device = cfg.device
    print("Device:", device)

    encoder = Encoder(cfg.vocab_size, cfg.emb_dim, cfg.hid_dim).to(device)
    # encoder is biGRU => enc_dim = 2*hid_dim
    enc_dim = 2 * cfg.hid_dim
    decoder = AttnDecoder(cfg.vocab_size, cfg.emb_dim, enc_dim, cfg.dec_dim, cfg.attn_dim).to(device)

    optim = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    steps = 1000
    current_time = time.perf_counter()
    for step in range(1, steps + 1):
        x, x_lens, y, y_lens = generate_batch(batch_size=64, vocab_size=cfg.vocab_size, device=device)

        # encoder outputs
        enc_out = encoder(x, x_lens)  # (B, T, enc_dim)

        # mask: 1 for real tokens, 0 for PAD
        enc_mask = (x != PAD).long()

        # decoder input: SOS + y[:-1]
        B, U = y.shape
        y_in = torch.full((B, U), PAD, dtype=torch.long, device=device)
        y_in[:, 0] = SOS
        y_in[:, 1:] = y[:, :-1]

        logits, attn = decoder(enc_out, enc_mask, y_in)  # logits: (B, U, vocab)

        # compute loss against y
        loss = loss_fn(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
        optim.step()

        if step % 100 == 0:
            print(f"step {step:4d} | loss {loss.item():.4f}")
            print("time usage:", time.perf_counter() - current_time)
            current_time = time.perf_counter()
            print("-------")

    # Quick demo: greedy decode one example and print attention shape
    x, x_lens, y, _ = generate_batch(batch_size=1, vocab_size=cfg.vocab_size, device=device)
    enc_out = encoder(x, x_lens)
    enc_mask = (x != PAD).long()

    # Greedy decode
    max_len = x_lens.item() + 1
    ys = [SOS]
    dec_state = torch.zeros(1, 1, decoder.gru.hidden_size, device=device)
    attn_all = []

    for _ in range(max_len):
        y_in = torch.tensor([[ys[-1]]], device=device)
        emb_t = decoder.emb(y_in).squeeze(1)
        s_t = dec_state.squeeze(0)
        c_t, alpha_t = decoder.attn(enc_out, enc_mask, s_t)
        gru_in = torch.cat([emb_t, c_t], dim=-1).unsqueeze(1)
        out_t, dec_state = decoder.gru(gru_in, dec_state)
        out_t = out_t.squeeze(1)
        logit = decoder.out(torch.cat([out_t, c_t], dim=-1))
        pred = int(logit.argmax(dim=-1).item())
        attn_all.append(alpha_t.detach().cpu())
        ys.append(pred)
        if pred == EOS:
            break

    print("Input x:", x[0].tolist())
    print("Target y:", y[0].tolist())
    print("Pred   y:", ys[1:])
    print("Attention steps:", len(attn_all), "each over T =", x_lens.item())

if __name__ == "__main__":
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("current device:", torch.cuda.current_device())
        print("device name:", torch.cuda.get_device_name(0))

    main()
