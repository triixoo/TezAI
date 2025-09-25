import torch
import torch.nn as nn
import torch.nn.functional as F

class TezAILM(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, max_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        pos = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)

        x = self.embed(input_ids) + self.pos_embed(pos)
        x = self.transformer(x)
        logits = self.fc_out(x)
        return logits

    def generate(self, input_ids, tokenizer, max_new_tokens=30):
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return tokenizer.decode(input_ids[0].tolist())
