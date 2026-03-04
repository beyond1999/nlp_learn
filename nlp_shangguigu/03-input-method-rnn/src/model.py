import torch
import config
from torch import nn

class InputMethodModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.EMBEDDING_DIM)
        # self.embedding = nn.Embedding().from_pretrained()
        self.rnn = nn.RNN(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.HIDDEN_SIZE,
                          num_layers=1,
                          batch_first=True)

        self.linear = nn.Linear(in_features=config.HIDDEN_SIZE,out_features=vocab_size)

    def forward(self, x):
        # x.shape: [batch_size, seq_len]

        embed = self.embedding(x)

        # embed.shape: [batch_size, seq_len, embedding_dim]
        output, _ = self.rnn(embed)
        # output, hn = self.rnn(embed)
        # output.shape: [batch_size, seq_len. hidden_size]
        # 为什么要切,因为output是每个时间步的输出的集合 [{h_i}]，所以要切出最后一个

        last_hidden_state = output[:, -1, :]
        # last_hidden_state.shape 变为二维了 [batch_size, hidden_size]

        output = self.linear(last_hidden_state)
        # output.shape: [batch_size, vocab_size]
        return output