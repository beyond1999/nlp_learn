from scipy._lib.array_api_compat import torch
from torch import nn

rnn = nn.RNN(input_size=3, hidden_size=4, num_layers=2, bidirectional=True)


# input.shape = [batch_size, seq_len, input_size]
input = torch.randn(2, 4, 3)


#output.shape: [batch_size, seq_len, 2*hidden_size]
# hn.shape: [num_layers * num_direction, batch_size, hidden_size]
output, hidden = rnn(input)
print(output.shape)
print(hidden.shape)