import torch
from torch import nn
import numpy as np

class TestNet(nn.Module):
    def __init__(self, seq_len=10):
        super(TestNet, self).__init__()
        
        self.seq_len = seq_len
        self.LSTM = nn.LSTM(input_size=1, hidden_size=16, num_layers=1, batch_first=True)
        self.linear1 = nn.Linear(16*self.seq_len,10)
        self.linear2 = nn.Linear(10,1)
    def forward(self, x):
        # out = (batch, seq_len, num_directions * hidden_size)
        # hidden = (hn, cn)，维度同上
        x, hidden = self.LSTM(x)
        x = x.reshape(-1, 16*self.seq_len)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    