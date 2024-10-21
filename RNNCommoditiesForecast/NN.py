"""https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9141105/pdf/entropy-24-00657.pdf
"""

import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self,input_size = 4, hidden_size = 50, num_layers = 1, output_size = 1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, seq):
        h0 = torch.zeros(self.num_layers, seq.size(0), self.hidden_size, dtype=torch.float32)
        c0 = torch.zeros(self.num_layers, seq.size(0), self.hidden_size, dtype=torch.float32)
        
        out, _ = self.lstm(seq)
        out = self.linear(out[:,-1,:])
        return out
    