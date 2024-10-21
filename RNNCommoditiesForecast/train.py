import os
import torch 
import pandas as pd
import numpy as np
import torch.nn as nn
from NN import LSTM
from sklearn.preprocessing import MinMaxScaler
from data_funcs import input_data
import matplotlib.pyplot as plt
#RNN predictive close price given yestrdays open, close, high, low

"""
Hyperparameters 
"""
window_size = 40
learning_rate = 0.01
h_size = 50
num_epochs = 10
test_percentage = 0.2 # Fraction of data that will be used for testing
layers = 1

    
"""
Data preparation
"""
# Loading in the commodities data as a pd dataframe
com_dir = os.path.dirname(os.path.realpath(__file__)) + "/data/GC=F.txt"
com_tsdata = pd.read_csv(com_dir, sep = ',', dtype="float64")

#Normalising data between 0 and 1
scaler = MinMaxScaler()
norm_com_tsdata = scaler.fit_transform(com_tsdata)
norm_com_tsdata = pd.DataFrame(norm_com_tsdata, dtype="float64")
norm_com_tsdata.columns = norm_com_tsdata.columns

# Split data into training and testing set
arr_ts_data =  norm_com_tsdata.to_numpy()
cut_off = round(len(arr_ts_data) * test_percentage)
training_set = arr_ts_data[:-cut_off]
testing_set = arr_ts_data[-cut_off:]

training_data = input_data(training_set, window_size)
testing_data = input_data(testing_set, window_size)

"""
Training the NN
"""
# Initialising model
in_size = len(training_data[0][0][0]) 
out_size = len(training_data[0][1])
model = LSTM(input_size = in_size, hidden_size = h_size, num_layers = layers, output_size = out_size)

#Setting loss and optimniser functions
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_loss =0
    for seq, y_train in training_data:
        seq = torch.tensor(seq, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        outputs = model(seq.unsqueeze(-1)).squeeze()
        optimizer.zero_grad()
        loss = criterion(outputs, y_train)
        total_loss += loss
        loss.backward()
        optimizer.step()
        
    print(f'Epoch: {epoch+1:2}, Loss: {total_loss.item():10.8f}')






    