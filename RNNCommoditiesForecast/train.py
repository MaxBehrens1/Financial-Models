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
num_epochs = 100
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
model = LSTM(input_size = in_size, hidden_size = h_size, num_layers = layers, output_size = out_size, seq_size = 1)  

#Setting loss and optimniser functions
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Starting training")
data_loss = []
for epoch in range(num_epochs):
    # Updates the model after every 'window' input
    total_loss = 0
    model.train()
    for inputs, expected in training_data:
        optimizer.zero_grad()
        model.hidden = (torch.zeros(layers,1,model.hidden_size),
                       torch.zeros(layers,1,model.hidden_size))
        
        inputs = inputs.unsqueeze(0) # To turn into 3D array: [seq_len (1 for this case as we update after every step), window size, input size]
        prediction = model(inputs)
        prediction = prediction.squeeze(1) # To turn into 1D array to compare to expected
        loss = criterion(prediction, expected)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(training_data)
    data_loss.append(average_loss)
    print(f"Epoch {epoch+1}, Loss {total_loss}")
    
plt.plot(data_loss, 'x')
plt.grid()
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
    