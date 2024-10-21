import os
import torch 
import pandas as pd
import numpy as np
import torch.nn as nn
from NN import LSTM
from sklearn.preprocessing import MinMaxScaler
from data_funcs import input_data
import matplotlib.pyplot as plt
from tqdm import tqdm
#RNN predictive close price given yestrdays open, close, high, low

"""
Hyperparameters 
"""
window_size = 40
learning_rate = 0.01
h_size = 50
num_epochs = 60
test_percentage = 0.2 # Fraction of data that will be used for testing
layers = 1
batch_size = 10

    
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

training_data = training_data[:-(len(training_data)%batch_size)] # To make training_data a multiple of batch size
batches = np.arange(0, len(training_data) + batch_size, batch_size) # Indexes at which the ends of the batches are located


"""
Training Model
"""
# Initialising model
in_size = len(training_data[0][0][0]) 
out_size = len(training_data[0][1])
model = LSTM(input_size = in_size, hidden_size = h_size, num_layers = layers, output_size = out_size, seq_size = batch_size)  

#Setting loss and optimniser functions
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Starting training:")
loss_data = []
for epoch in range(num_epochs):
    # Inputs batch_size number of windows at a time
    total_loss = 0
    model.train()
    for i in tqdm(range(len(batches)-1)):
        #input and expected data
        data = training_data[batches[i]:batches[i+1]]
        inputs = torch.stack([j[0] for j in data])
        expected = torch.stack([j[1] for j in data])
        model.h0c0 = (torch.zeros(layers,batch_size,model.hidden_size),
                       torch.zeros(layers,batch_size,model.hidden_size))
        
        optimizer.zero_grad()
        outputs = model(inputs) 
        
        # Calculates loss
        loss = criterion(outputs, expected)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss / (len(batches)-1)
    loss_data.append(average_loss)
    print(f"Epoch {epoch+1}, Loss {average_loss}")

plt.plot(loss_data)
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


"""
Testing Model
"""
model.eval()
model.h0c0 = (torch.zeros(1, len(arr_ts_data), model.hidden_size),
                       torch.zeros(1, len(arr_ts_data), model.hidden_size)) #[layers, batch size, hidden layer]


full_data = torch.tensor(arr_ts_data, dtype = torch.float32).unsqueeze(1) #[batch size, window size, inputs]
train_predict = model(full_data)

#train_predict = train_predict.squeeze()
data_predict = train_predict.detach().numpy()
dataY_plot = norm_com_tsdata.iloc[:,3].to_numpy()

# Need to make Lx4 array to inverse_tranform
dataY_plot = [[i,0,0,0] for i in dataY_plot]
data_predict = [[i[0],0,0,0] for i in data_predict]
dataY_plot = scaler.inverse_transform(dataY_plot)
data_predict = scaler.inverse_transform(data_predict)
dataY_plot = [i[0] for i in dataY_plot]
data_predict = [i[0] for i in data_predict]

plt.plot(dataY_plot, label = "Real data")
plt.xlabel("Days")
plt.ylabel("Price")
plt.plot(data_predict, lebl = "Prediction")
plt.grid()
plt.legend()
plt.title('Time-Series Prediction')
plt.show()
