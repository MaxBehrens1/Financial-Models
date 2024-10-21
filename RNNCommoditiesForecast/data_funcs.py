import yfinance as yf
import os
import torch

"""Functions to get data into the correct form for PyTorch:
    Silver: SI=F
    Gold: GC=F
    Copper: HG=F
    Platinum: PL=F
    Palladium: PA=F
"""

def generate_data(ticker = 'SI=F', dir = os.path.dirname(os.path.realpath(__file__)) + "/data/"):
    """Funciton that generates a data file given a ticker

    Args:
        ticker: Ticker of asset. Defaults to 'SI=F'.
        dir: Directory to save data file to. Defaults to os.path.dirname(os.path.realpath(__file__))+"/data/".
    """
    data = yf.download(tickers = ticker, 
                       period = 'max',
                       interval = '1d')
    
    len_data = len(data.loc[:,"Open"])
    open(dir + ticker + ".txt", 'w').close() # To empty file contents
    with open(dir + ticker + ".txt", "a") as write_file:
        write_file.write("#Open, High, Low, Close \n")
        for i in range(len_data):
            daily_data = data.iloc[i,[0,1,2,3]]
            write_file.write(str(daily_data[0]) + ", " + str(daily_data[1])+ ", "+ str(daily_data[2]) + ", " + str(daily_data[3]) + "\n")
        write_file.close()
    
    print(f"New data file in: {dir}")


def input_data(seq, ws):
    """Converts array of data into sliding window format

    Args:
        seq: Array of data 
        ws: Window size

    Returns:
        _type_: _description_
    """
    # Prepares data for LSTM window
    out = []
    L = len(seq)
    
    for i in range(L-ws):
        window = torch.tensor(seq[i:i+ws], dtype = torch.float32)
        label = torch.tensor(seq[i+ws:i+ws+1, 3], dtype = torch.float32)
        out.append((window,label))
    
    return out





    
    
    
    



