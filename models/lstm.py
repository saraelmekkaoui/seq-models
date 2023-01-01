import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from typing import Optional
import numpy as np 
import math

def build_dataset(batch_size):
  #create dataloaders
  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) 
  valid_loader = DataLoader(valid_data, batch_size=batch_size) 
  test_loader = DataLoader(test_data, batch_size=batch_size) 

  return train_loader, valid_loader, test_loader

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        self.lstm = nn.LSTM(input_dim, 
                            hidden_dim, 
                            layer_dim, 
                            batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out) 
        out = F.relu(out)

        return out

#Earlystopping adapted from "https://clay-atlas.com/us/blog/2021/08/25/pytorch-en-early-stopping/"
def train(model, epochs, optimizer, loss_function, train_loader, valid_loader, test_loader):
    # Early stopping
    the_last_loss = np.inf
    val_loss_best = np.inf
    patience = earlystop_patience
    trigger_times = 0

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0
        for batch, (X, y) in enumerate(train_loader, 1):
            X, y = X.to(device), y.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward and backward propagation
            outputs = model(X)
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Show progress

        train_loss = running_loss/len(train_loader)
        valid_loss = evaluation(model, valid_loader, loss_function)
        print('[{}/{}] train_loss is:{} ---- valid_loss is:{}'.format(epoch, epochs, train_loss, valid_loss))


        if valid_loss < val_loss_best:
            val_loss_best = valid_loss
            train_loss_best = train_loss
            best_epoch = epoch

        
         
        # Early stopping
        if valid_loss >= the_last_loss:
            trigger_times += 1
            print('trigger times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!')
                return model

        else:
            print('trigger times: 0')
            trigger_times = 0

        the_last_loss = valid_loss

    test_loss = evaluation(model, test_loader, loss_function)
    print('test_loss is:', test_loss)

    return model, train_loss, valid_loss, test_loss


def evaluation(model, data_loader, loss_function):
    # Settings
    model.eval()
    loss_total = 0

    # Test validation data
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            outputs = model(X)
            loss = loss_function(outputs, y)
            loss_total += loss.item()

    return loss_total/len(data_loader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(config=None):
  train_loader, valid_loader, test_loader = build_dataset(config.batch_size)
  model = LSTMModel(input_dim = input_size, 
                          hidden_dim = hidden_dim, 
                          layer_dim = layer_dim, 
                          output_dim = d_output).to(device)
  loss_function = nn.L1Loss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        

  # Train
  model, train_loss, valid_loss, test_loss = train(model, epochs, optimizer, loss_function, train_loader, valid_loader, test_loader)
