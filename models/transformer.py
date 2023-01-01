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


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=35):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

#Adapted from "https://n8henrie.com/2021/08/writing-a-transformer-classifier-in-pytorch/"

class Transformer(nn.Module):
    """
    Transformer model adapted for sequential data based on a pytorch TransformerEncoder.
    Embedding layer in standard Transformer is replaced by a fully connected layer.
    The last layer is a regressor using a fully connected layer.
    
    Parameters
    ----------
    d_input:
        Model input dimension.
    d_model:
        Dimension of the input vector to the encoder.
    d_output:
        Model output dimension.
    nhead:
        the number of heads in the multiheadattention models.
    dim_feedforward:
        the dimension of the feedforward network model).
    dropout:
        the dropout value of the TransformerEncoderLayer (after MultiheadAttention and Feedforward blocks).
    num_layers:
        Number of sub-encoder-layers in the encoder
    """

    def __init__(
        self,
        d_input,
        d_model,
        d_output,
        nhead,
        dim_feedforward,
        num_layers,
        dropout
        ):

        super().__init__()

        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        self.d_model = d_model

        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)
      

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        x = self._embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self._linear(x)
        x = F.relu(x)
        
        return x

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
def main():
  train_loader, valid_loader, test_loader = build_dataset(batch_size)
  model = Transformer(d_input=input_size,
                            d_model=d_model,
                            d_output=d_output,
                            nhead=nhead,
                            dim_feedforward=dim_feedforward,
                            num_layers=num_layers,
                            dropout=dropout).to(device)
  loss_function = nn.L1Loss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=log_weight_decay)
        

  # Train
  model, train_loss, valid_loss, test_loss = train(model, epochs, optimizer, loss_function, train_loader, valid_loader, test_loader)    

  torch.save(model, "model.h5")
