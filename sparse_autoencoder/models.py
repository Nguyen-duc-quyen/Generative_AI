import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear_Encoder(nn.Module):
    """
        Encoder module based on Fully Connected Layers
    """
    def __init__(self, in_features, hidden_features, embedding_features):
        super().__init__()
        self.hidden_features = hidden_features
        
        # If don't use any hidden layers
        if len(hidden_features) == 0:
            self.emb = nn.Linear(in_features, embedding_features)
        else:
            self.ln_0 = nn.Linear(in_features, hidden_features[0])
            for i in range(1, len(hidden_features)):
                setattr(self, "ln_{}".format(i), nn.Linear(hidden_features[i-1], hidden_features[i]))
            
            self.emb = nn.Linear(hidden_features[-1], embedding_features)

    
    def forward(self, x):
        if len(self.hidden_features) == 0:
            out = self.emb(x)

        else:
            out = self.ln_0(x)
            out = F.relu(out)

            for i in range(1, len(self.hidden_features)):
                out = getattr(self, "ln_{}".format(i))(out)
                out = F.relu(out)
            
            out = self.emb(out)
        return out
    

class Linear_Decoder(nn.Module):
    """
        Decoder module based on Fully Connected Layers
    """
    def __init__(self, embedding_features, hidden_features, output_features):
        super().__init__()
        self.hidden_features = hidden_features
        
        # If don't use any hidden layers
        if len(hidden_features) == 0:
            self.emb = nn.Linear(embedding_features, output_features)
        else:
            self.ln_0 = nn.Linear(embedding_features, hidden_features[0])
            for i in range(1, len(hidden_features)):
                setattr(self, "ln_{}".format(i), nn.Linear(hidden_features[i-1], hidden_features[i]))
            
            self.emb = nn.Linear(hidden_features[-1], output_features)


    def forward(self, x):
        if len(self.hidden_features) == 0:
            out = self.emb(x)

        else:
            out = self.ln_0(x)
            out = F.relu(out)

            for i in range(1, len(self.hidden_features)):
                out = getattr(self, "ln_{}".format(i))(out)
                out = F.relu(out)
            
            out = self.emb(out)
        return out
    

class Conv_Encoder(nn.Module):
    """
        Convolutional Encoder
    """
    def __init__(self, in_channels, hidden_channels, emb_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=hidden_channels,
                               kernel_size=3,
                               padding="same",
                               )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels,
                               out_channels=emb_channels,
                               kernel_size=3,
                               padding="same")
        self.pool2 = nn.MaxPool2d(2, 2)


    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.pool2(out)
        return out
    

class Conv_Decoder(nn.Module):
    """
        Convolutional Decoder
    """
    def __init__(self, emb_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels=emb_channels,
                                        out_channels=hidden_channels,
                                        kernel_size=2,
                                        stride=2)
        
        self.conv2 = nn.ConvTranspose2d(in_channels=hidden_channels,
                                        out_channels=out_channels,
                                        kernel_size=2,
                                        stride=2)
        
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        return out


class Linear_AutoEncoder(nn.Module):
    def __init__(self, in_features, enc_hidden_features, embedding_features, dec_hidden_features):
        super().__init__()
        self.encoder = Linear_Encoder(in_features, enc_hidden_features, embedding_features)
        self.decoder = Linear_Decoder(embedding_features, dec_hidden_features, in_features)


    def forward(self, x):
        emb = F.relu(self.encoder(x))
        out = F.sigmoid(self.decoder(emb))
        return out, emb


class Conv_AutoEncoder(nn.Module):
    def __init__(self, in_channels, enc_hidden_channels, emb_channels, dec_hidden_channels):
        super().__init__()
        self.encoder = Conv_Encoder(in_channels, enc_hidden_channels, emb_channels)
        self.decoder = Conv_Decoder(emb_channels, dec_hidden_channels, in_channels)

    
    def forward(self, x):
        emb = F.relu(self.encoder(x))
        out = F.sigmoid(self.decoder(emb))
        return out, emb