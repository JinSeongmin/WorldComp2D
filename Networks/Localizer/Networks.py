import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils import load_proximity_dependent_encoder
from Utils import load_landmark_coordinate_prior


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float


def param_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0, 0.01)
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, 0, 0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0, 0.01)
            if m.bias != None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)



class conv_layer_module(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p, bias=False):
        super(conv_layer_module, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                              kernel_size=k, stride=s, padding=p, bias=bias)
        self.bat = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bat(x)
        x = self.act(x)
        return x



class fc_layer_module(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super(fc_layer_module, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias)
        self.bat = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.bat(x)
        x = self.act(x)
        return x



class Proximity_dependent_encoder(nn.Module) : 
    def __init__(self, enc_channel) : 
        super(Proximity_dependent_encoder, self).__init__()
        self.conv = nn.Sequential(
            conv_layer_module(enc_channel, 32, 3, 2, 1),
            conv_layer_module(32, 32, 3, 1, 1),
            conv_layer_module(32, 64, 3, 2, 1),
            conv_layer_module(64, 64, 3, 1, 1),
            conv_layer_module(64, 128, 3, 2, 1),
            conv_layer_module(128, 256, 3, 2, 1),
            nn.Flatten(1))
        self.proj = nn.Sequential(
            fc_layer_module(1024, 512),
            nn.Linear(512, 256))
        
        param_init(self)
        
    def forward(self, x) : 
        z = F.normalize(self.proj(self.conv(x)), dim=1)
        return z



class Localizer(nn.Module) : 
    def __init__(self, n_l) : 
        super(Localizer, self).__init__()
        self.n_l = n_l
        self.head = nn.Sequential(
            fc_layer_module(9*(256+2), 512),
            fc_layer_module(512, 512),
            nn.Linear(512, n_l*2))
        param_init(self)
        
    def forward(self, z_ft_coord) : 
        landmark_coords = self.head(z_ft_coord)
        landmark_coords = F.tanh(landmark_coords).view(-1, self.n_l, 2)
        return landmark_coords



def encoder_localizer_making(task) : 
    if task == 'COFW' : 
        n_l, enc_channel = 29, 2
    elif task == '300W' : 
        n_l, enc_channel = 68, 6
    elif task == 'AFLW' : 
        n_l, enc_channel = 19, 6
    
    enc_1 = Proximity_dependent_encoder(enc_channel).to(device)
    enc_2 = Proximity_dependent_encoder(enc_channel).to(device)
    enc_3 = Proximity_dependent_encoder(enc_channel).to(device)
    enc_1, enc_2, enc_3 = load_proximity_dependent_encoder(task, enc_1, enc_2, enc_3)
    encoders = [enc_1, enc_2, enc_3]
    
    loc_1 = Localizer(n_l).to(device)
    loc_2 = Localizer(n_l).to(device)
    loc_3 = Localizer(n_l).to(device)
    localizers = [loc_1, loc_2, loc_3]
    
    landmark_coordinate_prior = load_landmark_coordinate_prior(task)
    
    return encoders, localizers, landmark_coordinate_prior



