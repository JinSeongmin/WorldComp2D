import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils import load_proximity_dependent_encoder
from Utils import load_landmark_coordinate_prior
from Utils import load_localizer

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



class DWBlock(nn.Module):
    def __init__(self, c, groups):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.gn = nn.GroupNorm(num_groups=groups, num_channels=c)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.dw(x)
        y = self.pw(y)
        y = self.gn(y)
        y = self.act(y)
        return x + y



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



class Auxiliary_localizer(nn.Module) : 
    def __init__(self, aux_loc_channel) : 
        super(Auxiliary_localizer, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(aux_loc_channel, 24, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=6, num_channels=24),
            nn.SiLU())
        self.blocks = nn.Sequential(*[DWBlock(c=24, groups=6) for _ in range(4)])
        self.head = nn.Conv2d(24, 1, 1)
        param_init(self)
        
    def forward(self, o) : 
        o = self.stem(o)
        o = self.blocks(o)
        return self.head(o) 



def framework_making(task) : 
    if task == 'COFW' : 
        n_l, enc_channel, aux_loc_channel = 29, 2, 2
    elif task == '300W' : 
        n_l, enc_channel, aux_loc_channel = 68, 6, 4
    elif task == 'AFLW' : 
        n_l, enc_channel, aux_loc_channel = 19, 6, 4
    
    enc_1 = Proximity_dependent_encoder(enc_channel).to(device)
    enc_2 = Proximity_dependent_encoder(enc_channel).to(device)
    enc_3 = Proximity_dependent_encoder(enc_channel).to(device)
    enc_1, enc_2, enc_3 = load_proximity_dependent_encoder(task, enc_1, enc_2, enc_3)
    encoders = [enc_1, enc_2, enc_3]    
    
    loc_1 = Localizer(n_l).to(device)
    loc_2 = Localizer(n_l).to(device)
    loc_3 = Localizer(n_l).to(device)
    loc_1, loc_2, loc_3 = load_localizer(task, loc_1, loc_2, loc_3)
    localizers = [loc_1, loc_2, loc_3]
    
    aux_loc_1 = Auxiliary_localizer(aux_loc_channel).to(device)
    aux_loc_2 = Auxiliary_localizer(aux_loc_channel).to(device)
    aux_loc_3 = Auxiliary_localizer(aux_loc_channel).to(device)
    auxiliary_localizers = [aux_loc_1, aux_loc_2, aux_loc_3]
    
    landmark_coordinate_prior = load_landmark_coordinate_prior(task)
    
    return encoders, localizers, auxiliary_localizers, landmark_coordinate_prior, n_l


