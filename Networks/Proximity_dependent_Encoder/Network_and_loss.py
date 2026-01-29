import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float


def param_init(model) : 
    for m in model.modules():
        if isinstance(m, nn.Conv2d) :
            nn.init.normal_(m.weight.data, 0, 0.01)
        if isinstance(m, nn.ConvTranspose2d) :
            nn.init.normal_(m.weight.data, 0, 0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear) : 
            nn.init.normal_(m.weight.data, 0, 0.01)
            if m.bias != None : 
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)



class conv_layer_module(nn.Module) : 
    def __init__(self, in_ch, out_ch, k, s, p, bias=False) : 
        super(conv_layer_module, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s, padding=p, bias=bias)
        self.bat = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x) : 
        x = self.conv(x)
        x = self.bat(x)
        x = self.relu(x)
        return x



class fc_layer_module(nn.Module) : 
    def __init__(self, in_dim, out_dim, bias=False) : 
        super(fc_layer_module, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias)
        self.bat = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x) : 
        x = self.fc(x)
        x = self.bat(x)
        x = self.relu(x)
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




class PWConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(PWConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z, object_class, i_l_distance, i_l_relationship, j_l_distance, j_l_relationship):  
        B = z.size(0)
        
        anchor_feature = z[:, 0]
        contrast_feature = torch.cat(torch.unbind(z, dim=1), dim=0)
        
        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T) / self.temperature
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        i_labels = object_class.view(-1, 1).to(device)
        
        i_l_distance = torch.sqrt( (i_l_distance**2).sum(-1) )
        weight_i_l = 1 + torch.exp(-0.025*i_l_distance)
        weight_i_l = torch.where(weight_i_l >= 1.0, weight_i_l, torch.zeros_like(weight_i_l))
        weight_i_l = weight_i_l * i_l_relationship
        mask_aa = weight_i_l.T[i_labels.view(-1)]
        
        j_l_distance = torch.sqrt( (j_l_distance**2).sum(-1) )
        weight_j_l = 1 + torch.exp(-0.025*j_l_distance)
        weight_j_l = torch.where(weight_j_l >= 1.0, weight_j_l, torch.zeros_like(weight_j_l))
        weight_j_l = weight_j_l * j_l_relationship
        mask_ap = weight_j_l.T[i_labels.view(-1)]
        
        mask = torch.cat((mask_aa, mask_ap), 1)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(B).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1E-6)
        
        num_positive = (mask != 0).sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (num_positive + 1E-6)
        
        loss = -1 * mean_log_prob_pos.view(1, B).mean()
        return loss



def encoder_making(task) : 
    enc_channel = 2 if task == 'COFW' else 6
    
    enc_1 = Proximity_dependent_encoder(enc_channel).to(device)
    enc_2 = Proximity_dependent_encoder(enc_channel).to(device)
    enc_3 = Proximity_dependent_encoder(enc_channel).to(device)
    
    encoders = [enc_1, enc_2, enc_3]
    
    return encoders




