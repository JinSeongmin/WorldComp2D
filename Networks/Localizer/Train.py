from __future__ import print_function
import torch
import torch.nn.functional as F
import argparse

from Utils import load_data
from Utils import extract_observation
from Utils import save_model
from Utils import scheduler_step

from Networks import encoder_localizer_making

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float

parser = argparse.ArgumentParser(description='Localizer training')

parser.add_argument('--task', type=str, default='COFW', help='COFW, 300W, AFLW')
parser.add_argument('--random_scale', default=True, help='Whether to apply random flip')
parser.add_argument('--random_flip', default=True, help='Whether to apply random flip')
parser.add_argument('--random_rotation', default=False, help='Whether to apply random rotation')

parser.add_argument('--batch_size', type=int, default=50, help='batch size for training')
parser.add_argument('--num_epochs', type=int, default=1000, help='maximum number of epochs')
parser.add_argument('--learning_rate', type=float, default=5E-4, help='Model learning rate')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Learning rate decay rate')
parser.add_argument('--lr_decay_interval', type=int, default=500, help='Learning rate decay interval')

args = parser.parse_args()


def main():
    torch.cuda.empty_cache()
    train_loader, _ = load_data(args.task, args.batch_size, args.random_scale, args.random_flip, args.random_rotation)
    
    encoders, localizers, landmark_coordinate_prior = encoder_localizer_making(args.task)
    
    optimizer = torch.optim.Adam([
        {'params': localizers[0].parameters(), 'lr': args.learning_rate},
        {'params': localizers[1].parameters(), 'lr': args.learning_rate},
        {'params': localizers[2].parameters(), 'lr': args.learning_rate}])
    
    loss_hist = torch.zeros(args.num_epochs, 3).to(device)
    
    for epoch in range(args.num_epochs):
        if epoch != 0 :
            scheduler_step(optimizer, epoch, args.lr_decay_interval, args.lr_decay_rate)
    
        loss = train(encoders, localizers, landmark_coordinate_prior, train_loader, optimizer)
        loss_hist[epoch] = loss
        
        print("\nEpoch: {}/{}.. \n".format(epoch+1, args.num_epochs).ljust(14),
              "Loss: ", loss) 
        
        if epoch == 0 or epoch % 100 == 99 : 
            save_model(args.task, localizers[0], localizers[1], localizers[2], loss_hist, optimizer, epoch+1)
        
    

def train(encoders, localizers, landmark_coordinate_prior, train_loader, optimizer) : 
    for m in encoders : 
        m.eval()
    for m in localizers : 
        m.train()
    
    loss_total = [0, 0, 0]
    
    img_size = torch.FloatTensor([256, 256]).to(device)
    fixation_point = torch.LongTensor([[64,64], [64,128], [64,192], 
                                       [128,64], [128,128], [128,192],
                                       [192,64], [192,128], [192,192]]).to(device)
    fixation_point = fixation_point.view(1,9,2).repeat(args.batch_size, 1, 1)
    fixation_point_norm = (2 * fixation_point / (img_size-1)) - 1
    
    for i, (images, tpts, pts, center, scale) in enumerate(train_loader) :         
        images = images.to(device)
        with torch.no_grad() : 
            landmark_coords = tpts.view(args.batch_size, -1, 2).to(device)
            coords_label = (2 * landmark_coords / (img_size-1)) - 1
            delta_coords_label = coords_label - landmark_coordinate_prior
            
            fixation_point += torch.randint(-5, 6, (args.batch_size, 9, 2)).to(device)
            
            o = extract_observation(args.task, images, fixation_point)
            
            z_1 = encoders[0](o)
            z_2 = encoders[1](o)
            z_3 = encoders[2](o)
            
            z_x_1 = torch.cat((z_1, fixation_point_norm.view(-1,2)), dim=1)
            z_x_2 = torch.cat((z_2, fixation_point_norm.view(-1,2)), dim=1)
            z_x_3 = torch.cat((z_3, fixation_point_norm.view(-1,2)), dim=1)
            
        localizers[0].zero_grad()
        localizers[1].zero_grad()
        localizers[2].zero_grad()
        optimizer.zero_grad()
        
        delta_x_hat_1 = localizers[0](z_x_1.view(args.batch_size, 9*(z_1.size(1)+2)))
        loss_1 = F.mse_loss(delta_x_hat_1, delta_coords_label)
        
        delta_x_hat_2 = localizers[1](z_x_2.view(args.batch_size, 9*(z_2.size(1)+2)))
        loss_2 = F.mse_loss(delta_x_hat_2, delta_coords_label)
        
        delta_x_hat_3 = localizers[2](z_x_3.view(args.batch_size, 9*(z_3.size(1)+2)))
        loss_3 = F.mse_loss(delta_x_hat_3, delta_coords_label)
    
        loss_1.backward()
        loss_2.backward()
        loss_3.backward()
        optimizer.step()
        
        loss_total[0] += loss_1.item() / len(train_loader)
        loss_total[1] += loss_2.item() / len(train_loader)
        loss_total[2] += loss_3.item() / len(train_loader)
        
    loss_total = torch.FloatTensor(loss_total)
    
    return loss_total



            


if __name__=='__main__':
    main()
    