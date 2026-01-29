from __future__ import print_function
import torch
import argparse

from Utils import load_data
from Utils import batch_augmented_crop
from Utils import save_model
from Utils import scheduler_step

from Network_and_loss import encoder_making
from Network_and_loss import PWConLoss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float

parser = argparse.ArgumentParser(description='Proximity-dependent encoder training with PWConLoss')

parser.add_argument('--task', type=str, default="COFW", help='COFW, 300W, AFLW')
parser.add_argument('--random_scale', default=True, help='Whether to apply random flip')
parser.add_argument('--random_flip', default=True, help='Whether to apply random flip')
parser.add_argument('--random_rotation', default=True, help='Whether to apply random rotation')

parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=2000, help='Maximum number of epochs')
parser.add_argument('--learning_rate', type=float, default=1E-2, help='Model learning rate')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Learning rate decay rate')
parser.add_argument('--lr_decay_interval', type=int, default=1000, help='Learning rate decay interval')

args = parser.parse_args()

def main():
    torch.cuda.empty_cache()
    train_loader, _ = load_data(args.task, args.batch_size, args.random_scale, args.random_flip, args.random_rotation)
    
    encoders = encoder_making(args.task)
    
    criterion = PWConLoss().to(device)
    
    optimizer = torch.optim.Adam([
        {'params': encoders[0].parameters(), 'lr': args.learning_rate},
        {'params': encoders[1].parameters(), 'lr': args.learning_rate},
        {'params': encoders[2].parameters(), 'lr': args.learning_rate}])
    
    loss_hist = torch.zeros(args.num_epochs, 3).to(device)
    
    for epoch in range(args.num_epochs):
        if epoch != 0 :
            scheduler_step(optimizer, epoch, args.lr_decay_interval, args.lr_decay_rate)
        
        loss = train(encoders, train_loader, criterion, optimizer)
        loss_hist[epoch] = loss
        
        print("\nEpoch: {}/{}.. ".format(epoch+1, args.num_epochs).ljust(14),
              "PWConLoss: {}.. ".format(loss).ljust(14)) 
        
        if epoch == 0 or epoch % 200 == 199 : 
            save_model(args.task, encoders[0], encoders[1], encoders[2], loss_hist, optimizer, epoch+1)
            
            
        
        
def train(encoders, train_loader, criterion, optimizer):
    for m in encoders : 
        m.train()
        
    loss_total = [0, 0, 0]
    
    for i, (images, tpts, pts, center, scale) in enumerate(train_loader) :
        images = images.to(device)
        landmark_coords = tpts.to(device).view(args.batch_size, -1, 2)
        
        B = images.size(0)
        
        encoders[0].zero_grad()
        encoders[1].zero_grad()
        encoders[2].zero_grad()
        optimizer.zero_grad()
        
        batch_i, augmented_j, landmark_select, i_l_distance, \
            i_l_relationship, j_l_distance, j_l_relationship \
                = batch_augmented_crop(args.task, images, landmark_coords)
        batch_augmented = torch.cat((batch_i, augmented_j), 0)
        
        z = encoders[0](batch_augmented)
        z_i, z_j = torch.split(z, [B, B], dim=0)
        z = torch.cat((z_i.unsqueeze(1), z_j.unsqueeze(1)), dim=1)
        pwconloss_1 = criterion(z, landmark_select, i_l_distance, i_l_relationship, j_l_distance, j_l_relationship)
        
        z = encoders[1](batch_augmented)
        z_i, z_j = torch.split(z, [B, B], dim=0)
        z = torch.cat((z_i.unsqueeze(1), z_j.unsqueeze(1)), dim=1)
        pwconloss_2 = criterion(z, landmark_select, i_l_distance, i_l_relationship, j_l_distance, j_l_relationship)
        
        z = encoders[2](batch_augmented)
        z_i, z_j = torch.split(z, [B, B], dim=0)
        z = torch.cat((z_i.unsqueeze(1), z_j.unsqueeze(1)), dim=1)
        pwconloss_3 = criterion(z, landmark_select, i_l_distance, i_l_relationship, j_l_distance, j_l_relationship)
        
        pwconloss_1.backward()
        pwconloss_2.backward()
        pwconloss_3.backward()
        optimizer.step()
        
        loss_total[0] += pwconloss_1.item() / len(train_loader)
        loss_total[1] += pwconloss_2.item() / len(train_loader)
        loss_total[2] += pwconloss_3.item() / len(train_loader)
        
    loss_total = torch.FloatTensor(loss_total)
    
    return loss_total



if __name__=='__main__':
    main()
    
    
    
    
    
