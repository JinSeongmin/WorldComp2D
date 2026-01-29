from __future__ import print_function
import torch
import argparse

from Utils import load_data


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float

parser = argparse.ArgumentParser(description='Landmark coordinates prior')

parser.add_argument('--task', type=str, default='COFW', help='COFW, 300W, AFLW')

# Data loader
parser.add_argument('--random_scale', default=False, help='Whether to apply random flip')
parser.add_argument('--random_flip', default=False, help='Whether to apply random flip')
parser.add_argument('--random_rotation', default=False, help='Whether to apply random rotation')
parser.add_argument('--batch_size', type=int, default=1345, help='batch size for training')
args = parser.parse_args()


def main():
    torch.cuda.empty_cache()
    train_loader, _ = load_data(args.task, args.batch_size, args.random_scale, args.random_flip, args.random_rotation)
    
    with torch.no_grad(): 
        for i, (images, tpts, _, _, _) in enumerate(train_loader):
            images = images.to(device)
            landmark_coords = tpts.to(device).view(args.batch_size, -1, 2)
            img_size = torch.FloatTensor([256, 256]).to(device)
            landmark_coords_norm = (2 * landmark_coords / (img_size - 1)) - 1
            
            landmark_coords_prior = landmark_coords_norm.mean(0)
            
    path = '../../Pretrained_modules/'
    torch.save(landmark_coords_prior, path + "{}_Landmark_coordinate_priors.pth".format(args.task))
    

if __name__=='__main__':
    main()
    