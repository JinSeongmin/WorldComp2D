import torch
import time
import argparse

from Utils import load_data
from Utils import extract_observation
from Utils import norm_coord_to_abs
from Utils import NME_calc

from Models import framework_making

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float

parser = argparse.ArgumentParser(description='Facial Landmark Localization')

parser.add_argument('--task', type=str, default='COFW', help='COFW, 300W, AFLW')
parser.add_argument('--random_scale', default=False, help='Whether to apply random flip')
parser.add_argument('--random_flip', default=False, help='Whether to apply random flip')
parser.add_argument('--random_rotation', default=False, help='Whether to apply random rotation')

parser.add_argument('--batch_size', type=int, default=39, help='batch size for test')
args = parser.parse_args()


def main():
    torch.cuda.empty_cache()
    _, test_loader = load_data(args.task, args.batch_size, args.random_scale, args.random_flip, args.random_rotation)
    
    encoders, localizers, aux_localizers, n_l, landmark_coordinate_prior = framework_making(args.task)
    
    # NME test
    NME = test(encoders, localizers, aux_localizers, n_l, landmark_coordinate_prior, test_loader)
    NME = torch.FloatTensor(NME).to(device)
    
    print("task = {}...".format(args.task))
    print("NME: {}, mean/std: {:.3f}±{:.3f}..".format(NME, NME.mean(), NME.std()).ljust(15))
    
    

def test(encoders, localizers, aux_localizers, n_l, landmark_coordinate_prior, test_loader) : 
    for m in (encoders + localizers + aux_localizers) : 
        m.eval()
    
    NME_mean_ls = []
    fixation_points = torch.LongTensor([[64,64], [64,128], [64,192], 
                                        [128,64], [128,128], [128,192],
                                        [192,64], [192,128], [192,192]]).to(device)
    fixation_points = fixation_points.view(1,9,2).repeat(args.batch_size,1,1)
    img_size = torch.FloatTensor([256, 256]).to(device)
    fixation_points_norm = (2 * fixation_points / (img_size-1)) - 1
    
    l_idx = 2 * torch.arange(0, n_l).to(device) / (n_l-1) - 1
    class_embedding = l_idx.view(-1, 1, 1, 1).expand(n_l, 1, 27, 27).repeat(args.batch_size, 1, 1, 1)
    
    NME_samples = [[] for _ in range(3)]
    
    delta_clamp = 2 if args.task == 'COFW' else 1
    
    with torch.no_grad() : 
        for i, (images, tpts, pts, center, scale) in enumerate(test_loader) :         
            images = images.to(device)
            
            o = extract_observation(args.task, images, fixation_points)
            
            # encoder-localizer
            z_1 = encoders[0](o)
            z_2 = encoders[1](o)
            z_3 = encoders[2](o)
            
            z_x_1 = torch.cat((z_1, fixation_points_norm.view(-1,2)), dim=1).view(args.batch_size, 9*(z_1.size(1)+2))
            x_hat_1 = norm_coord_to_abs(landmark_coordinate_prior + localizers[0](z_x_1)).long()
            
            z_x_2 = torch.cat((z_2, fixation_points_norm.view(-1,2)), dim=1).view(args.batch_size, 9*(z_2.size(1)+2))
            x_hat_2 = norm_coord_to_abs(landmark_coordinate_prior + localizers[1](z_x_2)).long()
            
            z_x_3 = torch.cat((z_3, fixation_points_norm.view(-1,2)), dim=1).view(args.batch_size, 9*(z_3.size(1)+2))
            x_hat_3 = norm_coord_to_abs(landmark_coordinate_prior + localizers[2](z_x_3)).long()
            
            
            # auxiliary_localizer
            o = extract_observation(args.task, images, x_hat_1, False)
            o_l = torch.cat((o, class_embedding), dim=1)
            h = aux_localizers[0](o_l)
            delta_y = torch.argmax(h.view(args.batch_size, n_l, -1), dim=-1) // 27 - 13
            delta_x = torch.argmax(h.view(args.batch_size, n_l, -1), dim=-1) % 27 - 13
            delta = torch.cat((delta_y.view(args.batch_size, n_l, 1), delta_x.view(args.batch_size, n_l, 1)), dim=-1)
            delta = torch.clamp(delta, -delta_clamp, delta_clamp)
            x_hat_1 = x_hat_1 + delta
            NME = NME_calc(args.task, x_hat_1.flip(dims=[-1]), pts, center, scale)
            NME_samples[0].extend(NME.tolist())
            
            o = extract_observation(args.task, images, x_hat_2, False)
            o_l = torch.cat((o, class_embedding), dim=1)
            h = aux_localizers[1](o_l)
            delta_y = torch.argmax(h.view(args.batch_size, n_l, -1), dim=-1) // 27 - 13
            delta_x = torch.argmax(h.view(args.batch_size, n_l, -1), dim=-1) % 27 - 13
            delta = torch.cat((delta_y.view(args.batch_size, n_l, 1), delta_x.view(args.batch_size, n_l, 1)), dim=-1)
            delta = torch.clamp(delta, -delta_clamp, delta_clamp)
            x_hat_2 = x_hat_2 + delta
            NME = NME_calc(args.task, x_hat_2.flip(dims=[-1]), pts, center, scale)
            NME_samples[1].extend(NME.tolist())
            
            o = extract_observation(args.task, images, x_hat_3, False)
            o_l = torch.cat((o, class_embedding), dim=1)
            h = aux_localizers[2](o_l)
            delta_y = torch.argmax(h.view(args.batch_size, n_l, -1), dim=-1) // 27 - 13
            delta_x = torch.argmax(h.view(args.batch_size, n_l, -1), dim=-1) % 27 - 13
            delta = torch.cat((delta_y.view(args.batch_size, n_l, 1), delta_x.view(args.batch_size, n_l, 1)), dim=-1)
            delta = torch.clamp(delta, -delta_clamp, delta_clamp)
            x_hat_3 = x_hat_3 + delta
            NME = NME_calc(args.task, x_hat_3.flip(dims=[-1]), pts, center, scale)
            NME_samples[2].extend(NME.tolist())


    for net_idx in range(3) : 
        NME_mean = 100*torch.FloatTensor(NME_samples[net_idx]).mean()
        NME_mean_ls.append(NME_mean)
                
    return NME_mean_ls







if __name__=='__main__':
    main()
    