from __future__ import print_function
from scipy.io import loadmat
from PIL import Image
from PIL import ImageFile

import torch
import math
import torch.nn.functional as F
import torchvision.transforms.functional as F_
import random
import numpy as np
import pandas as pd
import h5py
import cv2
import scipy.ndimage as ndimage
import os


ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def fliplr_joints(task, landmark_coordinates, width):
    if task == 'COFW' : 
        matched_parts = [[1, 2], [5, 7], [3, 4], [6, 8], [9, 10], [11, 12], 
                         [13, 15], [17, 18], [14, 16], [19, 20], [23, 24]]
    elif task == '300W' : 
        matched_parts = [[1, 17], [2, 16], [3, 15], [4, 14], [5, 13], [6, 12], [7, 11], [8, 10],
                         [18, 27], [19, 26], [20, 25], [21, 24], [22, 23],
                         [32, 36], [33, 35],
                         [37, 46], [38, 45], [39, 44], [40, 43], [41, 48], [42, 47],
                         [49, 55], [50, 54], [51, 53], [62, 64], [61, 65], [68, 66], [59, 57], [60, 56]]
    elif task == 'AFLW' : 
        matched_parts = [[1, 6], [2, 5], [3, 4], [7, 12], [8, 11], [9, 10], [13, 15], [16, 18]]
    
    landmark_coordinates[:, 0] = width - landmark_coordinates[:, 0]
    for pair in matched_parts:
        tmp = landmark_coordinates[pair[0] - 1, :].copy()
        landmark_coordinates[pair[0] - 1, :] = landmark_coordinates[pair[1] - 1, :]
        landmark_coordinates[pair[1] - 1, :] = tmp
    return landmark_coordinates



def get_transform(center, scale, output_size, rot=0):
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(output_size[1]) / h
    t[1, 1] = float(output_size[0]) / h
    t[0, 2] = output_size[1] * (-float(center[0]) / h + .5)
    t[1, 2] = output_size[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    
    if not rot == 0:
        rot = -rot
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        t_mat = np.eye(3)
        t_mat[0, 2] = -output_size[1]/2
        t_mat[1, 2] = -output_size[0]/2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t



def transform_pixel(pt, center, scale, output_size, invert=0, rot=0):
    t = get_transform(center, scale, output_size, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1



def crop(img, center, scale, output_size, rot=0):
    center_new = center.clone()
    scale_adjustment = 1.
    
    sf = scale * 200.0 / output_size[0]
    if sf > 2 : 
        ht, wd = img.shape[0], img.shape[1]
        new_size = int(np.math.floor(max(ht, wd) / sf))
        new_ht = int(np.math.floor(ht / sf))
        new_wd = int(np.math.floor(wd / sf))
        
        if new_size < 2:
            return torch.zeros(output_size[0], output_size[1], img.shape[2]) \
                        if len(img.shape) > 2 else torch.zeros(output_size[0], output_size[1])
        else:
            img = cv2.resize(img.astype(np.float32), (new_wd, new_ht))
            center_new[0] = center_new[0] * 1.0 / sf
            center_new[1] = center_new[1] * 1.0 / sf
            scale = scale / sf
    
    ul = np.array(transform_pixel([0, 0], center_new, scale, output_size, invert=1))
    br = np.array(transform_pixel(output_size, center_new, scale, output_size, invert=1))
    original_size = (br-ul)[0]
    
    if not rot == 0:
        pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
        ul -= pad
        br += pad
    
    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape, dtype=np.float32)
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]] / 255
    
    if not rot == 0:
        new_img = ndimage.rotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]
        scale_adjustment = new_img.shape[0] / original_size
    
    new_img = cv2.resize(new_img.astype(np.float32), output_size)
    return new_img, scale_adjustment



class COFW(torch.utils.data.Dataset):
    def __init__(self, data_path, is_train, random_scale, random_flip, random_rotation):
        self.is_train = is_train
        self.data_root = data_path
        self.scale_factor = 0.05
        self.rot_factor = 10
        self.random_scale = random_scale
        self.random_flip = random_flip
        self.random_rotation = random_rotation
        
        if is_train:
            self.f = h5py.File(data_path+"COFW/COFW_train.mat", 'r')
            self.images = self.f['IsTr']
            self.images = self.images[0]
            self.pts = self.f['phisTr']
        else:
            self.f = h5py.File(data_path+"COFW/COFW_test.mat", 'r')
            self.images = self.f['IsT']
            self.images = self.images[0]
            self.pts = self.f['phisT']
            
        self.mean = np.array([0.4637], dtype=np.float32)
        self.std = np.array([0.2591], dtype=np.float32)
        
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_ref = self.images[index]
        img = self.f[image_ref]
        img = np.array(img).T
        # COFW dataset : 29 landmarks. [[x0,y0], [x1,y1], ..., [x28,y28]] order.
        pts = np.transpose(self.pts)[index][0:58].reshape(2, -1).transpose()
        
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)
        
        xmin = np.min(pts[:, 0])
        xmax = np.max(pts[:, 0])
        ymin = np.min(pts[:, 1])
        ymax = np.max(pts[:, 1])
        center_w = (math.floor(xmin) + math.ceil(xmax)) / 2.0
        center_h = (math.floor(ymin) + math.ceil(ymax)) / 2.0
        center = torch.Tensor([center_w, center_h])   # x, y order
        scale = max(math.ceil(xmax) - math.floor(xmin), math.ceil(ymax) - math.floor(ymin)) / 200.0
        scale *= 1.25
        
        r = 0
        
        if self.random_scale:
            scale = scale * (random.uniform(1 - self.scale_factor, 1 + self.scale_factor))
        
        if self.random_rotation : 
            r = random.uniform(-self.rot_factor, self.rot_factor) if random.random() <= 0.6 else 0
        
        if self.random_flip: 
            if random.random() <= 0.5:
                img = np.fliplr(img)
                pts = fliplr_joints('COFW', pts, width=img.shape[1])
                center[0] = img.shape[1] - center[0]
        
        img, scale_factor = crop(img, center, scale, [256, 256], rot=r)

        tpts = pts.copy()
        for i in range(pts.shape[0]):
            if tpts[i, 1] > 0 :
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center, scale*scale_factor, [256,256], rot=r)
        
        img = img.reshape(256, 256, 1)
        img = (img - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        img = torch.Tensor(img)
        
        # for [y1, x1, y2, x2, ...] order
        tpts = np.fliplr(tpts).flatten()
        tpts = torch.LongTensor(tpts)
        
        # [x1, y1, x2, y2, ...] order
        pts = torch.FloatTensor(pts).flatten()
        
        return img, tpts, pts, center, scale



class IBUG_300W(torch.utils.data.Dataset):
    def __init__(self, data_path, is_train, random_scale, random_flip, random_rotation):
        self.is_train = is_train
        self.data_root = data_path
        self.scale_factor = 0.05
        self.rot_factor = 10
        self.random_scale = random_scale
        self.random_flip = random_flip
        self.random_rotation = random_rotation
        
        if is_train:
            self.df = pd.read_csv(data_path + "300W/300W_train_data.csv")
        else:
            self.df = pd.read_csv(data_path + "300W/300W_test_data_full.csv")
        
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = os.path.join(self.data_root, row['image_path'])
        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
        
        # 300W dataset : 68 landmarks. [[x0, y0], [x1, y1], ..., [x67, y67]] order.
        pts = np.array(row[1:]).reshape(-1, 2)
        
        xmin = np.min(pts[:, 0])
        xmax = np.max(pts[:, 0])
        ymin = np.min(pts[:, 1])
        ymax = np.max(pts[:, 1])
        center_w = (math.floor(xmin) + math.ceil(xmax)) / 2.0
        center_h = (math.floor(ymin) + math.ceil(ymax)) / 2.0
        center = torch.Tensor([center_w, center_h])   # x, y order
        
        scale = max(math.ceil(xmax) - math.floor(xmin), math.ceil(ymax) - math.floor(ymin)) / 200.0
        scale *= 1.25
        
        if self.random_scale:
            scale = scale * (random.uniform(1 - self.scale_factor, 1 + self.scale_factor))
        
        r = 0
        if self.random_rotation : 
            r = random.uniform(-self.rot_factor, self.rot_factor) if random.random() <= 0.6 else 0
        
        if self.random_flip: 
            if random.random() <= 0.5:
                img = np.fliplr(img)
                pts = fliplr_joints('300W', pts, width=img.shape[1])
                center[0] = img.shape[1] - center[0]
        
        img, scale_factor = crop(img, center, scale, [256, 256], rot=r)

        tpts = pts.copy()
        for i in range(pts.shape[0]):
            if tpts[i, 1] > 0 :
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center, scale*scale_factor, [256,256], rot=r)
        
        img = (img - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        img = torch.Tensor(img)
        
        # for [y1, x1, y2, x2, ...] order
        tpts = np.fliplr(tpts).flatten().astype(np.float32)
        tpts = torch.LongTensor(tpts)
        
        # [x1, y1, x2, y2, ...] order
        pts = torch.FloatTensor(pts.astype(np.float32)).flatten()
        
        return img, tpts, pts, center, scale



class AFLW(torch.utils.data.Dataset):
    def __init__(self, data_path, is_train, random_scale, random_flip, random_rotation):
        self.is_train = is_train
        self.data_root = data_path
        self.scale_factor = 0.05
        self.rot_factor = 10
        self.random_scale = random_scale
        self.random_flip = random_flip
        self.random_rotation = random_rotation
        
        if is_train:
            self.f = loadmat(data_path + "AFLWinfo_release.mat")
            self.images = self.f['nameList'][:20000]
            self.pts = self.f['data'][:20000]
            
        else:
            self.f = loadmat(data_path + "AFLWinfo_release.mat")
            self.images = self.f['nameList'][20000:]
            self.pts = self.f['data'][20000:]
        
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.images[index][0][0])
        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
        
        # AFLW dataset : 19 landmarks. [[x0, y0], [x1, y1], ..., [x18, y18]] order.
        pts = self.pts[index].reshape(2,-1).transpose()
        
        xmin = np.min(pts[:, 0])
        xmax = np.max(pts[:, 0])
        ymin = np.min(pts[:, 1])
        ymax = np.max(pts[:, 1])
        center_w = (math.floor(xmin) + math.ceil(xmax)) / 2.0
        center_h = (math.floor(ymin) + math.ceil(ymax)) / 2.0
        center = torch.Tensor([center_w, center_h])   # x, y order
        
        scale = max(math.ceil(xmax) - math.floor(xmin), math.ceil(ymax) - math.floor(ymin)) / 200.0
        scale *= 1.25
        
        if self.random_scale:
            scale = scale * (random.uniform(1 - self.scale_factor, 1 + self.scale_factor))
        
        r = 0
        if self.random_rotation : 
            r = random.uniform(-self.rot_factor, self.rot_factor) if random.random() <= 0.6 else 0
        
        if self.random_flip: 
            if random.random() <= 0.5:
                img = np.fliplr(img)
                pts = fliplr_joints('AFLW', pts, width=img.shape[1])
                center[0] = img.shape[1] - center[0]
        
        img, scale_factor = crop(img, center, scale, [256, 256], rot=r)

        tpts = pts.copy()
        for i in range(pts.shape[0]):
            if tpts[i, 1] > 0 :
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center, scale*scale_factor, [256,256], rot=r)
        
        img = (img - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        img = torch.Tensor(img)
        
        # for [y1, x1, y2, x2, ...] order
        tpts = np.fliplr(tpts).flatten().astype(np.float32)
        tpts = torch.LongTensor(tpts)
        
        # [x1, y1, x2, y2, ...] order
        pts = torch.FloatTensor(pts.astype(np.float32)).flatten()
        
        return img, tpts, pts, center, scale



def load_data(task, batch_size, random_scale, random_flip, random_rotation):
    path = '../../Dataset/'
    if task == 'COFW' : 
        train_set = COFW(path, True, random_scale, random_flip, random_rotation)
        test_set = COFW(path, False, random_scale=False, random_flip=False, random_rotation=False)
    
    elif task == '300W' : 
        train_set = IBUG_300W(path, True, random_scale, random_flip, random_rotation)
        test_set = IBUG_300W(path, False, random_scale=False, random_flip=False, random_rotation=False)
    
    elif task == 'AFLW' : 
        path = path + '/AFLW/AFLW/aflw/data/flickr/'
        train_set = AFLW(path, True, random_scale, random_flip, random_rotation)
        test_set = AFLW(path, False, random_scale=False, random_flip=False, random_rotation=False)
        
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    return train_loader, test_loader



def extract_shifted_patches(task, imgs, starting_point, shifts, patch_size):
    B, C, H, W = imgs.shape    # 1, 1, 256+p, 256+p
    r = patch_size // 2

    delta = torch.arange(-r, r + 1, device=imgs.device)

    idx_h_local = starting_point + delta[:, None].repeat(1, patch_size)
    idx_w_local = starting_point + delta[None, :].repeat(patch_size, 1)

    idx_h_local = idx_h_local.unsqueeze(0)
    idx_w_local = idx_w_local.unsqueeze(0)

    shift_h = shifts[:, 0].view(-1, 1, 1)
    shift_w = shifts[:, 1].view(-1, 1, 1)

    idx_h = (idx_h_local + shift_h) % H 
    idx_w = (idx_w_local + shift_w) % W 
    
    B_expand = shifts.shape[0] // B
    batch_idx = torch.arange(B, device=imgs.device).repeat_interleave(B_expand)
    batch_idx = batch_idx.view(-1, 1, 1).expand(-1, patch_size, patch_size)

    if task == 'COFW' : 
        imgs_ = imgs.squeeze(1) 
        patches = imgs_[batch_idx, idx_h, idx_w] 
        return patches.unsqueeze(1) 
    
    else : 
        channel_idx = torch.arange(C, device=imgs.device).view(1, C, 1, 1).expand(idx_h.size(0), C, patch_size, patch_size)
        idx_h = idx_h.unsqueeze(1).expand(-1, C, patch_size, patch_size)
        idx_w = idx_w.unsqueeze(1).expand(-1, C, patch_size, patch_size)
        batch_idx = batch_idx.unsqueeze(1).expand(-1, C, patch_size, patch_size)
        patches = imgs[batch_idx, channel_idx, idx_h, idx_w]
        return patches



def batch_augmented_crop(task, imgs, landmark_coords) :
    B = imgs.size(0)
    view_size = [27, 27]
    scale_ratio = [1, 4]
    view_size_2 = [view_size[0]*scale_ratio[1], view_size[1]*scale_ratio[1]]
    view2 = int((view_size_2[0]-1)/2)
    
    need_pad = view2 + 1
    channel = 1 if task == 'COFW' else 3
    pad_value = 3 if task == 'COFW' else 0
    
    imgs = F.pad(imgs, (need_pad, need_pad, need_pad, need_pad), mode='constant', value=pad_value)
    
    n_l = landmark_coords.size(1)
    landmark_select = torch.randint(0, n_l, (B,))
    anchor_coords = landmark_coords.view(B, n_l, 2)[torch.arange(B), landmark_select]
    random_coords = torch.randint(0, 256, size=(B, 2)).to(device)
    
    batch_i_1ch = extract_shifted_patches(task, imgs, need_pad, anchor_coords, 27)
    batch_i_2ch = F_.resize(extract_shifted_patches(task, imgs, need_pad, anchor_coords, 27*4+1), view_size)
    batch_i = torch.cat((batch_i_1ch, batch_i_2ch), dim=1).view(B, 2*channel, 27, 27)
    
    augmented_j_1ch = extract_shifted_patches(task, imgs, need_pad, random_coords, 27)
    augmented_j_2ch = F_.resize(extract_shifted_patches(task, imgs, need_pad, random_coords, 27*4+1), view_size)
    augmented_j = torch.cat((augmented_j_1ch, augmented_j_2ch), dim=1).view(B, 2*channel, 27, 27)
    
    i_l_distance = (anchor_coords.view(B,1,2).repeat(1,n_l,1) - landmark_coords.view(B,n_l,2)).to(device)
    i_l_relationship = ((torch.abs(i_l_distance) <= view2).sum(-1) == 2).float().to(device)
    j_l_distance = (random_coords.view(B,1,2).repeat(1,n_l,1) - landmark_coords.view(B,n_l,2)).to(device)
    j_l_relationship = ((torch.abs(j_l_distance) <= view2).sum(-1) == 2).float().to(device)
    return batch_i, augmented_j, landmark_select, i_l_distance, i_l_relationship, j_l_distance, j_l_relationship




def save_model(task, enc_1, enc_2, enc_3, loss_hist, optimizer, epoch):
    state = {
        'enc_1' : enc_1.state_dict(),
        'enc_2' : enc_2.state_dict(),
        'enc_3' : enc_3.state_dict(),
        'opt': optimizer.state_dict(),
        'loss_hist' : loss_hist,
        'epoch': epoch}
    path = "../../Checkpoint/"
    
    torch.save(state, path+'{}_Proximity_dependent_encoders_{}epoch.pth'.format(task, epoch))
    
    

def scheduler_step(opt, epoch, decay_interval, gamma) : 
    lr = []
    if epoch % decay_interval == 0 :
        for g in opt.param_groups:
            g['lr'] = g['lr']*gamma
            lr.append(g['lr'])
    
        print("=== learning rate decayed to {}.\n".format(lr))

