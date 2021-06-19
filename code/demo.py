import os
import glob
import torch
import torch.nn
import torch.utils.data
import torchvision
from PIL import Image

from config_val import get_config
import numpy as np
import res2net2
import math


def trans_img(img_path, data_transform, device):
    ori_image = data_transform(Image.open(str(img_path))).unsqueeze(0)
    ori_image = ori_image.to(device)
    ori_image = ori_image[:,:3,:,:]
    return ori_image

def make_test_data(cfg, img_path_list, device):
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),

    ])
    imgs = []
    for img_path in img_path_list:

        depth_name = img_path.replace('png', 'npy')

        guide_image_name = img_path.replace('input', 'guide')
        guide_depth_name = depth_name.replace('input', 'guide')

        ori_image = trans_img(img_path, data_transform, device)
        guide_image = trans_img(guide_image_name, data_transform, device)

        depth =  np.load(depth_name, allow_pickle=True).item()['normalized_depth']
        ori_depth = torch.unsqueeze(torch.from_numpy(depth), 0)
        ori_depth = torch.unsqueeze(ori_depth, 0)

        guide_depth = np.load(guide_depth_name, allow_pickle=True).item()['normalized_depth']
        guide_depth = torch.unsqueeze(torch.from_numpy(guide_depth), 0)
        guide_depth = torch.unsqueeze(guide_depth, 0)

        imgs.append([ori_image, guide_image, ori_depth, guide_depth])

    return imgs

    

def main(cfg):
    # -------------------------------------------------------------------
    # basic config
    print(cfg)
    if cfg.gpu > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # -------------------------------------------------------------------

    path = cfg.input_data_path
    if cfg.gt:
        gt_path = cfg.ori_data_path
    output_dir = cfg.output_dir


    name = os.listdir(path)
    print('Start eval')
    networks_description = cfg.model_path
    network = res2net2.Dehaze3().cuda()
    network.load_state_dict(torch.load(networks_description)['state_dict'])

    networks = [network]
    
    
    if not os.path.exists(os.path.join(output_dir, cfg.name)):
        os.makedirs(os.path.join(output_dir, cfg.name))
    
    if cfg.gt:
        psrn, ssim = [], []

 

    for i in name:

        if i.endswith('.npy'):
            continue

        test_file_path = os.path.join(path, i)
        test_file_path = glob.glob(test_file_path)
        ori_image, guide_image, ori_depth, guide_depth = make_test_data(cfg, test_file_path, device)[0]
        ori_image, guide_image, ori_depth, guide_depth = ori_image.cuda(), guide_image.cuda(), ori_depth.cuda(), guide_depth.cuda()

        
        for idx, network in enumerate(networks):
            
            network.cuda().eval()
            
            if idx == 0:

                dehaze_image = network(ori_image, guide_image, ori_depth, guide_depth)
                dehaze_image = dehaze_image.cpu()
            else:
                tmp =  network(test_img.cuda(), test_depth.cuda()).detach()
                dehaze_image += tmp.cpu()
            
            network.cpu()
            

        dehaze_image = dehaze_image/(len(networks))
        dehaze_image = dehaze_image.cuda()

        torchvision.utils.save_image(dehaze_image, os.path.join(output_dir, cfg.name, i))
      
        del dehaze_image
    
    if cfg.gt:
        print(sum(psrn)/len(psrn), sum(ssim)/len(ssim))




if __name__ == '__main__':
    config_args, unparsed_args = get_config()
    main(config_args)
