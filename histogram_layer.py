import torch.nn as nn
import numpy as np
import torch

import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HistogramLayers(object):
    '''
    1D와 2D 히스토그램 계산
    EMD, MI 손실 함수 계산
    '''
    def __init__(self, out_img, tar_img, args):
        super().__init__()

        self.bin_num = args.bin_num
        self.min_val = args.min_val
        self.max_val = args.max_val
        self.interval_length = (self.max_val - self.min_val) / self.bin_num
        self.kernel_width = self.interval_length / args.kernel_width_ratio
        self.maps_out = self.calc_activation_maps(out_img)
        self.maps_tar = self.calc_activation_maps(tar_img)
        self.n_pixels = self.maps_out.size(1) # number of pixels in image (HxW)
        self.bs = self.maps_out.size(0) # batch size

    def calc_activation_maps(self, img):
        # bin_av : 히스토그램 bin의 평균값들

        bins_min_max = np.linspace(self.min_val, self.max_val, self.bin_num + 1)
        bins_av = (bins_min_max[0:-1] + bins_min_max[1:]) / 2
        bins_av = torch.tensor(bins_av, dtype = torch.float32).to(device)
        bins_av = bins_av.unsqueeze(0).unsqueeze(0) # shape = [1, 1, bin_num]

        # Flatten the image and add an extra dimension
        img_flat = img.view(img.size(0), -1).unsqueeze(-1) # shape = [batch_size, H*W, 1]

        maps = self.activation_func(img_flat, bins_av)

        return maps

    def activation_func(self, img_flat, bins_av):
        # 히스토그램 빈에 대한 이미지 픽셀 값의 활성화를 계산

        img_minus_bins_av = img_flat - bins_av # shape = [batch_size, H*W, 256]
        img_plus_bins_av = img_flat + bins_av # shape = [batch_size, H*W, 256]

        # 위에 2 줄은 중간 값에 대해 히스토그램 빈의 경계에 얼마나 가까운지 평가
        # 아래 4 줄은 -1과 1에 가까운 히스토그램 빈에 대해 평가
        maps = torch.sigmoid((img_minus_bins_av + self.interval_length / 2) / self.kernel_width)\
            - torch.sigmoid((img_minus_bins_av - self.interval_length / 2) / self.kernel_width)\
            + torch.sigmoid((img_plus_bins_av - 2 * self.min_val + self.interval_length / 2) / self.kernel_width)\
            - torch.sigmoid((img_plus_bins_av - 2 * self.min_val - self.interval_length / 2) / self.kernel_width)\
            + torch.sigmoid((img_plus_bins_av - 2 * self.max_val + self.interval_length / 2) / self.kernel_width)\
            - torch.sigmoid((img_plus_bins_av - 2 * self.max_val - self.interval_length / 2) / self.kernel_width)
        # plt.imshow(maps[0,:,128].reshape([256, 256]).detach().cpu())
        # plt.savefig('Test_activation.png')
        return maps
    
    def calc_cond_entropy_loss(self, maps_x, maps_y):
        pxy = torch.matmul(maps_x.transpose(1, 2), maps_y) / self.n_pixels # Joint PDF
        py = torch.sum(pxy, dim = 1) # Marginal PDF

        # calc conditional entropy
        '''
        H(X|Y) = - Sigma_x,Sigma_y{p(x,y)log(p(x,y)/p(y))}
                = - Sigma_x,Sigma_y{p(x,y)log(p(x,y)) - p(x,y)log(p(y))}
                = - Sigma_x,Sigma_y{p(x,y)log(p(x,y))} + Sigma_x,Sigma_y{p(x,y)log(p(y))}
                = - Sigma_x,Sigma_y{p(x,y)log(p(x,y))} + Sigma_y{p(y)log(p(y))}
                = - H(X,Y) + H(Y)
        '''
        hy = torch.sum(py * torch.log(py + 1e-9), dim = 1)
        hxy = torch.sum(pxy * torch.log(pxy + 1e-9), dim = [1, 2])
        cond_entropy = hy - hxy
        mean_cond_entropy = torch.mean(cond_entropy)
        # plt.imshow(pxy[0].detach().cpu())
        # plt.savefig('Test_pxy.png')
        
        return mean_cond_entropy
    
    def ecdf(self, maps):
        # calculate the CDF of p
        p = torch.sum(maps, dim = 1) / self.n_pixels # shape = [batch_size, bin_num]

        return torch.cumsum(p, dim = 1)
    
    def emd_loss(self, maps, maps_hat):
        ecdf_p = self.ecdf(maps) # shape = [batch_size, bin_num]
        ecdf_p_hat = self.ecdf(maps_hat) # shape = [batch_size, bin_num]
        emd = torch.mean(torch.pow(torch.abs(ecdf_p - ecdf_p_hat), 2), dim = -1) # shape = [batch_size, 1]
        emd = torch.pow(emd, 1/2)

        return torch.mean(emd)
    
    def calc_hist_loss_tar_out(self):
        return self.emd_loss(self.maps_tar, self.maps_out)
    
    def calc_cond_entropy_loss_tar_out(self):
        return self.calc_cond_entropy_loss(self.maps_tar, self.maps_out)
    

class HistogramLayersCT(HistogramLayers):
    '''
    Used for Color Transfer
    EMD(Target, Output), MI(Source, Output)
    '''

    def __init__(self, out_img, tar_img, src_img, args):
        super().__init__(out_img, tar_img, args)

        self.maps_src = self.calc_activation_maps(src_img)

    def calc_cond_entropy_loss_src_out(self):
        return self.calc_cond_entropy_loss(self.maps_src, self.maps_out)