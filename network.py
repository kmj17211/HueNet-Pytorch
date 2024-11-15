import torch
import torch.nn as nn
import torch.nn.functional as F

class Down_Sampling(nn.Module):
    def __init__(self, in_channel, out_channel, apply_batchnorm = True):
        super().__init__()

        self.conv_block = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size = 4, stride = 2, padding = 1, bias = False))
        
        if apply_batchnorm:
            self.conv_block.append(nn.BatchNorm2d(out_channel))

        self.conv_block.append(nn.LeakyReLU(inplace = True))
        
    def forward(self, x):
        return self.conv_block(x)
    
class Up_Sampling(nn.Module):
    def __init__(self, in_channel, out_channel, apply_dropout = False):
        super().__init__()

        self.conv_block = nn.Sequential(nn.Upsample(scale_factor = 2, mode = 'bilinear'),
                                        nn.ReflectionPad2d(1),
                                        nn.Conv2d(in_channel, out_channel, kernel_size = 3, bias = True),
                                        nn.BatchNorm2d(out_channel))
        
        if apply_dropout:
            self.conv_block.append(nn.Dropout2d(0.5))
        
        self.conv_block.append(nn.ReLU(inplace = True))

    def forward(self, x):
        return self.conv_block(x)
    
class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.task = args.task
        self.img_h = args.img_h
        self.img_w = args.img_w
        self.img_c = args.img_c
        self.img_out_c = args.img_out_c
        self.feature_dim = args.feature_dim
        self.bin_num = args.bin_num

        self.down_layers = nn.ModuleList([                                                                      # SAMPLE        # Color Transfer
                            Down_Sampling(self.img_c, self.feature_dim, apply_batchnorm = False),               # 64 x 64 x 16  # 128 x 128 x 64
                            Down_Sampling(self.feature_dim, self.feature_dim * 2),                              # 32 x 32 x 32  # 64 x 64 x 128
                            Down_Sampling(self.feature_dim * 2, self.feature_dim * 4),                          # 16 x 16 x 64  # 32 x 32 x 256
                            Down_Sampling(self.feature_dim * 4, self.feature_dim * 8),                          # 8 x 8 x 128   # 16 x 16 x 512
                            Down_Sampling(self.feature_dim * 8, self.feature_dim * 8),                          # 4 x 4 x 128   # 8 x 8 x 512
                            Down_Sampling(self.feature_dim * 8, self.feature_dim * 8),                          # 2 x 2 x 128   # 4 x 4 x 512
                            Down_Sampling(self.feature_dim * 8, self.feature_dim * 8)                           # 1 x 1 x 128   # 2 x 2 x 512
        ])
        self.up_layers = nn.ModuleList([
                            Up_Sampling(self.feature_dim * 8 * 2, self.feature_dim * 8, apply_dropout = True),  # 2 x 2 x 128   # 4 x 4 x 512
                            Up_Sampling(self.feature_dim * 8 * 2, self.feature_dim * 8, apply_dropout = True),  # 4 x 4 x 128   # 8 x 8 x 512
                            Up_Sampling(self.feature_dim * 8 * 2, self.feature_dim * 8),                        # 8 x 8 x 128   # 16 x 16 x 512
                            Up_Sampling(self.feature_dim * 8 * 2, self.feature_dim * 4),                        # 16 x 16 x 64  # 32 x 32 x 256
                            Up_Sampling(self.feature_dim * 4 * 2, self.feature_dim * 2),                        # 32 x 32 x 32  # 64 x 64 x 128
                            Up_Sampling(self.feature_dim * 2 * 2, self.feature_dim),                            # 64 x 64 x 16  # 128 x 128 x 64
        ])
        self.last_layers = nn.Sequential(
                            nn.ConvTranspose2d(self.feature_dim * 2, self.img_out_c, kernel_size = 4, stride = 2, padding = 1), # 128 x 128 x 1  # 256 x 256 x 3
                            nn.Tanh()
        )

        if self.task == 'color_transfer':
            self.down_layers.append(Down_Sampling(self.feature_dim * 8, self.feature_dim * 8))
            self.up_layers.insert(0, Up_Sampling(self.feature_dim * 8 * 2, self.feature_dim * 8, apply_dropout = True))

            self.hist_1_em = nn.Embedding(self.img_h * self.img_w + 1, 1)
            self.hist_2_em = nn.Embedding(self.img_h * self.img_w + 1, 1)
            self.hist_3_em = nn.Embedding(self.img_h * self.img_w + 1, 1)

            self.hist_dense = nn.Linear(self.bin_num * 3, self.bin_num * 2)

        elif self.task == 'SAMPLE':
            self.hist_em = nn.Embedding(self.img_h * self.img_w + 1, 1)

            self.hist_dense = nn.Linear(self.bin_num, int(self.bin_num))


    def forward(self, input_img, input_hist):
        
        # Histogram Embedding
        if self.task == 'color_transfer':
            input_hist_1, input_hist_2, input_hist_3 = input_hist
            hist_1_em = self.hist_1_em(input_hist_1)
            hist_2_em = self.hist_2_em(input_hist_2)
            hist_3_em = self.hist_3_em(input_hist_3)
            hist_em = torch.cat([hist_1_em, hist_2_em, hist_3_em], dim = 1)
            hist_em = hist_em.view(hist_em.shape[0], -1)
        elif self.task == 'SAMPLE':
            hist_em = self.hist_em(input_hist)
        
        hist = F.relu(self.hist_dense(hist_em.view(hist_em.shape[0], -1)))

        # Downsampling
        skips = []
        for down in self.down_layers:
            input_img = down(input_img)
            skips.append(input_img)
        skips = reversed(skips[:-1])
        
        # Concatenation Histogram
        input_img = torch.cat([input_img, hist.unsqueeze(dim = 2).unsqueeze(dim = 3)], dim = 1)

        # Upsampling
        for up, skip in zip(self.up_layers, skips):
            input_img = up(input_img)
            input_img = torch.cat([input_img, skip], dim = 1)
        
        return self.last_layers(input_img)
    
class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.task = args.task
        self.img_c = args.img_c
        self.feature_dim = args.feature_dim

        self.down_layer = nn.Sequential(                                                                                       # SAMPLE        # Color Transfer
                            Down_Sampling(self.img_c, self.feature_dim, apply_batchnorm = False),                               # 16 x 64 x 64  # 64 x 128 x 128
                            Down_Sampling(self.feature_dim, self.feature_dim * 2),                                              # 32 x 32 x 32  # 128 x 64 x 64
                            Down_Sampling(self.feature_dim * 2, self.feature_dim * 4),                                          # 64 x 16 x 16  # 256 x 32 x 32
                            nn.Conv2d(self.feature_dim * 4, self.feature_dim * 8, kernel_size = 4, padding = 1, bias = False),  # 128 x 15 x 15 # 512 x 31 x 31
                            nn.BatchNorm2d(self.feature_dim * 8),
                            nn.LeakyReLU(inplace = True)
        )

        if self.task == 'color_transfer':
            self.down_layer.append(nn.Conv2d(self.feature_dim * 8, 1, kernel_size = 4, padding = 1, bias = False)) # 1 x 30 x 30
        
        elif self.task == 'SAMPLE':
            self.down_layer.append(nn.Conv2d(self.feature_dim * 8, 1, kernel_size = 3, padding = 1, bias = False)) # 1 x 15 x 15

    def forward(self, input_img):
        return self.down_layer(input_img)