import os
import time
import random
import datetime
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from utils import imshow, Random_Crop, initialize_weight, yuv_to_rgb_tensor
from dataset import CT_Dataset_Train, CT_Dataset_Test, SAMPLE_Real_Dataset, SAMPLE_Synth_Dataset
from network import Generator, Discriminator
from loss import G_loss_color_transfer, D_loss
import config

import atexit

# Parameters #
parser = config.options()
config.define_task_default_params(parser)
args = parser.parse_args()

output_dir = args.output_dir
data_dir = args.data_dir
weight_root = args.weight_root
task = args.task

writer = SummaryWriter(weight_root + '/run')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if task == 'SAMPLE':
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds_source = SAMPLE_Synth_Dataset(transform = transform, args = args, aug = True, real = False, hist = False, train = True)
    train_ds_color_ref = SAMPLE_Real_Dataset(transform = transform, args = args, aug = True, real = True, hist = True, train = True)

    train_dl_source = DataLoader(train_ds_source, batch_size = args.batch_size, shuffle = True)
    train_dl_color_ref = DataLoader(train_ds_color_ref, batch_size = args.batch_size, shuffle = True)

elif task == 'color_transfer':
    transform = transforms.Compose([transforms.ToTensor(), Random_Crop(args = args)])
    train_ds_source = CT_Dataset_Train(transform = transform, args = args, hist = False)
    train_ds_color_ref = CT_Dataset_Train(transform = transform, args = args, hist = True)
    #test_ds_source = CT_Dataset_Test(transform = transform, args = args, hist = False)
    #test_ds_color_ref = CT_Dataset_Test(transform = transform, args = args, hist = True)
    #imshow(train_ds_source[10])

    train_dl_source = DataLoader(train_ds_source, batch_size = args.batch_size, shuffle = True)
    train_dl_color_ref = DataLoader(train_ds_color_ref, batch_size = args.batch_size, shuffle = True)
    #test_dl_source = DataLoader(test_ds_source, batch_size = args.batch_size, shuffle = True)
    #test_dl_color_ref = DataLoader(test_ds_color_ref, batch_size = args.batch_size, shuffle = True)

G = Generator(args).to(device)
D = Discriminator(args).to(device)

G.apply(initialize_weight)
D.apply(initialize_weight)

opt_G = optim.Adam(G.parameters(), lr = args.gen_lr, betas = [args.beta_1, args.beta_2])
opt_D = optim.Adam(D.parameters(), lr = args.dis_lr, betas = [args.beta_1, args.beta_2])

sche_G = optim.lr_scheduler.StepLR(opt_G, step_size = 50, gamma = 0.5)
sche_D = optim.lr_scheduler.StepLR(opt_D, step_size = 50, gamma = 0.5)

G.train()
D.train()

start_time = time.time()
print('Start Train')
iter_count = 0

for epoch in range(args.epochs):
    for source_img, (target_img, target_hist) in zip(train_dl_source, train_dl_color_ref):
        # Target = Color Reference
        iter_count += 1
        source_img = source_img.to(device)
        target_img = target_img.to(device)

        # Discrminator #
        D.zero_grad()

        disc_real_output = D(source_img)
        generated_img = G(source_img, target_hist)
        disc_generated_output = D(generated_img.detach())
        d_loss = D_loss(disc_real_output, disc_generated_output)
        d_loss.backward()
        opt_D.step()

        # Generator #
        G.zero_grad()

        #generated_img = G(source_img, target_hist)
        disc_generated_output = D(generated_img)
        g_loss, g_gen_loss, g_mi_loss, g_hist_loss = G_loss_color_transfer(disc_generated_output, generated_img, source_img, target_img, args)
        g_loss.backward()
        opt_G.step()

        writer.add_scalar("Total Generater Loss", g_loss, iter_count)
        writer.add_scalar("Generator Loss", g_gen_loss, iter_count)
        writer.add_scalar("Mutual Information", g_mi_loss, iter_count)
        writer.add_scalar("Earth Mover's Distance", g_hist_loss, iter_count)
        writer.add_scalar("Total Discriminator Loss", d_loss, iter_count)

        if not (iter_count % 100):
            if task == 'color_transfer':
                plot_generated_img = make_grid(yuv_to_rgb_tensor(generated_img), nrow = 4, padding = 20, pad_value = 0.5)
                plot_source_img = make_grid(yuv_to_rgb_tensor(source_img), nrow = 4, padding = 20, pad_value = 0.5)
                plot_target_img = make_grid(yuv_to_rgb_tensor(target_img), nrow = 4, padding = 20, pad_value = 0.5)

                # plot_generated_Y_img = make_grid(generated_img[:,0].view(generated_img.shape[0], 1, 256, 256), nrow = 4, padding = 20, pad_value = 0.5)
                # plot_generated_U_img = make_grid(generated_img[:,1].view(generated_img.shape[0], 1, 256, 256), nrow = 4, padding = 20, pad_value = 0.5)
                # plot_generated_V_img = make_grid(generated_img[:,2].view(generated_img.shape[0], 1, 256, 256), nrow = 4, padding = 20, pad_value = 0.5)

                writer.add_image("Generated Image", plot_generated_img, iter_count)
                writer.add_image("Source (Content) Image", plot_source_img, iter_count)
                writer.add_image("Target (Color Reference) Image", plot_target_img, iter_count)

                # writer.add_image("Generated Y Image", plot_generated_Y_img, iter_count)
                # writer.add_image("Generated U Image", plot_generated_U_img, iter_count)
                # writer.add_image("Generated V Image", plot_generated_V_img, iter_count)
            elif task == 'SAMPLE':
                plot_generated_img = make_grid((generated_img + 1) / 2, nrow = 4, padding = 20, pad_value = 0.5)
                plot_source_img = make_grid((source_img + 1) / 2, nrow = 4, padding = 20, pad_value = 0.5)
                plot_target_img = make_grid((target_img + 1) / 2, nrow = 4, padding = 20, pad_value = 0.5)

                writer.add_image("Generated Image", plot_generated_img, iter_count)
                writer.add_image("Source (Content) Image", plot_source_img, iter_count)
                writer.add_image("Target (Color Reference) Image", plot_target_img, iter_count)

            sche_G.step()
            #sche_D.step()
            print('Epoch: {}, Total Generator Loss: {:.2f}, Total Discirminator Loss: {:.2f}, Running Time: {:.2f}min'.format(epoch, g_loss, d_loss, (time.time() - start_time) / 60))
    

writer.close()
weight_root = os.path.join(weight_root, 'HueNet_231126_weight_3.pt')
torch.save(G.state_dict(), weight_root)