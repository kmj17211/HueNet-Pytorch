import torch
from histogram_layer import HistogramLayersCT

# Non Saturation Loss
bc_loss = torch.nn.BCEWithLogitsLoss()
# Least Square Loss
mse_loss = torch.nn.MSELoss()

def G_loss_color_transfer(disc_generated_output, gen_output, src_img, tar_img, args):

    gan_loss_weight = args.gan_loss_weight
    mi_loss_weight = args.mi_loss_weight
    hist_loss_weight = args.hist_loss_weight
    task = args.task

    gan_loss = bc_loss(disc_generated_output, torch.ones_like(disc_generated_output))

    if task == 'color_transfer':
        # Histograms Instaces
        hist_1 = HistogramLayersCT(out_img = gen_output[:,0], tar_img = tar_img[:,0], src_img = src_img[:,0], args = args)
        hist_2 = HistogramLayersCT(out_img = gen_output[:,1], tar_img = tar_img[:,1], src_img = src_img[:,1], args = args)
        hist_3 = HistogramLayersCT(out_img = gen_output[:,2], tar_img = tar_img[:,2], src_img = src_img[:,2], args = args)

        # MI Loss
        mi_loss_1 = hist_1.calc_cond_entropy_loss_src_out()
        mi_loss_2 = hist_2.calc_cond_entropy_loss_src_out()
        mi_loss_3 = hist_3.calc_cond_entropy_loss_src_out()
        mi_loss = (mi_loss_1 + mi_loss_2 + mi_loss_3) / 3

        # His Loss
        hist_loss_1 = hist_1.calc_hist_loss_tar_out()
        hist_loss_2 = hist_2.calc_hist_loss_tar_out()
        hist_loss_3 = hist_3.calc_hist_loss_tar_out()
        hist_loss = (hist_loss_1 + hist_loss_2 + hist_loss_3) / 3

    elif task == 'SAMPLE':
        # Histograms Instances
        gen_output = gen_output.squeeze(1)
        src_img = src_img.squeeze(1)
        tar_img = tar_img.squeeze(1)

        hist = HistogramLayersCT(out_img = gen_output, tar_img = tar_img, src_img = src_img, args = args)

        # MI Loss
        mi_loss = hist.calc_cond_entropy_loss_src_out()

        # His Loss
        hist_loss = hist.calc_hist_loss_tar_out()

    total_gen_loss = gan_loss_weight * gan_loss + mi_loss_weight * mi_loss + hist_loss_weight * hist_loss

    return total_gen_loss, gan_loss, mi_loss, hist_loss

def D_loss(disc_real_output, disc_generated_output):
    
    real_loss = bc_loss(disc_real_output, torch.ones_like(disc_real_output))

    fake_loss = bc_loss(disc_generated_output, torch.zeros_like(disc_generated_output))

    total_disc_loss = real_loss + fake_loss

    return total_disc_loss