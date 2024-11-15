import argparse

def options():
    parser = argparse.ArgumentParser()

    # basic parameter
    parser.add_argument('--data_dir', default = './', help = 'path to data directory')
    parser.add_argument('--output_dir', default = './', help = 'path to translated data directory')
    parser.add_argument('--weight_root', default = './', help = 'path to saved weight')
    parser.add_argument('--task', default = '', choices = ['color_transfer', 'SAMPLE'])

    # model parameters
    parser.add_argument('--img_h', type=int, default=128, help='crop image to this image height')
    parser.add_argument('--img_w', type=int, default=128, help='crop image to this image width')
    parser.add_argument('--img_c', type=int, default=1, help='# of input image channels')
    parser.add_argument('--img_out_c', type=int, default=1, help='# of output image channels')
    parser.add_argument('--feature_dim', type=int, default = 32, help='# of feature dimension')

    # histogram layers parameters
    parser.add_argument('--bin_num', type = int, default = 256, help = 'histogram layers - number of bins')
    parser.add_argument('--kernel_width_ratio', type = float, default = 2.5, help = 'histogram layers - scale kernel width')

    # optimizer parameter
    parser.add_argument('--gen_lr', type = float, default = 2e-4)
    parser.add_argument('--dis_lr', type = float, default = 1e-4)
    parser.add_argument('--beta_1', type = float, default = 0.5)
    parser.add_argument('--beta_2', type = float, default = 0.999)

    # data preprocessing parameter
    parser.add_argument('--min_val', type = float, default = -1, help = "normalize image values to this min")
    parser.add_argument('--max_val', type = float, default = 1, help = "normalize image values to this max")
    parser.add_argument('--yuv', type = bool, default = True, help = "convert images to YUV colorspace")
    parser.add_argument('--a2b', type = bool, help = 'image translation direction AtoB or BtoA')
    parser.add_argument('--random_crop', type = bool, help = 'random crop for data augmentation')
    parser.add_argument('--ep', type = float, default = 1e-3, help = 'log transform parameter in SAMPLE dataset')

    # training params
    parser.add_argument('--batch_size', type = int, help = 'input batch size')
    parser.add_argument('--epochs', type = int, help = 'number of training epochs')

    # loss params
    parser.add_argument('--gan_loss_weight', type = float)
    parser.add_argument('--mi_loss_weight', type = float)
    parser.add_argument('--hist_loss_weight', type = float)
    
    return parser

def define_task_default_params(parser):
    args = parser.parse_args()

    if args.task == 'color_transfer':
        parser.set_defaults(data_dir = './Data/102flowers/data', output_dir = './Data/102flowers/results', weight_root = './Weight/HueNet/102flowers', 
                            img_h = 256, img_w = 256, img_c = 3, img_out_c = 3, feature_dim = 64,
                            batch_size=4, epochs=100, a2b=False,
                            random_crop=True, gan_loss_weight=1, mi_loss_weight=1, hist_loss_weight=100)
        
    elif args.task == 'SAMPLE':
        parser.set_defaults(batch_size = 4, epochs = 300, a2b = False, 
                            random_crop = False, gan_loss_weight = 1, mi_loss_weight = 5, hist_loss_weight = 100)
    