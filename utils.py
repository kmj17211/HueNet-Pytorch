import cv2
import torch
import random
import matplotlib.pyplot as plt

from torchvision import transforms

class Random_Crop(object):
    def __init__(self, args, prob = 0.5):
        self.prob = prob
        self.img_h = args.img_h
        self.img_w = args.img_w
        self.img_c = args.img_c

    def __call__(self, img):

        random_crop = transforms.RandomCrop((self.img_h, self.img_w))

        if random.uniform(0, 1) > self.prob:
            resize = transforms.Resize(size = (round(self.img_h * 1.2), round(self.img_w * 1.2)))
            
            return random_crop(resize(img))
        else:
            return random_crop(img)

def initialize_weight(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv2d') != -1:
        torch.nn.init.normal_(model.weight.data, 0.0, 0.02)

def yuv_to_rgb(yuv_img):
    # yuv_img는 tensor 형식이어야 함
    # YUV에서 RGB로 변환하는 행렬
    y, u, v = yuv_img[0, :, :] * 255, yuv_img[1, :, :] * 255, yuv_img[2, :, :] * 255
    
    r = y + 1.402 * (v - 128)
    g = y - 0.344136 * (u - 128) - 0.714136 * (v - 128)
    b = y + 1.772 * (u - 128)

    rgb_img = torch.stack([r, g, b]).clamp(0, 255)

    return rgb_img / 255

def yuv_to_rgb_tensor(yuv_img):
    
    for i in range(yuv_img.shape[0]):
        yuv_img[i] = yuv_to_rgb((yuv_img[i] + 1) / 2)
    
    return yuv_img

def imshow(img):
    img = (img + 1) / 2
    plt.figure(figsize = [10, 5])
    plt.subplot(2, 2, 1)
    plt.imshow(img[0])
    plt.title('Y Channel')
    plt.subplot(2, 2, 2)
    plt.imshow(img[1])
    plt.title('U Channel')
    plt.subplot(2, 2, 3)
    plt.imshow(img[2])
    plt.title('V Channel')
    plt.subplot(2, 2, 4)
    plt.imshow(yuv_to_rgb(img).permute(1, 2, 0))
    plt.title('Image')
    plt.colorbar()
    plt.savefig('Test.png')