from typing import List

import torch.nn as nn
from torchvision.utils import save_image
from torchvision import transforms

import cv2
import PIL
from .unet import unet
from .utils import *
from PIL import Image


def transformer(resize, totensor, normalize, centercrop, imsize):
    options = []
    if centercrop:
        options.append(transforms.CenterCrop(160))
    if resize:
        options.append(transforms.Resize((imsize,imsize), interpolation=PIL.Image.NEAREST))
    if totensor:
        options.append(transforms.ToTensor())
    if normalize:
        options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(options)
    
    return transform


class ParseNet:
    def __init__(self):
        # Model hyper-parameters
        self.imsize = 512

        self.G = unet().cuda()

        self.G.load_state_dict(torch.load('./face_parsing/models/parsenet/model.pth'))
        self.G.eval()
        self.transform = transformer(True, True, True, False, self.imsize)
        print(self.G)

    def inference(self, img_batch: List[PIL.Image.Image]):
        imgs = []
        for img in img_batch:
            img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)
        imgs = imgs.cuda()
        labels_predict = self.G(imgs)
        pred_batch = []
        for input in labels_predict:
            input = input.view(1, 19, self.imsize, self.imsize)
            pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
            pred_batch.append(PIL.Image.fromarray((pred == 13).astype(np.uint8), 'L'))
        #labels_predict_plain = generate_label_plain(labels_predict, self.imsize)
        #labels_predict_color = generate_label(labels_predict, self.imsize).cpu().numpy()
        return pred_batch
        #for k in range(self.batch_size):
        #    cv2.imwrite(os.path.join(self.test_label_path, str(i * self.batch_size + k) +'.png'), labels_predict_plain[k])
        #    save_image(labels_predict_color[k], os.path.join(self.test_color_label_path, str(i * self.batch_size + k) +'.png'))
