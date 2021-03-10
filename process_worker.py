from queue import Queue
from pathlib import Path
from threading import Thread
from io import StringIO
from skimage.io import imread, imsave
from skimage.transform import rescale, resize

import requests as req
import cv2
import numpy as np
import os

from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util.util import tensor2im, tensor2label, blend_image
from util import html
from data.base_dataset import single_inference_dataLoad
from PIL import Image
import torch
import math
import numpy as np
import torch.nn as nn


class ProcessWorker(Thread):
    def __init__(self, image_q: Queue, done_dict, path):
        super(ProcessWorker, self).__init__()
        self.image_q = image_q
        self.done_dict = done_dict
        self.path = path
        self.model = Pix2PixModel(opt)
        self.model.eval()
        self.texture_size = 512

    def run(self):
        while True:
            if not self.image_q.empty():
                image_id = self.image_q.get_nowait()
                print('Starting to process', image_id)
                path = self.path + '/' + image_id + '/result'
                res = self.process_image(image_path=path)
                if res:
                    print('Done!')
                else:
                    print('Face not found')
                self.done_dict[image_id] = res

    def process_image(self, image=None, image_path=None):
        if image_path is not None:
            for ext in ['.jpg', '.png', '.jpeg']:
                if os.path.isfile(image_path + ext):
                    image = imread(image_path + ext)
                    break
        else:
            raise Exception('image and image_path variables are None')

        h, w, c = image.shape
        if c > 3:
            image = image[:, :, :3]

        # the core: regress position map
        max_size = max(image.shape[0], image.shape[1])
        if max_size > 1000:
            image = rescale(image, 1000. / max_size)
            image = (image * 255).astype(np.uint8)

        # -----------------------
        # forward
        generated = self.model(data, mode='inference')
        img_path = data['path']
        print('process image... %s' % img_path)

        fake_image = tensor2im(generated[0])
        if opt.add_feat_zeros or opt.add_zeros:
            th = opt.add_th
            H, W = opt.crop_size, opt.crop_size
            fake_image_tmp = fake_image[int(th / 2):int(th / 2) + H, int(th / 2):int(th / 2) + W, :]
            fake_image = fake_image_tmp

        fake_image_np = fake_image.copy()
        fake_image = Image.fromarray(np.uint8(fake_image))
        fake_image.save('./inference_samples/inpaint_fake_image.jpg')
        return True


if __name__ == '__main__':
    pass
