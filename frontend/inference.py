from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.util import tensor2im, tensor2label, blend_image
from data.base_dataset import single_inference_dataLoad
from PIL import Image
import numpy as np


class GanInference:
    def __init__(self):
        self.opt = TestOptions().parse()
        self.opt.which_epoch = 50
        self.opt.use_encoder = True
        self.opt.noise_background = True
        self.opt.use_ig = True
        self.opt.add_feat_zeros = True
        self.opt.data_dir = './datasets/frontend_upload'
        self.opt.checkpoints_dir = './checkpoints'
        self.opt.gpu_ids = '0'
        self.model = Pix2PixModel(self.opt)
        self.model.eval()

    def inference(self, ref_name, target_name):
        print(ref_name, target_name)
        self.opt.inference_ref_name = ref_name
        self.opt.inference_tag_name = target_name
        self.opt.inference_orient_name = target_name
        # read data
        data = single_inference_dataLoad(self.opt)
        # forward
        generated = self.model(data, mode='inference')
        img_path = data['path']
        print('process image... %s' % img_path)

        fake_image = tensor2im(generated[0])
        if self.opt.add_feat_zeros:
            th = self.opt.add_th
            H, W = self.opt.crop_size, self.opt.crop_size
            fake_image_tmp = fake_image[int(th/2):int(th/2)+H,int(th/2):int(th/2)+W,:]
            fake_image = fake_image_tmp

        fake_image = Image.fromarray(np.uint8(fake_image))
        fake_image.save(f'{self.opt.data_dir}/results/{target_name}.jpg')
