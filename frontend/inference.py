from options.demo_options import DemoOptions
from models.pix2pix_model import Pix2PixModel
from util.util import tensor2im, tensor2label, blend_image
from data.base_dataset import single_inference_dataLoad, demo_inference_dataLoad
from PIL import Image
import numpy as np
import cv2


class GanInference:
    def __init__(self):
        self.opt = DemoOptions().parse()
        self.opt.data_dir = './datasets/frontend_upload'
        self.opt.checkpoints_dir = './checkpoints'
        self.model = Pix2PixModel(self.opt)
        self.model.opt.inpaint_mode = 'ref'
        self.model.eval()

    def inference(self, ref_name, target_name):
        print(ref_name, target_name)
        self.opt.inference_ref_name = ref_name
        self.opt.inference_tag_name = target_name
        self.opt.inference_orient_name = target_name
        # read data

        ref_mask_path = f'{self.opt.data_dir}/labels/{ref_name}.png'
        ref_img = Image.open(f'{self.opt.data_dir}/images/{ref_name}.jpg')

        mask = cv2.imread(f'{self.opt.data_dir}/labels/{target_name}.png', cv2.IMREAD_GRAYSCALE)
        orient_mask = mask.copy()
        orient = cv2.imread(f'{self.opt.data_dir}/orients/{target_name}_orient_dense.png', cv2.IMREAD_GRAYSCALE)
        target_img = Image.open(f'{self.opt.data_dir}/images/{target_name}.jpg')


        data = demo_inference_dataLoad(self.opt, ref_mask_path, mask, orient_mask, orient, ref_img, target_img)
        #data = single_inference_dataLoad(self.opt)
        # forward
        generated, new_orient_rgb = self.model(data, mode='demo_inference')
        img_path = data['path']
        print('process image... %s' % img_path)

        #fake_image = tensor2im(generated[0])
        th = self.opt.add_th
        generated = generated[:, :, int(th/2):int(th/2)+self.opt.crop_size, int(th/2):int(th/2)+self.opt.crop_size]

        result = generated.permute(0, 2, 3, 1)
        result = result.cpu().numpy()
        result = (result + 1) * 127.5
        result = np.asarray(result[0, :, :, :], dtype=np.uint8)

        fake_image = Image.fromarray(result)
        fake_image.save(f'{self.opt.data_dir}/results/{target_name}.jpg')
