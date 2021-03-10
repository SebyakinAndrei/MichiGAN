import os
import sys
from PIL import Image
from io import BytesIO
import importlib
from pathlib import Path


def import_parents(level=1):
    global __package__
    file = Path(__file__).resolve()
    parent, top = file.parent, file.parents[level]

    sys.path.append(str(top))
    try:
        sys.path.remove(str(parent))
    except ValueError:  # already removed
        pass

    __package__ = '.'.join(parent.parts[len(top.parts):])
    importlib.import_module(__package__)  # won't be needed after that

print('Importing model code...')
import_parents(2)

from ..SR.Compression.dataset import OneImageData
from ..SR.Compression.model import Autoencoder
from torch.utils.data import DataLoader
import torch
from torchvision.utils import save_image

print('Code imported')

model = None
dataset = None
dataloader = None


def load_model():
    print('Loading model...')
    global model, dataset, dataloader
    resume = 140
    model = Autoencoder().cuda()
    model.load_state_dict(torch.load('../SR/Compression/models/model_{}.pt'.format(resume)))

    dataset = OneImageData()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    print('Done.')


if model is None:
    load_model()


def process_image(path, name):
    dataset.set_image(path)
    for img in dataloader:
        img = img.to('cuda')
        img = model.forward(img, return_bottleneck=False)
        save_image(img, 'temp/'+name+'_ours.png', padding=0)
        del img
        torch.cuda.empty_cache()


def compress(img: Image, imsize=None, bpp=1.0):
    if imsize is None:
        #imsize = (img.size[0]*img.size[1]*3*8) // k
        imsize = bpp*img.size[0]*img.size[1]//8
        print('Target imsize:', imsize)
    ql, qr = 0, 100
    while qr - ql > 1:
        out = BytesIO()
        mid = (ql + qr)//2
        img.save(out, 'jpeg', optimize=True, quality=mid, subsampling=0)
        print(ql, qr, mid, out.getbuffer().nbytes)
        if out.getbuffer().nbytes >= imsize:
            qr = mid
        else:
            ql = mid
    return out, imsize

if __name__ == "__main__":
    img = Image.open('test1.png')
    img = Image.open('test1.png')
    res = compress(img, bpp=0.5)[0]
    print('Result size:', res.getbuffer().nbytes)

    with open('result.jpg', 'wb') as out_f:
        out_f.write(res.getbuffer())