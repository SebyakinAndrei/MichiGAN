from bottle import *
from PIL import Image
from random import randrange
import json

import sys

#os.chdir('../')
print('Working dir:', os.getcwd())

from cal_orientation import process_image
from inference import GanInference
from face_parsing.tester import ParseNet


gan_inference = GanInference()
parsenet = ParseNet()


# https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
def pure_pil_alpha_to_color_v2(image, color=(255, 255, 255)):
    """Alpha composite an RGBA Image with a specified color.

    Simpler, faster version than the solutions above.

    Source: http://stackoverflow.com/a/9459208/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    image.load()  # needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background


@hook('after_request')
def enable_cors():
    """
    You need to add some headers to each request.
    Don't use the wildcard '*' for Access-Control-Allow-Origin in production.
    """
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'


@route('/upload', method='POST')
def do_upload():
    #for file in request.files:
    #    print(file)
    ref_img = request.files.get('ref-img')
    #ref_mask = request.files.get('ref-mask')
    target_img = request.files.get('target-img')
    #target_mask = request.files.get('target-mask')
    #print(ref_img, ref_mask, target_img, target_mask)

    base_path = './datasets/frontend_upload'

    ref_img_path = f'{base_path}/images/{ref_img.filename}'
    target_img_path = f'{base_path}/images/{target_img.filename}'

    ref_img.save(ref_img_path)
    target_img.save(target_img_path)

    #ref_mask_path = f'{base_path}/labels/{ref_mask.filename}'
    #target_mask_path = f'{base_path}/labels/{target_mask.filename}'

    #ref_mask.save(ref_mask_path)
    #target_mask.save(target_mask_path)


    orientation_root = f'{base_path}/orients'
    ref_name, target_name = ref_img.filename.split('.')[0], target_img.filename.split('.')[0]
    ref_img_img = Image.open(ref_img.file)
    target_img_img = Image.open(target_img.file)
    ref_mask_img, target_mask_img = parsenet.inference([ref_img_img, target_img_img])

    ref_mask_img.save(f'{base_path}/labels/{ref_name}.png')
    target_mask_img.save(f'{base_path}/labels/{target_name}.png')

    process_image(ref_img_img, ref_mask_img, orientation_root, ref_name)
    process_image(target_img_img, target_mask_img, orientation_root, target_name)

    gan_inference.inference(ref_name, target_name)

    #img = Image.open(upload.file)
    #compressed, imsize = None, 0
    #try:
    #    compressed, imsize = compress(img, bpp=0.2)
    #except:
    #    compressed, imsize = compress(pure_pil_alpha_to_color_v2(img), bpp=0.2)
    #with open('temp/'+randname+'_compr.jpg', 'wb') as out_f:
    #    out_f.write(compressed.getbuffer())
    #process_image('temp/'+randname+ext, randname)
    return json.dumps({'status': 'ok', 'result': f'{target_name}.jpg'})


@route('/static/<name:path>')
def get_img(name):
    return static_file(name, root='./datasets/frontend_upload')


@route('/results/<name>')
def get_img(name):
    return static_file(name, root='./datasets/frontend_upload/results')


@route('/web/<name:path>')
def get_img(name):
    return static_file(name, root='./frontend/web')


@route('/')
def index():
    return static_file('index.html', root='./frontend/web')

if __name__ == '__main__':
    run(host='0.0.0.0', port=4000, server='tornado')