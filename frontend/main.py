from bottle import *
#from compression import compress, process_image
from PIL import Image
from random import randrange
import json


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
    upload = request.files.get('upload')
    name, ext = os.path.splitext(upload.filename)
    if ext not in ('.png', '.jpg', '.jpeg'):
        return 'File extension not allowed.'
    randname = '%030x' % randrange(16**30)
    upload.save('temp/'+randname+ext)

    img = Image.open(upload.file)
    compressed, imsize = None, 0
    #try:
    #    compressed, imsize = compress(img, bpp=0.2)
    #except:
    #    compressed, imsize = compress(pure_pil_alpha_to_color_v2(img), bpp=0.2)
    with open('temp/'+randname+'_compr.jpg', 'wb') as out_f:
        out_f.write(compressed.getbuffer())
    #process_image('temp/'+randname+ext, randname)
    return json.dumps({'filename': randname, 'size': '{:.1f}'.format(imsize/1024)})


@route('/static/<name>')
def get_img(name):
    return static_file(name, root='./temp')


@route('/web/<name:path>')
def get_img(name):
    return static_file(name, root='./web')


@route('/')
def index():
    return static_file('index.html', root='./web')

if __name__ == '__main__':
    run(host='0.0.0.0', port=4000, server='tornado')