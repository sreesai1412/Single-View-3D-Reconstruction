"""
Sample usage:

python demo.py --name bird_net --num_train_epoch 500 --img_path misc/demo_data/img1.jpg
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags, app
import numpy as np
import skimage.io as io
from PIL import Image as im
import plotly.express as px
import pandas as pd
from saveo import save_obj

import torch

from nnutils import test_utils
from nnutils import predictor as pred_util
from utils import image as img_util
import soft_renderer.functional as srf

flags.DEFINE_string('img_path', 'misc/demo_data/img1.jpg', 'Image to run')
flags.DEFINE_integer('img_size', 256, 'image size the network was trained on.')

opts = flags.FLAGS


def preprocess_image(img_path, img_size=256):
    img = io.imread(img_path) / 255.

    # Scale the max image size to be img_size
    scale_factor = float(img_size) / np.max(img.shape[:2])
    img, _ = img_util.resize_img(img, scale_factor)

    # Crop img_size x img_size from the center
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # img center in (x, y)
    center = center[::-1]
    bbox = np.hstack([center - img_size / 2., center + img_size / 2.])

    img = img_util.crop(img, bbox, bgval=1.)

    # Transpose the image to 3xHxW
    img = np.transpose(img, (2, 0, 1))

    return img


def visualize(img, outputs, renderer):
    vert = outputs['verts'][0]
    cam = outputs['cam_pred'][0]
    texture = outputs['texture'][0]
    faces = outputs['faces'][0]
    shape_pred = renderer(vert, cam)
    img_pred = renderer(vert, cam, texture=texture)
    #renderer.saveMesh(vert, texture)

    vertex_seg_map = torch.argmax(outputs['vertex_seg_map'][0], dim=1).unsqueeze(1).type(torch.FloatTensor)
    x = torch.cat([vert.cpu(), vertex_seg_map], dim=1)
    print(outputs['vertex_seg_map'][0].max(1)[0].sum())
    df = x.numpy()
    df = pd.DataFrame(df)

    fig = px.scatter_3d(df, x=0, y=1, z=2, color=3)
    fig.write_html("./file.html")
    print("file.html written")

    tex_seg = vertex_seg_map.repeat(1, 3)
    for i in range(642):
        if(tex_seg[i][0].item() == 0.0):
            tex_seg[i] = torch.tensor([0.0, 0.0, 1.])
        elif(tex_seg[i][0].item() == 1.0):
            tex_seg[i] = torch.tensor([0.0, 1.0, 0.0])
        elif(tex_seg[i][0].item() == 2.0):
            tex_seg[i] = torch.tensor([0.0, 1.0, 1.0])
        elif(tex_seg[i][0].item() == 3.0):
            tex_seg[i] = torch.tensor([1., 0., 0.])
        elif(tex_seg[i][0].item() == 4.0):
            tex_seg[i] = torch.tensor([1., 0., 1.])
        elif(tex_seg[i][0].item() == 5.0):
            tex_seg[i] = torch.tensor([1., 1., 0.])
        elif(tex_seg[i][0].item() == 6.0):
            tex_seg[i] = torch.tensor([1., 0.5, 0.5])
        elif(tex_seg[i][0].item() == 7.0):
            tex_seg[i] = torch.tensor([0.5, 1, 1])



    save_obj("demo_seg.obj", vert, outputs['faces'][0], tex_seg.contiguous(), texture_type='vertex')
    print("seg_obj file written")

    # Different viewpoints.
    vp1 = renderer.diff_vp(
        vert, cam, angle=30, axis=[0, 1, 0], texture=texture, extra_elev=True)
    vp2 = renderer.diff_vp(
        vert, cam, angle=60, axis=[0, 1, 0], texture=texture, extra_elev=True)
    vp3 = renderer.diff_vp(
        vert, cam, angle=90, axis=[0, 1, 0], texture=texture)

    img = np.transpose(img, (1, 2, 0))
    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(shape_pred)
    plt.title('pred mesh')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(img_pred)
    plt.title('pred mesh w/texture')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(vp1)
    plt.title('different viewpoints')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(vp2)
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(vp3)
    plt.axis('off')
    plt.draw()
    plt.show()
    print('saving file to demo_image.png')
    plt.savefig('demo_image.png')


def main(_):

    img = preprocess_image(opts.img_path, img_size=opts.img_size)

    batch = {'img': torch.Tensor(np.expand_dims(img, 0))}

    predictor = pred_util.MeshPredictor(opts)
    outputs = predictor.predict(batch)

    # This is resolution
    renderer = predictor.vis_rend
    renderer.set_light_dir([0, 1, -1], 0.4)

    visualize(img, outputs, predictor.vis_rend)


if __name__ == '__main__':
    opts.batch_size = 1
    app.run(main)
