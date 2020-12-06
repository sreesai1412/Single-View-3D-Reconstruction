from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.misc
import tqdm

import torch

import soft_renderer as sr

from nnutils import geom_utils
#from apex import amp

#############
### Utils ###
#############


def convert_as(src, trg):
    return src.to(trg.device).type_as(trg)


class SoftNeuralRenderer(torch.nn.Module):
    """
    This is the core pytorch function to call.
    Only fwd/bwd once per iteration.
    """

    def __init__(self, img_size=256, texture_type='surface'):
        super(SoftNeuralRenderer, self).__init__()
        self.renderer = sr.SoftRenderer(image_size=img_size, camera_mode='look_at',
                                        aggr_func_rgb='softmax', anti_aliasing=True,
                                        sigma_val=1e-5, gamma_val=1e-4, dist_eps=1e-10, perspective=False,
                                        light_intensity_ambient=0.8, eye=[0, 0, -2.732],
                                        texture_type=texture_type, light_mode=texture_type)

        self.proj_fn = geom_utils.orthographic_proj_withz
        self.offset_z = 5.
        self.part_seg = texture_type == 'vertex'

    def ambient_light_only(self):
        # Make light only ambient.
        self.renderer.lighting = sr.Lighting(intensity_ambient=1., intensity_directionals=0.)

    def set_bgcolor(self, color):
        self.renderer.rasterizer.background_color = color

    def set_light_dir(self, direction):
        self.renderer.lighting.directionals[0].light_direction = direction

    def project_points(self, verts, cams):
        proj = self.proj_fn(verts, cams)
        return proj[:, :, :2]

    def forward(self, vertices, faces, cams, textures=None, part_seg=False):
        verts = self.proj_fn(vertices, cams, offset_z=self.offset_z)
        vs = verts.clone()
        vs[:, :, 1] *= -1
        fs = faces.clone()
        if textures is None:
            self.mask_only = True
            masks = self.renderer(vs, fs)[:, -1]
            return masks
        else:
            self.mask_only = False
            ts = textures.clone().contiguous()
            if self.part_seg:
                # Vertex Part segmentation projection requires no lighting
                # and texture mode to be 'vertex'. Lighting could affect the softmax values.
                mesh = sr.Mesh(vs, fs, textures=ts, texture_type='vertex')
                self.renderer.set_texture_mode(mesh.texture_type)
                mesh = self.renderer.transform(mesh)
                return self.renderer.rasterizer(mesh, None)[:, :3]

            return self.renderer(vs, fs, ts)[:, :3]


############# TESTS #############
def exec_main():
    obj_file = 'birds3d/external/neural_renderer/examples/data/teapot.obj'
    vertices, faces = sr.load_obj(obj_file)

    renderer = sr.SoftRenderer()
    renderer.to_gpu(device=0)

    masks = renderer.forward_mask(vertices[None, :, :], faces[None, :, :])
    print(np.sum(masks))
    print(masks.shape)

    grad_masks = masks * 0 + 1
    vert_grad = renderer.backward_mask(grad_masks)
    print(np.sum(vert_grad))
    print(vert_grad.shape)

    # Torch API
    mask_renderer = sr.SoftRenderer()
    vertices_var = torch.autograd.Variable(torch.from_numpy(vertices[None, :, :]).cuda(device=0), requires_grad=True)
    faces_var = torch.autograd.Variable(torch.from_numpy(faces[None, :, :]).cuda(device=0))

    for ix in range(100):
        masks_torch = mask_renderer.forward(vertices_var, faces_var)
        vertices_var.grad = None
        masks_torch.backward(torch.from_numpy(grad_masks).cuda(device=0))

    print(torch.sum(masks_torch))
    print(masks_torch.shape)
    print(torch.sum(vertices_var.grad))

# @DeprecationWarning


def teapot_deform_test():
    #
    obj_file = 'birds3d/external/neural_renderer/examples/data/teapot.obj'
    img_file = 'birds3d/external/neural_renderer/examples/data/example2_ref.png'
    img_save_dir = 'birds3d/cachedir/softras/'

    vertices, faces = sr.load_obj(obj_file)
    from skimage.io import imread
    image_ref = imread(img_file).astype('float32').mean(-1) / 255.
    image_ref = torch.autograd.Variable(torch.Tensor(image_ref[None, :, :]).cuda(device=0))

    mask_renderer = sr.SoftRenderer()
    faces_var = torch.autograd.Variable(torch.from_numpy(faces[None, :, :]).cuda(device=0))
    cams = np.array([1., 0, 0, 1, 0, 0, 0], dtype=np.float32)
    cams_var = torch.autograd.Variable(torch.from_numpy(cams[None, :]).cuda(device=0))

    class TeapotModel(torch.nn.Module):
        def __init__(self):
            super(TeapotModel, self).__init__()
            vertices_var = torch.from_numpy(vertices[None, :, :]).cuda(device=0)
            self.vertices_var = torch.nn.Parameter(vertices_var)

        def forward(self):
            return mask_renderer.forward(self.vertices_var, faces_var, cams_var)

    opt_model = TeapotModel()

    optimizer = torch.optim.Adam(opt_model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    # from time import time
    loop = tqdm.tqdm(range(300))
    print('Optimizing Vertices: ')
    for ix in loop:
        # t0 = time()
        optimizer.zero_grad()
        masks_pred = opt_model()
        loss = torch.nn.MSELoss()(masks_pred, image_ref)
        loss.backward()
        if ix % 20 == 0:
            im_rendered = masks_pred.data[0, :, :]
            scipy.misc.imsave(img_save_dir + 'iter_{}.png'.format(ix), im_rendered)
        optimizer.step()
        # t1 = time()
        # print('one step %g sec' % (t1-t0))


if __name__ == '__main__':
    # exec_main()
    teapot_deform_test()
