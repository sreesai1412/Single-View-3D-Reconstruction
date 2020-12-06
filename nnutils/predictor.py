"""
Takes an image, returns stuff.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import os
import os.path as osp
import numpy as np
import scipy.misc
import torch
import torchvision
from torch.autograd import Variable
import scipy.io as sio

from nnutils import mesh_net
from nnutils import geom_utils
from nnutils.softras import SoftNeuralRenderer
from utils import bird_vis
from nnutils import loss_utils

# These options are off by default, but used for some ablations reported.
flags.DEFINE_boolean('ignore_pred_delta_v', False, 'Use only mean shape for prediction')
flags.DEFINE_boolean('use_sfm_ms', False, 'Uses sfm mean shape for prediction')
flags.DEFINE_boolean('use_sfm_camera', False, 'Uses sfm mean camera')
flags.DEFINE_string('dataset', 'cub', 'cub or pascal or p3d')
flags.DEFINE_boolean('seg_train', True, 'Segmentation training')
flags.DEFINE_boolean('query_basis', True, 'Use query vectors as part basis')
flags.DEFINE_boolean('predict_flow', True, 'Texture is predicteed via texture flow. dfault:false')
flags.DEFINE_boolean('ucmr_prob', True, 'predict probability like U-CMR')
flags.DEFINE_integer('hidden_enc_dim', 512, 'Dimension of latent vector')
flags.DEFINE_integer('num_parts', 8, 'Number of parts')

class MeshPredictor(object):
    def __init__(self, opts):
        self.opts = opts

        self.symmetric = opts.symmetric

        img_size = (opts.img_size, opts.img_size)
        self.num_cams = opts.num_hypo_cams
        print('Setting up model..')
        self.model = mesh_net.MeshNet(img_size, opts, nz_feat=opts.nz_feat, num_parts=opts.num_parts, num_encoder_layers=6,
                                      num_decoder_layers=6, predict_flow=opts.predict_flow, hidden_enc_dim=opts.hidden_enc_dim)
        self.load_network(self.model, 'pred', self.opts.num_train_epoch)
        self.model.eval()
        self.model = self.model.cuda(device=self.opts.gpu_id)
        if opts.seg_train and not opts.query_basis:
            self.part_basis_generator = mesh_net.PartBasisGenerator(opts.hidden_enc_dim, opts.num_parts, normalize=True)
            self.load_network(self.part_basis_generator, 'part_basis', opts.num_train_epoch)
            self.part_basis_generator.eval()
            self.part_basis_generator = self.part_basis_generator.cuda(device=opts.gpu_id)

        self.renderer = SoftNeuralRenderer(opts.img_size)

        if opts.texture:
            self.tex_renderer = SoftNeuralRenderer(opts.img_size)
            # Only use ambient light for tex renderer
            self.tex_renderer.ambient_light_only()

        if opts.use_sfm_ms:
            anno_sfm_path = osp.join(opts.cub_cache_dir, 'sfm', 'anno_testval.mat')
            anno_sfm = sio.loadmat(
                anno_sfm_path, struct_as_record=False, squeeze_me=True)
            sfm_mean_shape = torch.Tensor(np.transpose(anno_sfm['S'])).cuda(
                device=opts.gpu_id)
            self.sfm_mean_shape = Variable(sfm_mean_shape, requires_grad=False)
            self.sfm_mean_shape = self.sfm_mean_shape.unsqueeze(0).repeat(
                opts.batch_size, 1, 1)
            sfm_face = torch.LongTensor(anno_sfm['conv_tri'] - 1).cuda(
                device=opts.gpu_id)
            self.sfm_face = Variable(sfm_face, requires_grad=False)
            faces = self.sfm_face.view(1, -1, 3)
        else:
            # For visualization
            faces = self.model.faces.view(1, -1, 3)
        self.faces = faces.repeat(opts.batch_size, 1, 1)
        self.vis_rend = bird_vis.VisRenderer(opts.img_size,
                                             faces.data.cpu().numpy())
        self.vis_rend.set_bgcolor([1., 1., 1.])

        # SoftRas for rendering vertex segmentation
        self.vertex_seg_renderer = SoftNeuralRenderer(opts.img_size, texture_type='vertex')

        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.texture_loss = loss_utils.PerceptualTextureLoss()

    def load_network(self, network, network_label, epoch_label):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        network_dir = os.path.join(self.opts.checkpoint_dir, self.opts.name)
        save_path = os.path.join(network_dir, save_filename)
        print('loading {}..'.format(save_path))
        network.load_state_dict(torch.load(save_path))

        return

    def set_input(self, batch):
        opts = self.opts

        # original image where texture is sampled from.
        img_tensor = batch['img'].clone().type(torch.FloatTensor)

        # input_img is the input to resnet
        input_img_tensor = batch['img'].type(torch.FloatTensor)
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])

        #In order to mimic input during train time. Later we'll discard the repeated part.
        #input_img_tensor = input_img_tensor.repeat(2, 1, 1, 1)
        #img_tensor = img_tensor.repeat(2, 1, 1, 1)

        self.input_imgs = Variable(
            input_img_tensor.cuda(device=opts.gpu_id), requires_grad=False)
        self.imgs = Variable(
            img_tensor.cuda(device=opts.gpu_id), requires_grad=False)
        mask_tensor = batch['mask'].type(torch.FloatTensor)
        self.masks = Variable(
            mask_tensor.cuda(device=opts.gpu_id), requires_grad=False)
        if opts.use_sfm_camera:
            cam_tensor = batch['sfm_pose'].type(torch.FloatTensor)
            self.sfm_cams = Variable(
                cam_tensor.cuda(device=opts.gpu_id), requires_grad=False)

    def predict(self, batch):
        """
        batch has B x C x H x W numpy
        """
        self.set_input(batch)
        self.forward()
        return self.collect_outputs()

    def predict_az_ele(self, batch):
        """
        batch has B x C x H x W numpy
        """
        self.set_input(batch)
        return self.fwd_az_ele()

    def get_cam(self, proj_cam):
        num_cams = self.num_cams

        proj_cam = proj_cam.permute(1, 0, 2).contiguous()  # Nc x B x 7
        proj_cam = proj_cam.view(-1, *proj_cam.shape[2:])  # Nc*B x 7

        all_imgs = self.imgs

        if self.opts.texture:
            if self.opts.predict_flow:
                self.texture_flow = self.textures
                # b x f x t*t x 3
                self.textures = geom_utils.sample_textures(self.texture_flow, all_imgs)
                # B x 2 x H x W
                uv_flows = self.model.texture_predictor.uvimage_pred
                # B x H x W x 2
                self.uv_flows = uv_flows.permute(0, 2, 3, 1)
                self.uv_images = torch.nn.functional.grid_sample(all_imgs,
                                                                 self.uv_flows, align_corners=True)
            else:
                B, F, T, T, C = self.textures.shape  # (B x F x T x T x 3)
                self.textures = self.textures.view(B, F, T * T, C)
        else:
            self.textures = None
        # Render mask.
        vs = torch.cat([self.pred_v] * num_cams)
        fs = torch.cat([self.faces[:1].repeat(self.pred_v.shape[0], 1, 1)] * num_cams)
        ts = torch.cat([self.textures] * num_cams)
        mask_pred = self.renderer(vs, fs, proj_cam)  # Nc*B x H x W
        texture_pred = self.tex_renderer(vs, fs, proj_cam, textures=ts)

        # Need to take ths loss first to decide which cam to use
        masks_gt = self.masks
        cam_choose_loss_per_instance, mask_loss, tex_loss = loss_utils.cam_select_crit(mask_pred, torch.cat([masks_gt] * num_cams),
                                                                                 texture_pred, torch.cat([all_imgs] * num_cams),
                                                                                 self.texture_loss)
        cam_choose_loss_per_instance = torch.stack(cam_choose_loss_per_instance.chunk(num_cams)).detach()  # Nc x B
        loss_based_cam_prob = torch.nn.functional.softmin(cam_choose_loss_per_instance, dim=0)
        cam_probs = loss_based_cam_prob.detach().data
        # for each instance, find the camera with least loss
        cam_ids = cam_probs.argmax(0)
        proj_cam = torch.stack(proj_cam.chunk(num_cams)) # Nc x B x 7
        proj_cam = proj_cam[cam_ids, torch.arange(proj_cam.shape[1])] # B x 7

        mask_pred = torch.stack(mask_pred.chunk(num_cams))
        self.mask_pred = mask_pred[cam_ids, torch.arange(mask_pred.shape[1])]

        texture_pred = torch.stack(texture_pred.chunk(num_cams))
        self.texture_pred = texture_pred[cam_ids, torch.arange(texture_pred.shape[1])]

        return proj_cam

    def forward(self):
        if self.opts.texture:
            pred_codes, self.textures = self.model.forward(self.input_imgs)
        else:
            pred_codes = self.model.forward(self.input_imgs)#, self.part_basis_generator)

        self.delta_v = pred_codes['shape_pred']
        #self.cam_pred = pred_codes['cam']
        if self.opts.seg_train:
            vertex_part_pred = pred_codes['vertex_part_pred']
        cam_ids = pred_codes['cam_probs'].argmax(1)
        self.cam_pred = pred_codes['cam_hypotheses'][torch.arange(cam_ids.shape[0]).long(), cam_ids]

        if self.opts.use_sfm_camera:
            self.cam_pred = self.sfm_cams
        else:
            self.cam_pred = self.cam_pred#torch.cat([scale, trans, quat], 1)

        if self.opts.only_mean_sym:
            del_v = self.delta_v
            if opts.seg_train:
                self.vertex_seg_map = vertex_part_pred
        else:
            del_v = self.model.symmetrize(self.delta_v)
            if self.opts.seg_train:
                self.vertex_seg_map = torch.cat([vertex_part_pred, vertex_part_pred[:, -self.model.num_sym:]], 1)

        # Deform mean shape:
        self.mean_shape = self.model.get_mean_shape()

        if self.opts.use_sfm_ms:
            self.pred_v = self.sfm_mean_shape
        elif self.opts.ignore_pred_delta_v:
            self.pred_v = self.mean_shape + del_v * 0
        else:
            self.pred_v = self.mean_shape + del_v
        # Compute keypoints.
        if self.opts.use_sfm_ms:
            self.kp_verts = self.pred_v
        else:
            self.vert2kp = torch.nn.functional.softmax(
                self.model.vert2kp, dim=1)
            self.kp_verts = torch.matmul(self.vert2kp, self.pred_v)

        if self.opts.ucmr_prob:
            self.cam_pred = self.get_cam(pred_codes['cam_hypotheses'])

        self.faces =  self.faces[:1].repeat(self.pred_v.shape[0], 1, 1)

        # Project keypoints
        self.kp_pred = self.renderer.project_points(self.kp_verts,
                                                    self.cam_pred)
        if self.opts.ucmr_prob:
            return

        self.mask_pred = self.renderer.forward(self.pred_v, self.faces,
                                               self.cam_pred)
        # Render texture.
        if self.opts.texture and not self.opts.use_sfm_ms:
            if self.opts.predict_flow:
                if self.textures.size(-1) == 2:
                    # Flow texture!
                    self.texture_flow = self.textures
                    self.textures = geom_utils.sample_textures(self.textures,
                                                               self.imgs)
                # B x 2 x H x W
                uv_flows = self.model.texture_predictor.uvimage_pred
                # B x H x W x 2
                self.uv_flows = uv_flows.permute(0, 2, 3, 1)
                self.uv_images = torch.nn.functional.grid_sample(self.imgs,
                                                                 self.uv_flows, align_corners=True)

            if self.textures.dim() == 5:  # B x F x T x T x 3
                tex_size = self.textures.size(2)
                self.textures = self.textures.unsqueeze(4).repeat(1, 1, 1, 1,
                                                              tex_size, 1)


            # Render texture:
            B, F =  self.textures.shape[:2]
            self.texture_pred = self.tex_renderer.forward(
                self.pred_v, self.faces, self.cam_pred, textures=self.textures.view(B, F, -1, 3))
        else:
            self.textures = None

        # Render segmentation
        if self.opts.seg_train:
            seg_img_list = []
            for k in range(self.opts.num_parts):
                texture_vert = self.vertex_seg_map[:, :, k:k + 1].repeat(1, 1, 3)  # (B x Nv x 1) --> (B x Nv x 3)
                temp_img = self.vertex_seg_renderer(self.pred_v, self.faces, self.cam_pred, texture_vert, True)
                temp_img = temp_img[:, 0:1, :, :]  # (B x 3 x H x W) --> (B x 1 x H x W)
                seg_img_list.append(temp_img)

            self.seg_img = torch.cat(seg_img_list, dim=1)  # B x 3 x H x W

    def fwd_az_ele(self):
        _, wide_img_feat = self.model.encoder.forward(self.input_imgs)
        B, C, H, W = wide_img_feat.shape
        pos = torch.cat([
            self.model.code_predictor.cam_predictor.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
             self.model.code_predictor.cam_predictor.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        trans_in1 = pos + 0.1 * wide_img_feat.flatten(2).permute(2, 0, 1)  # (HW x B x C)
        trans_in2 = self.model.code_predictor.cam_predictor.query_pos.weight.unsqueeze(1).repeat(1, B, 1)  # (Nq x B x C) (Nq == Np)

        h = self.model.code_predictor.cam_predictor.transformer(trans_in1, trans_in2).transpose(0, 1).contiguous().view(-1, C)
        feat = self.model.code_predictor.cam_predictor.camera_predictor.fc_layer(h)
        az, ele = self.model.code_predictor.cam_predictor.camera_predictor.quat_predictor(h, True) # B*Nc x 1, B*Nc x 1
        az = torch.stack(az.chunk(B)).squeeze(-1) # B x Nc
        ele = torch.stack(ele.chunk(B)).squeeze(-1) # B x Nc
        return az, ele


    def collect_outputs(self):
        outputs = {
            'kp_pred': self.kp_pred.data,
            'verts': self.pred_v.data,
            'kp_verts': self.kp_verts.data,
            'cam_pred': self.cam_pred.data,
            'mask_pred': self.mask_pred.data,
            'faces': self.faces,
        }
        if self.opts.texture and not self.opts.use_sfm_ms:
            outputs['texture'] = self.textures
            outputs['texture_pred'] = self.texture_pred.data
            if self.opts.predict_flow:
                outputs['uv_image'] = self.uv_images.data
                outputs['uv_flow'] = self.uv_flows.data
                outputs['tex_flow'] = self.texture_flow
        if self.opts.seg_train and False:
            outputs['vertex_seg_map'] = self.vertex_seg_map
            outputs['seg_img']= self.seg_img
        return outputs
