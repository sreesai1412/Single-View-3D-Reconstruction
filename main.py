"""
Script for the bird shape, pose and texture experiment.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags

import sys
import io
import os.path as osp
import time

import numpy as np
import torch
import torchvision
from torch.autograd import Variable
import scipy.io as sio
from collections import OrderedDict

from data import cub as cub_data
from utils import visutil
from utils import bird_vis
from utils import image as image_utils
from nnutils import train_utils
from nnutils import loss_utils
from nnutils import mesh_net
#from nnutils.nmr import NeuralRenderer
from nnutils.softras import SoftNeuralRenderer
from nnutils import geom_utils
#from tps.rand_tps import RandTPS
from nnutils.feature_extraction import FeatureExtraction, featureL2Norm
import soft_renderer as sr

from matplotlib import pyplot as plt
from PIL import Image

flags.DEFINE_string('dataset', 'cub', 'cub or pascal or p3d')
# Weights:
flags.DEFINE_float('kp_loss_wt', 30., 'keypoint loss weight')
flags.DEFINE_float('mask_loss_wt', 20., 'mask loss weight')
flags.DEFINE_float('cam_loss_wt', 2., 'weights to camera loss')
flags.DEFINE_float('deform_reg_wt', 20, 'reg to deformation')
flags.DEFINE_float('triangle_reg_wt', 0.2, 'weights to triangle smoothness prior')
flags.DEFINE_float('flatten_reg_wt', 0.0006, 'weights to  flatness prior')
flags.DEFINE_float('vert2kp_loss_wt', 0.16, 'reg to vertex assignment')
flags.DEFINE_float('vert2part_loss_wt', 10., 'reg to part assignment')
flags.DEFINE_float('tex_loss_wt', 5, 'weights to tex loss')
flags.DEFINE_float('tex_dt_loss_wt', .5, 'weights to tex dt loss')
flags.DEFINE_float('mask_dt_loss_wt', 5., 'weights to mask dt loss')
flags.DEFINE_boolean('use_gtpose', False, 'if true uses gt pose for projection, but camera still gets trained.')

flags.DEFINE_integer('num_parts', 8, 'number of parts')
flags.DEFINE_float('conc_loss_wt', 1, 'concentration loss weight')
flags.DEFINE_float('cam_ent_loss_wt', 1., 'cross entropy loss weight')
flags.DEFINE_float('cam_ce_loss_wt', 0., 'cross entropy loss weight')
flags.DEFINE_float('rot_reg', 1., 'rotation reg weight')
flags.DEFINE_float('invar_loss_wt', 100., 'invariance loss weight')
flags.DEFINE_float('chamf_loss_wt', 100., 'chamfer loss weight')
flags.DEFINE_float('cam_flip_loss_wt', 0., 'camera flip loss weight')
flags.DEFINE_float('tex_flip_loss_wt', 0., 'texture flow flip loss weight')
flags.DEFINE_float('deform_flip_loss_wt', 0., 'deformation flip loss weight')
flags.DEFINE_string('restore_part_basis', '', 'path to partbasis stat
flags.DEFINE_float('sc_loss_wt', 10., 'semantic consistency loss weight')
flags.DEFINE_boolean('seg_train', True, 'whether to train seg')
flags.DEFINE_boolean('query_basis', False, 'Use camera queries as part basis')

flags.DEFINE_float('warmup_pose_iter', 5000, 'Warm up iter for pose prediction')
flags.DEFINE_float('warmup_shape_iter', 0, 'Warm up iter for updation of mean shape')
flags.DEFINE_boolean('predict_flow', True, 'whether to predict texture flow')
flags.DEFINE_boolean('ucmr_prob', True, 'use softmin of loss to assign prob')
flags.DEFINE_integer('hidden_enc_dim', 512, 'number of hidden dim in encoder output')


opts = flags.FLAGS

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'misc', 'cachedir')


def hook(module, grad_input, grad_output):
    print(str(module), grad_input.shape, grad_input.device, grad_output.shape, grad_output.device)
    return grad_input


class ShapeTrainer(train_utils.Trainer):
    def define_model(self):
        opts = self.opts
        # ----------
        # Options
        # ----------
        self.symmetric = opts.symmetric
        anno_sfm_path = osp.join(opts.cub_cache_dir, 'sfm', 'anno_train.mat')
        anno_sfm = sio.loadmat(anno_sfm_path, struct_as_record=False, squeeze_me=True)
        sfm_mean_shape = (np.transpose(anno_sfm['S']), anno_sfm['conv_tri'] - 1)

        if opts.seg_train:
            # Initialize feature extractor and part basis for the semantic consistency loss.

            self.zoo_feat_net = FeatureExtraction(feature_extraction_cnn='vgg19', normalization=True,
                                                  last_layer='relu5_4')
            self.zoo_feat_net.eval()

        if not opts.query_basis and opts.seg_train:
            self.part_basis_generator = mesh_net.PartBasisGenerator(512,opts.num_parts, normalize=True)
            if opts.num_pretrain_epochs > 0:
                self.load_network(self.part_basis_generator, 'part_basis', opts.num_pretrain_epochs)

            self.part_basis_generator.cuda(device=opts.gpu_id)
            self.part_basis_generator.train()

            if opts.restore_part_basis != '':
                self.part_basis_generator.load_state_dict(
                    {'w': torch.load(opts.restore_part_basis)})

        img_size = (opts.img_size, opts.img_size)
        num_parts = opts.num_parts
        self.num_cams = opts.num_hypo_cams
        self.model = mesh_net.MeshNet(
            img_size, opts, nz_feat=opts.nz_feat, num_kps=opts.num_kps, sfm_mean_shape=None,#sfm_mean_shape,
            num_parts=num_parts, num_encoder_layers=6, num_decoder_layers=6, predict_flow=opts.predict_flow,
            hidden_enc_dim=opts.hidden_enc_dim)

        if opts.num_pretrain_epochs > 0:
            self.load_network(self.model, 'pred', opts.num_pretrain_epochs)

        self.model = self.model.cuda(device=opts.gpu_id)

        # Data structures to use for triangle priors.
        edges2verts = self.model.edges2verts
        # B x E x 4
        edges2verts = np.tile(np.expand_dims(edges2verts, 0), (opts.batch_size, 1, 1))
        self.edges2verts = Variable(torch.LongTensor(edges2verts).cuda(device=opts.gpu_id), requires_grad=False)
        # For renderering.
        faces = self.model.faces.view(1, -1, 3)
        self.faces = faces.repeat(2 * opts.batch_size, 1, 1)
        self.renderer = SoftNeuralRenderer(opts.img_size)
        self.renderer_predcam = SoftNeuralRenderer(opts.img_size)  # for camera loss via projection

        # Need separate NMR for each fwd/bwd call.
        if opts.texture:
            self.tex_renderer = SoftNeuralRenderer(opts.img_size)
            # Only use ambient light for tex renderer
            self.tex_renderer.ambient_light_only()

        # SoftRas for rendering vertex segmentation
        self.vertex_seg_renderer = SoftNeuralRenderer(opts.img_size, texture_type='vertex')

        # For visualization
        self.vis_rend = bird_vis.VisRenderer(opts.img_size, faces.data.cpu().numpy())

        self.interp = torch.nn.Upsample(
            size=(opts.img_size, opts.img_size), mode='bilinear', align_corners=True)

        # import ipdb
        # ipdb.set_trace()
        # for k,v in self.model.named_modules():
        #         v.register_backward_hook(hook)

        # use lists to store the outputs via up-values
        self.conv_features, self.enc_attn_weights, self.dec_attn_weights = [], [], []

        self.model.encoder.resnet_conv.resnet.layer4[-1].register_forward_hook(
            lambda module, input, output: \
            self.conv_features.append(output.detach().cpu()[0])
        ),
        self.model.code_predictor.cam_predictor.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda module, input, output:\
            self.enc_attn_weights.append(output[1].detach().cpu()[0])
        ),
        self.model.code_predictor.cam_predictor.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
            lambda module, input, output:\
            self.dec_attn_weights.append(output[1].detach().cpu()[0])
        )

        return

    def init_dataset(self):
        opts = self.opts
        if opts.dataset == 'cub':
            self.data_module = cub_data
        else:
            print('Unknown dataset %d!' % opts.dataset)

        self.dataloader = self.data_module.data_loader(opts)
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def define_criterion(self):
        self.projection_loss = loss_utils.kp_l2_loss
        self.mask_loss_fn = loss_utils.neg_iou_loss  # torch.nn.MSELoss()
        self.entropy_cam_loss = loss_utils.neg_entropy
        self.mask_dt_loss_fn = loss_utils.mask_dt_loss
        self.entropy_loss = loss_utils.entropy_loss
        self.deform_reg_fn = loss_utils.deform_l2reg
        self.camera_loss = loss_utils.camera_loss
        #self.triangle_loss_fn = loss_utils.LaplacianLoss(self.faces)
        self.triangle_loss_fn = sr.LaplacianLoss(self.model.get_mean_shape().data.cpu(), self.faces[0].data.cpu()).cuda(device=opts.gpu_id)
        self.flatten_loss_fn = sr.FlattenLoss(self.faces[0].data.cpu()).cuda(device=opts.gpu_id)

        # New stuff:
        self.invariance_loss = loss_utils.invariance_loss
        self.cam_choose_crit = loss_utils.cam_select_crit
        self.chamfer_loss = loss_utils.chamfer_loss
        self.orthonormal_loss = loss_utils.orthonormal_loss
        self.ce_loss = loss_utils.cam_ce_loss
        self.rot_loss = loss_utils.rotation_reg

        if self.opts.seg_train:
            self.semantic_consistency_loss = loss_utils.semantic_consistency_loss
            self.concentration_loss = loss_utils.concentration_loss

        if self.opts.texture:
            self.texture_loss = loss_utils.PerceptualTextureLoss()
            self.texture_dt_loss_fn = loss_utils.texture_dt_loss

    def set_input(self, batch):
        opts = self.opts

        # Image with annotations.
        input_img_tensor = batch['img'].type(torch.FloatTensor)
        input_img_tensor2 = input_img_tensor.clone()
        imgs2 = batch['img'].type(torch.FloatTensor)

        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])
            imgs2[b] = torch.flip(imgs2[b], [-1])  # self.transform_custom(imgs2[b])
            input_img_tensor2[b] = torch.flip(input_img_tensor[b], [-1])  # self.transform_custom(imgs2[b])
            #input_img_tensor2[b] = self.resnet_transform(imgs2[b])
            #input_img_tensor2[b] = self.resnet_transform(imgs2[b])

        img_tensor = batch['img'].type(torch.FloatTensor)
        mask_tensor = batch['mask'].type(torch.FloatTensor)
        kp_tensor = batch['kp'].type(torch.FloatTensor)
        cam_tensor = batch['sfm_pose'].type(torch.FloatTensor)

        self.input_imgs = Variable(
            input_img_tensor.cuda(device=opts.gpu_id), requires_grad=False)
        self.input_imgs_2 = Variable(
            input_img_tensor2.cuda(device=opts.gpu_id), requires_grad=False)
        self.imgs = Variable(
            img_tensor.cuda(device=opts.gpu_id), requires_grad=False)
        self.imgs2 = Variable(
            imgs2.cuda(device=opts.gpu_id), requires_grad=False)

        self.masks = Variable(
            mask_tensor.cuda(device=opts.gpu_id), requires_grad=False)
        self.masks_2 = torch.flip(self.masks, [-1])
        self.kps = Variable(
            kp_tensor.cuda(device=opts.gpu_id), requires_grad=False)
        self.cams = Variable(
            cam_tensor.cuda(device=opts.gpu_id), requires_grad=False)

        # Compute barrier distance transform.
        mask_dts = np.stack([image_utils.compute_dt_barrier(m) for m in batch['mask']])
        mask_dts_2 = np.stack([image_utils.compute_dt_barrier(torch.flip(m, [-1])) for m in batch['mask']])
        dt_tensor = torch.FloatTensor(mask_dts).cuda(device=opts.gpu_id)
        dt_tensor_2 = torch.FloatTensor(mask_dts_2).cuda(device=opts.gpu_id)
        # B x 1 x N x N
        self.dts_barrier = Variable(dt_tensor, requires_grad=False).unsqueeze(1)
        self.dts_barrier_2 = Variable(dt_tensor_2, requires_grad=False).unsqueeze(1)

    def reflect_cam_pose(self, cam_pose):
        new_cam_pose = cam_pose * torch.FloatTensor([1, -1, 1, 1, 1, -1, -1]).view(1, 1, -1).cuda()
        return new_cam_pose

    def flip_train_predictions_swap(self, cams_hypo, true_size):
        # Copy cam
        # Copy Cam Probs
        keys_to_copy = ['cam_probs', 'cam_sample_inds']
        for key in keys_to_copy:
            codes_pred[key]= torch.cat([codes_pred[key][:true_size], codes_pred[key][:true_size]])

        ## mirror rotation
        new_cam_pose = self.reflect_cam_pose(codes_pred['cam'][:true_size, None,:]).squeeze(1)
        if not (codes_pred['cam'][:true_size].shape == new_cam_pose.shape):
            pdb.set_trace()
        codes_pred['cam'] = torch.cat([codes_pred['cam'][:true_size], new_cam_pose])

        cam_hypos_flip = self.reflect_cam_pose(cams_hypo[:true_size])
        #codes_pred['cam_hypotheses'] = torch.cat([codes_pred['cam_hypotheses'][:true_size], new_cam_hypos])
        return cam_hypos_flip

    def forward(self, iters=0):
        opts = self.opts

        warmup_pose = False
        warmup_shape = False
        vertex_update=True

        self.conv_features = []
        self.enc_attn_weights = []
        self.dec_attn_weights = []

        if opts.warmup_pose_iter > iters:
            # only train the pose predictor. Without training the probs.
            warmup_pose = True

        if opts.warmup_shape_iter > iters:
            # do not update mean shape.
            warmup_shape = True

        if opts.warmup_shape_iter > 0:
            vertex_update = iters % (2*opts.warmup_shape_iter) >= opts.warmup_shape_iter

        inp_imgs = torch.cat([self.input_imgs, self.input_imgs_2], dim=0)
        #fwd_start_time = time.time()
        if opts.texture:
            pred_codes, self.textures = self.model(inp_imgs)
            #self.textures, self.textures_2 = self.textures.chunk(2)
        else:
            pred_codes = self.model(inp_imgs)

        #pred_codes = self.flip_train_predictions_swap(pred_codes, true_size=self.masks.shape[0])

        #self.fwd_time = time.time()-fwd_start_time
        self.delta_v = pred_codes['shape_pred']
        if opts.seg_train:
            vertex_part_pred = pred_codes['vertex_part_pred']
            #vertex_part_pred, vertex_part_pred_2 = vertex_part_pred.chunk(2)
            if not opts.only_mean_sym:
                vertex_part_pred = torch.cat([vertex_part_pred, vertex_part_pred[:, -self.model.num_sym:]], 1)
            #    vertex_part_pred_2 = torch.cat([vertex_part_pred_2, vertex_part_pred_2[:, -self.model.num_sym:]], 1)
            self.vertex_seg_map = vertex_part_pred  # (B x Nv x Np)
            #self.vertex_seg_map_2 = vertex_part_pred_2  # (B x Nv x Np)

        #self.delta_v, self.delta_v_2 = self.delta_v.chunk(2)
        #scale, scale_2 = scale.chunk(2, dim=1)
        #trans, trans_2 = trans.chunk(2, dim=1)
        #quat, quat_2 = quat.chunk(2, dim=1)
        #scores, scores_2 = scores.chunk(2, dim=1)

        cams_all_hypo = pred_codes['cam_hypotheses']
        self.cam_hypotheses = pred_codes['cam_hypotheses']
        cam_probs = pred_codes['cam_probs']
        # pGT to train cam for flipped images
        cams_all_hypo_flipped = self.flip_train_predictions_swap(cams_all_hypo, self.opts.batch_size)
        cams_all_hypo = torch.cat([cams_all_hypo[:self.opts.batch_size], cams_all_hypo_flipped])
        cams_probs = torch.cat([cam_probs[:self.opts.batch_size]] * 2)
        #cams_pred_flip = cams_all_flip_hypo[torch.arange(cams_all_flip_hypo.shape[0]), cam_flip_probs.argmax(1)]

        if opts.only_mean_sym:
            del_v = self.delta_v
        else:
            del_v = self.model.symmetrize(self.delta_v)

        if opts.only_mean_sym:
            del_v_2 = self.delta_v_2
        else:
            del_v_2 = self.model.symmetrize(self.delta_v_2)

        # Deform mean shape:
        self.mean_shape = self.model.get_mean_shape()
        if warmup_shape:
            self.mean_shape = self.mean_shape.detach()
        self.pred_v = self.mean_shape + del_v * (0 if warmup_pose else 1)
        #self.pred_v_2 = self.mean_shape + del_v_2 * (0 if warmup_pose else 1)

        if warmup_pose or (not vertex_update):
            self.pred_v = self.pred_v.detach()
            # self.pred_v_2 = self.pred_v_2.data#detach()

        # Compute keypoints.
        self.vert2kp = torch.nn.functional.softmax(self.model.vert2kp, dim=1)
        self.kp_verts = torch.matmul(self.vert2kp, self.pred_v)

        # Decide which camera to use for projection.
        # if opts.use_gtpose:
        #    proj_cam = self.cams
        # else:
        proj_cam = cams_all_hypo

        cam_probs = cam_probs.t().contiguous()  # Nc x B
        self.cam_probs = cam_probs
        # reshape cameras to Nc*B x 7
        num_cams = self.num_cams

        proj_cam = proj_cam.permute(1, 0, 2).contiguous()  # Nc x B x 7
        proj_cam = proj_cam.view(-1, *proj_cam.shape[2:])  # Nc*B x 7

        all_imgs = torch.cat([self.imgs, self.imgs2])

        if opts.texture:
            if opts.predict_flow:
                self.texture_flow = self.textures
                #self.texture_flow_2_pgt = self.texture_flow.detach()
                #self.texture_flow_2_pgt[:,:,:,:,0] = -self.texture_flow_2_pgt[:,:,:,:,0]

                #all_imgs = torch.cat([self.imgs, self.imgs2])
                # b x f x t*t x 3
                self.textures = geom_utils.sample_textures(self.texture_flow, all_imgs)

                self.textures_vis = geom_utils.sample_textures_2(self.texture_flow, all_imgs)
                tex_size = self.textures_vis.size(2)
                self.textures_vis = self.textures_vis.unsqueeze(4).repeat(1, 1, 1, 1, tex_size, 1)
                #tex_size = self.textures.size(2)
                #self.textures_2 = self.textures.unsqueeze(4).repeat(1, 1, 1, 1, tex_size, 1)
                #print(self.textures.shape, 'final tex_pred')
                #tex_start_time = time.time()
                #ts = torch.cat([self.textures]*num_cams)
                #swap_texture_pred = self.tex_renderer(self.pred_v, self.faces, swap_cams, textures=self.textures)
            else:
                B, F, T, T, C = self.textures.shape  # (B x F x T x T x 3)
                self.textures_vis = self.textures.unsqueeze(4).repeat(1, 1, 1, 1, T, 1)
                self.textures = self.textures.view(B, F, T * T, C)
        else:
            self.textures = None
        # Render mask.
        vs = torch.cat([self.pred_v] * num_cams)
        fs = torch.cat([self.faces] * num_cams)
        ts = torch.cat([self.textures] * num_cams)
        mask_start = time.time()
        self.mask_pred = self.renderer(vs, fs, proj_cam)  # Nc*B x H x W
        self.texture_pred = self.tex_renderer(vs, fs, proj_cam, textures=ts)
        # render segmentation
        if opts.seg_train:
            seg_img_list = []
            for k in range(opts.num_parts):
                texture_vert = self.vertex_seg_map[:, :, k:k + 1].repeat(1, 1, 3)  # (b x nv x 1) --> (b x nv x 3)
                temp_img = self.vertex_seg_renderer(vs, fs, proj_cam, torch.cat([texture_vert]*num_cams), True)
                temp_img = temp_img[:, 0:1, :, :]  # (nc*b x 3 x h x w) --> (nc*b x 1 x h x w)
                seg_img_list.append(temp_img)

            seg_imgs = torch.cat(seg_img_list, dim=1).permute(0, 2, 3, 1).contiguous()  # (nc*b x np x h x w) --> (nc*b x h x w x np)
            seg_imgs = torch.nn.functional.softmax(seg_imgs, dim=1)
            self.seg_imgs = seg_imgs.chunk(num_cams)
        #self.proj_verts = self.renderer.project_points(vs, proj_cam)

        # Need to take ths loss first to decide which cam to use
        masks_gt = torch.cat([self.masks, self.masks_2])
        cam_choose_loss_per_instance, mask_loss, tex_loss = self.cam_choose_crit(self.mask_pred, torch.cat([masks_gt] * num_cams),
                                                                                 self.texture_pred, torch.cat([all_imgs] * num_cams),
                                                                                 self.texture_loss)
        cam_choose_loss_per_instance = torch.stack(cam_choose_loss_per_instance.chunk(num_cams)).detach()  # Nc x B

        if self.opts.ucmr_prob:
            loss_based_cam_prob = torch.nn.functional.softmin(cam_choose_loss_per_instance, dim=0)
            cam_probs = loss_based_cam_prob.detach().data

        # for each instance, find the camera with least loss
        cam_ids = cam_choose_loss_per_instance.argmin(0)

        if not self.opts.ucmr_prob:
            self.mask_pred = torch.stack(self.mask_pred.chunk(num_cams))
            self.mask_pred = self.mask_pred[cam_ids, torch.arange(self.mask_pred.shape[1])]
            self.texture_pred = torch.stack(self.texture_pred.chunk(num_cams))
            self.texture_pred = self.texture_pred[cam_ids, torch.arange(self.texture_pred.shape[1])]

        proj_cam = torch.stack(proj_cam.chunk(num_cams))  # Nc x B x 7
        proj_cam = proj_cam[cam_ids, torch.arange(proj_cam.shape[1])]  # B x 7
        if self.opts.ucmr_prob:
            self.cam_pred = proj_cam.detach().data
        #cams_flip_pGT = cams_all_hypo_flipped[torch.arange(cams_all_hypo_flipped.shape[0]), cam_ids]
        #self.cam_pred = proj_cam
        #swap_cams = torch.cat(proj_cam.chunk(2)[::-1])
        #swap_masks = self.renderer(self.pred_v, self.faces, swap_cams)
        #Project keypoints
        self.kp_pred = self.renderer.project_points(self.kp_verts, self.cam_pred)

        #proj_cam = torch.stack(proj_cam.chunk(num_cams)) # nc x b x 7
        #proj_cam = proj_cam[cam_ids, torch.arange(proj_cam.shape[1])]

        #seg_start_time = time.time()
        #self.seg_img = self.vertex_seg_renderer(self.pred_v, self.faces, proj_cam, textures=self.vertex_seg_map).permute(0, 2, 3, 1)
        #self.seg_time = time.time() - seg_start_time

        # semantic consistency loss
        if opts.seg_train:
            images_zoo = inp_imgs#all_imgs
            labels = torch.cat([self.masks.unsqueeze(1), self.masks_2.unsqueeze(1)])
            #labels[labels < 0.1] = 0
            with torch.no_grad():
                zoo_feats = self.zoo_feat_net(images_zoo)
                zoo_feat = torch.cat([self.interp(zoo_feat) for zoo_feat in zoo_feats], dim=1).detach()
                # saliency masking
                if labels is not None:
                    zoo_feat = zoo_feat * labels
        # compute losses for this instance.
        #self.kp_loss = self.projection_loss(self.kp_pred, self.kps)
        # if warmup_pose:
        #    cam_probs = torch.ones_like(cam_probs) / num_cams

        #masks =  torch.cat([self.masks]*num_cams)
        dts_ncb = torch.cat([self.dts_barrier] * num_cams)
        all_masks = torch.cat([masks_gt] * num_cams)  # torch.cat([self.masks, self.masks_2])
        self.mask_loss = self.mask_loss_fn(self.mask_pred, all_masks, cam_probs, num_cams)
        self.cam_loss = self.camera_loss(self.cam_pred[:self.cam_pred.shape[0] // 2, -4:], self.cams[:, -4:], 0)
        self.cam_ce_loss = self.ce_loss(cam_probs, cam_ids)  # * (0 if warmup_pose else 1)
        #self.cam_flip_loss = torch.nn.functional.mse_loss(cams_pred_flip, cams_flip_pgt)
        #self.cam_entropy_loss = self.entropy_cam_loss(cam_probs) * (0 if warmup_pose else 1)
        self.rot_reg = 0.
        if self.num_cams > 1:
           self.rot_reg = self.rot_loss(cams_all_hypo, self.num_cams) * (0 if (warmup_pose and self.opts.az_ele_quat) else 1)
        #self.mask_dt_loss = self.mask_dt_loss_fn(self.proj_verts, dts_ncb, cam_probs, num_cams)
        if opts.texture:
            tex_loss = torch.stack(tex_loss.chunk(num_cams))  # Nc x B
            self.tex_loss = (tex_loss * cam_probs).sum(0).mean()
            #self.tex_loss = tex_loss[cam_ids, torch.arange(tex_loss.shape[1])].mean()
            # self.texture_loss(self.texture_pred, all_imgs, self.mask_pred, all_masks)#, cam_probs, num_cams)
            # tex_loss = torch.stack(tex_loss.chunk(num_cams))  # Nc x B
            if opts.predict_flow:
                self.tex_dt_loss = self.texture_dt_loss_fn(self.texture_flow, torch.cat([self.dts_barrier, self.dts_barrier_2]))  # , probs=cam_probs, num_cams=num_cams)

        #self.vert2part_diversity = torch.nn.functional.softmax(
        #        self.model.code_predictor.vertex_part_predictor.vert2part.squeeze(0), dim=0).t()
        # priors:
        #self.vert2kp_loss = self.entropy_loss(self.vert2kp)
        #self.vert2part_loss = (self.entropy_loss(self.vert2part) - self.entropy_loss(self.vert2part_diversity)) / 2.
        self.deform_reg = self.deform_reg_fn(self.delta_v)
        #self.triangle_loss = self.triangle_loss_fn(self.pred_v)
        self.triangle_loss = self.triangle_loss_fn(self.pred_v).mean()
        self.flatten_loss = self.flatten_loss_fn(self.pred_v).mean()

        # new losses
        #loss_start_time = time.time()
        #self.deform_flip_loss = torch.nn.functional.mse_loss(del_v_2, del_v.detach())
        if opts.seg_train:
            vertex_seg_map, vertex_seg_map_flip = self.vertex_seg_map.chunk(2)
            pred_v, pred_v_flip = self.pred_v.chunk(2)
            self.invar_loss = self.invariance_loss(vertex_seg_map_flip, vertex_seg_map.detach())
            self.chamf_loss = self.chamfer_loss(pred_v.detach(), pred_v_flip)
            self.conc_loss = self.concentration_loss(self.vertex_seg_map, self.pred_v.detach())

            if not opts.query_basis:
                part_basis = self.part_basis_generator()
            else:
                part_basis = self.model.get_query_basis()

            self.sc_loss = sum([\
                    self.semantic_consistency_loss(zoo_feat, self.seg_imgs[i], \
                                                   part_basis, cam_probs[i:i+1], 1)/num_cams \
                    for i in range(num_cams)]) #* (0 if warmup_pose else 1)  # , cam_probs, num_cams)
            self.vert2part = torch.nn.functional.softmax(self.model.code_predictor.vertex_part_predictor.vert2part.squeeze(0), dim=1)
            self.vert2part_loss = (self.entropy_loss(self.vert2part))# - self.entropy_loss(self.vert2part_diversity)) / 2.
            self.orth_loss = self.orthonormal_loss(part_basis) #* (0 if warmup_pose else 1)

        # finally sum up the loss.
        # instance loss:
        self.total_loss = 0.
        #self.total_loss = opts.kp_loss_wt * self.kp_loss
        self.total_loss += opts.mask_loss_wt * self.mask_loss
        #self.total_loss += opts.cam_loss_wt * self.cam_loss
        self.total_loss += opts.cam_ce_loss_wt * self.cam_ce_loss
        #self.total_loss += opts.cam_ent_loss_wt * self.cam_entropy_loss
        self.total_loss += opts.rot_reg * self.rot_reg
        if opts.texture:
            self.total_loss += opts.tex_loss_wt * self.tex_loss

        # priors:
        #self.total_loss += opts.vert2kp_loss_wt * self.vert2kp_loss
        self.total_loss += opts.deform_reg_wt * self.deform_reg
        self.total_loss += opts.triangle_reg_wt * self.triangle_loss
        self.total_loss += opts.flatten_reg_wt * self.flatten_loss
        if opts.predict_flow:
            self.total_loss += opts.tex_dt_loss_wt * self.tex_dt_loss
        #self.total_loss += opts.mask_dt_loss_wt * self.mask_dt_loss

        # new stuff
        if opts.seg_train:
            self.total_loss += opts.invar_loss_wt * self.invar_loss
            self.total_loss += opts.chamf_loss_wt * self.chamf_loss
            self.total_loss += opts.orth_loss_wt * self.orth_loss
            self.total_loss += opts.conc_loss_wt * self.conc_loss
            self.total_loss += opts.sc_loss_wt * self.sc_loss
            self.total_loss += opts.vert2part_loss_wt * self.vert2part_loss

        if self.opts.ucmr_prob:  # for visuals
            self.mask_pred = torch.stack(self.mask_pred.chunk(num_cams))
            self.mask_pred = self.mask_pred[cam_ids, torch.arange(self.mask_pred.shape[1])]
            self.texture_pred = torch.stack(self.texture_pred.chunk(num_cams))
            self.texture_pred = self.texture_pred[cam_ids, torch.arange(self.texture_pred.shape[1])]

    def get_current_visuals(self):
        vis_dict = {}
        #masks_pred = self.mask_pred.view(self.num_cams, *self.masks.shape)
        # masks_pred = masks_pred[self.cam_probs.argmax(0), torch.arange(masks_pred.shape[1])] # b x h x w
        b = self.masks.shape[0]
        all_masks = torch.cat([self.masks, self.masks_2])
        mask_concat = torch.cat([all_masks, self.mask_pred], 2)

        all_imgs = torch.cat([self.imgs, self.imgs2])
        if self.opts.texture:
            if self.opts.predict_flow:
                # b x 2 x h x w
                uv_flows = self.model.texture_predictor.uvimage_pred
                # b x h x w x 2
                b = all_imgs.shape[0]
                uv_flows = uv_flows.permute(0, 2, 3, 1).contiguous()
                with torch.cuda.amp.autocast(enabled=False):
                    uv_images = torch.nn.functional.grid_sample(all_imgs.float(), uv_flows.float(), align_corners=True)

        num_show = min(2, self.opts.batch_size)
        num_show = list(range(num_show//2)) + list(range(self.opts.batch_size, self.opts.batch_size + num_show//2))
        show_uv_imgs = []
        show_uv_flows = []

        kps_2 = self.kps.clone()
        kps_2[:, 0] = -self.kps[:, 0]
        all_kps =  torch.cat([self.kps, kps_2])
        for i in num_show:
            input_img = bird_vis.kp2im(all_kps[i].data, all_imgs[i].data)
            pred_kp_img = bird_vis.kp2im(self.kp_pred[i].data, all_imgs[i].data)
            masks = bird_vis.tensor2mask(mask_concat[i].data)
            if self.opts.texture:
                texture_here = self.textures_vis[i]
            else:
                texture_here = None

            rend_predcam = self.vis_rend(self.pred_v[i], self.cam_pred[i], texture=texture_here)
            # render from front & back:
            rend_frontal = self.vis_rend.diff_vp(self.pred_v[i], self.cam_pred[i], texture=texture_here, kp_verts=self.kp_verts[i])
            rend_top = self.vis_rend.diff_vp(self.pred_v[i], self.cam_pred[i], axis=[0, 1, 0], texture=texture_here, kp_verts=self.kp_verts[i])
            diff_rends = np.hstack((rend_frontal, rend_top))

            if self.opts.texture:
                if self.opts.predict_flow:
                    uv_img = bird_vis.tensor2im(uv_images[i].data)
                    show_uv_imgs.append(uv_img)
                    uv_flow = bird_vis.visflow(uv_flows[i].data)
                    show_uv_flows.append(uv_flow)

                tex_img = bird_vis.tensor2im(self.texture_pred[i].data)
                imgs = np.hstack((input_img, pred_kp_img, tex_img))
            else:
                imgs = np.hstack((input_img, pred_kp_img))

            rend_gtcam = self.vis_rend(self.pred_v[i], self.cams[i%self.opts.batch_size], texture=texture_here)
            rends = np.hstack((diff_rends, rend_predcam, rend_gtcam))
            vis_dict['%d' % i] = np.hstack((imgs, rends, masks))
            vis_dict['masked_img %d' % i] = bird_vis.tensor2im((all_imgs[i] * all_masks[i]).data)

        ###################################################
        ############## Decoder Visualization ##############
        ###################################################

        im = np.asarray(torchvision.transforms.functional.to_pil_image(self.imgs[0].detach().cpu()))
        img = np.asarray(torchvision.transforms.functional.to_pil_image(self.input_imgs[0].detach().cpu()))

        # don't need the list anymore
        conv_features = self.conv_features[0]
        enc_attn_weights = self.enc_attn_weights[0]
        dec_attn_weights = self.dec_attn_weights[0]

         # get the feature map shape
        h, w = 16, 16#conv_features.shape[-2:]

        # colors for visualization
        COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                  [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

        fig, axs = plt.subplots(ncols=8, nrows=2, figsize=(22, 7))
        colors = COLORS * 100
        for idx, ax_i in zip(range(self.num_cams), axs.T):
            ax = ax_i[0]
            ax.imshow(dec_attn_weights[idx].view(h, w))
            ax.axis('off')
            ax.set_title(f'query id: {idx}')
            ax = ax_i[1]
            ax.imshow(im)

            ax.axis('off')
        fig.tight_layout()

        # Saves in a buffer and open that as PIL image. No need to actually save in disk
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        vis_dict['dec_vis'] = np.asarray(Image.open(buf))
        buf.close()

        ###################################################
        ############## Encoder Visualization ##############
        ###################################################
        sattn = enc_attn_weights.reshape((h, w, h, w)).detach().cpu()

        # downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
        fact = 256 // 16

        # let's select 4 reference points for visualization
        idxs = [(70, 110), (150, 100), (150, 150), (120, 180),]

        # here we create the canvas
        fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
        # and we add one plot per reference point
        gs = fig.add_gridspec(2, 4)
        axs = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[0, -1]),
            fig.add_subplot(gs[1, -1]),
        ]

        # for each one of the reference points, let's plot the self-attention
        # for that point
        for idx_o, ax in zip(idxs, axs):
            idx = (idx_o[0] // fact, idx_o[1] // fact)
            ax.imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
            ax.axis('off')
            ax.set_title(f'self-attention{idx_o}')

        # and now let's add the central image, with the reference points as red circles
        fcenter_ax = fig.add_subplot(gs[:, 1:-1])


        fcenter_ax.imshow(im)
        for (y, x) in idxs:
            scale = im.shape[0] / img.shape[1]
            x = ((x // fact) + 0.5) * fact
            y = ((y // fact) + 0.5) * fact
            fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='b'))
            fcenter_ax.axis('off')


        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        vis_dict['enc_vis'] = np.asarray(Image.open(buf))
        buf.close()

        if self.opts.texture and self.opts.predict_flow:
            vis_dict['uv_images'] = np.hstack(show_uv_imgs)
            vis_dict['uv_flow_vis'] = np.hstack(show_uv_flows)


        fig, axs = plt.subplots(ncols=8, nrows=1, figsize=(22, 7))
        #texture = textures[0].view(textures.shape[1], )
        for i in num_show:
            fig, axs = plt.subplots(ncols=self.num_cams, nrows=1, figsize=(22, 7))
            for ax, cam in zip(axs, self.cam_hypotheses[i]):
                img_pred = self.vis_rend(self.pred_v[i], cam, texture=self.textures_vis[i])
                ax.imshow(img_pred)
                ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            vis_dict['cam_vis_%d'%i] = np.asarray(Image.open(buf))
            buf.close()

        return vis_dict

    def get_current_points(self):
        return {
            'mean_shape': visutil.tensor2verts(self.mean_shape.data),
            'verts': visutil.tensor2verts(self.pred_v.data),
        }

    def get_current_scalars(self):
        sc_dict = OrderedDict([
            ('smoothed_total_loss', self.smoothed_total_loss),
            ('total_loss', self.total_loss.item()),
            ('kp_loss', self.kp_loss.item()),
            ('mask_loss', self.mask_loss.item()),
            ('vert2kp_loss', self.vert2kp_loss.item()),
            ('deform_reg', self.deform_reg.item()),
            ('tri_loss', self.triangle_loss.item()),
            ('flatten_loss', self.flatten_loss.item()),
            ('cam_loss', self.cam_loss.item()),
            ('cam_ce_loss', self.cam_ce_loss.item()),
            ('cam_flip_loss', self.cam_flip_loss.item()),
            ('deform_flip_loss', self.deform_flip_loss.item()),
            ('tex_flip_loss', self.tex_flip_loss.item()),
            ('cam_entropy_loss', self.cam_entropy_loss.item()),
            ('rot_reg', self.rot_reg.item()),
            ('mask_dt_loss', self.mask_dt_loss.item()),
        ])
        if self.opts.texture:
            sc_dict['tex_loss'] = self.tex_loss.item()
            if opts.predict_flow:
                sc_dict['tex_dt_loss'] = self.tex_dt_loss.item()
        if self.opts.seg_train:
            sc_dict['conc_loss'] = self.conc_loss.item()
            sc_dict['sc_loss'] = self.sc_loss.item()
            sc_dict['vert2part_loss'] = self.vert2part_loss.item()
            sc_dict['invar_loss'] =  self.invar_loss.item()
            sc_dict['chamf_loss'] = self.chamf_loss.item()
            sc_dict['orth_loss'] = self.orth_loss.item()

        return sc_dict


def main(_):
    torch.manual_seed(0)
    trainer = ShapeTrainer(opts)
    trainer.init_training()
    #fwd, bwd, proj, mask, tex, seg,  loss = trainer.train()
    trainer.train()
    '''print('Fwd time/iter: ', np.array(fwd).mean())
    print('Bwd time/iter: ', np.array(bwd).mean())
    print('proj time/iter: ', np.array(proj).mean())
    print('mask time/iter: ', np.array(mask).mean())
    print('tex time/iter: ', np.array(tex).mean())
    print('seg time/iter: ', np.array(seg).mean())
    print('loss time/iter: ', np.array(loss).mean())
    return'''


if __name__ == '__main__':
    app.run(main)
