"""
Mesh net model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import os
import os.path as osp
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

import sys
from utils import mesh
from utils import geometry as geom_utils
from . import net_blocks as nb

import soft_renderer as sr

from nnutils.feature_extraction import featureL2Norm

#-------------- flags -------------#
#----------------------------------#
flags.DEFINE_boolean('symmetric', True, 'Use symmetric mesh or not')
flags.DEFINE_integer('nz_feat', 200, 'Encoded feature size')

flags.DEFINE_boolean('texture', True, 'if true uses texture!')
flags.DEFINE_boolean('symmetric_texture', True, 'if true texture is symmetric!')
flags.DEFINE_integer('tex_size', 6, 'Texture resolution per face')

flags.DEFINE_integer('subdivide', 3, '# to subdivide icosahedron, 3=642verts, 4=2562 verts')

flags.DEFINE_boolean('use_deconv', False, 'If true uses Deconv')
flags.DEFINE_string('upconv_mode', 'bilinear', 'upsample mode')

flags.DEFINE_boolean('only_mean_sym', False, 'If true, only the meanshape is symmetric')

flags.DEFINE_boolean('multiple_cam_hypo', True, 'Multiple camera hypothesis')
flags.DEFINE_boolean('trans_cam', True, 'Transformer camera')
flags.DEFINE_integer('num_hypo_cams', 8, 'number of hypo cams')
flags.DEFINE_boolean('az_ele_quat', True, 'Predict camera as azi elev')
flags.DEFINE_float('scale_lr_decay', 0.05, 'Scale multiplicative factor')
flags.DEFINE_float('scale_bias', .75, 'Scale bias factor')  # NOTE: originlly 1.
flags.DEFINE_string('csm_bird_template_path', 'misc/cachedir/cub/symmetric_csm_bird.obj', 'Template of bird used by CSM')
flags.DEFINE_integer('num_encoder_layers', 2, 'Number of Encoders in transformer')
flags.DEFINE_integer('num_decoder_layers', 2, 'Number of Decoders in transformer')
flags.DEFINE_integer('num_heads', 8, 'Number of heads in multi-head attn in transformer')

#------------- Modules ------------#
#----------------------------------#


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 transpose convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                              padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 transpose convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class TexturePredictor(nn.Module):
    def __init__(self, uv_sampler, opts):
        super(TexturePredictor, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.resnet_block1 = BasicBlock(256, 256)
        self.resnet_block2 = BasicBlock(256, 256)
        self.resnet_block3 = BasicBlock(256, 256)
        self.resnet_block4 = BasicBlock(256, 128, downsample=conv1x1(256, 128))
        self.resnet_block5 = BasicBlock(128, 64, downsample=conv1x1(128, 64))
        self.resnet_block6 = BasicBlock(64, 32, downsample=conv1x1(64, 32))
        self.resnet_block7 = BasicBlock(32, 16, downsample=conv1x1(32, 16))
        x = torch.tensor([0., 0., 1., 1., 2., 2., 3., 3.]).unsqueeze(0).repeat(4, 1)
        y = torch.linspace(0, 3, steps=8).unsqueeze(0).repeat(4, 1)
        self.grid = torch.stack((x, y), dim=2).unsqueeze(0).repeat(32, 1, 1, 1).cuda()
        self.final_conv = nb.conv2d(True, 16, 3, stride=1, kernel_size=1)
        self.F = uv_sampler.size(1)
        self.T = uv_sampler.size(2)
        self.symmetric = opts.symmetric_texture
        self.num_sym_faces = 624
        # B x F x T x T x 2 --> B x F x T*T x 2
        self.uv_sampler = uv_sampler.view(-1, self.F, self.T * self.T, 2)

    def forward(self, feat):  # (feat is 32 x 256 x 4 x 4)
        B = feat.shape[0]
        grid = self.grid[0:1].repeat(B, 1, 1, 1)
        with torch.cuda.amp.autocast(enabled=False):
            x = torch.nn.functional.grid_sample(feat.float(), grid, align_corners=True)  # (32 x 256 x 4 x 8)
        x = self.resnet_block1(x)
        x = self.upsample(x)
        x = self.resnet_block2(x)
        x = self.resnet_block3(x)
        x = self.upsample(x)
        x = self.resnet_block4(x)
        x = self.upsample(x)
        x = self.resnet_block5(x)
        x = self.upsample(x)
        x = self.resnet_block6(x)
        x = self.upsample(x)
        x = self.resnet_block7(x)
        x = self.final_conv(x)
        B = x.size(0)
        with torch.cuda.amp.autocast(enabled=False):
            tex_pred = torch.nn.functional.grid_sample(x.float(), self.uv_sampler[:B], align_corners=True)
        tex_pred = tex_pred.view(x.size(0), -1, self.F, self.T, self.T).permute(0, 2, 3, 4, 1)
        if self.symmetric:
            # Symmetrize.
            tex_left = tex_pred[:, -self.num_sym_faces:]
            tex_full = torch.cat([tex_pred, tex_left], 1)
            return tex_full
        else:
            # Contiguous Needed after the permute..
            return tex_pred.contiguous()


class ResNetConv(nn.Module):
    def __init__(self, n_blocks=4):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.n_blocks = n_blocks

    def forward(self, x):
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
        if n_blocks >= 4:
            x = self.resnet.layer4(x)
        return x


class Encoder(nn.Module):
    """
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat.
    """

    def __init__(self, input_shape, n_blocks=4, nz_feat=100, batch_norm=True, hidden_enc_dim=256):
        super(Encoder, self).__init__()
        self.resnet_conv = ResNetConv(n_blocks=n_blocks)
        in_enc_dim = 2**n_blocks * 32
        self.enc_conv1 = nb.conv2d(batch_norm, in_enc_dim, hidden_enc_dim, stride=2, kernel_size=4)
        #self.enc_conv2 = nn.Conv2d(512, 512, 1)
        # nb.net_init(self.enc_conv2)
        downsample = 2**n_blocks * 4
        nc_input = hidden_enc_dim * (input_shape[0] // downsample) * (input_shape[1] // downsample)
        self.enc_fc = nb.fc_stack(nc_input, nz_feat, 2)

        nb.net_init(self.enc_conv1)

    def forward(self, img):
        resnet_feat = self.resnet_conv.forward(img)
        wide_img_feat = self.enc_conv1(resnet_feat)

        #out_enc_conv1 = self.enc_conv1(resnet_feat)
        out_enc_conv1 = wide_img_feat.view(img.size(0), -1)
        feat = self.enc_fc.forward(out_enc_conv1)

        return feat, resnet_feat


class TexturePredictorUV(nn.Module):
    """
    Outputs mesh texture
    """

    def __init__(self, nz_feat, uv_sampler, opts, img_H=64, img_W=128, n_upconv=5, nc_init=256, predict_flow=False, symmetric=False, num_sym_faces=624):
        super(TexturePredictorUV, self).__init__()
        self.feat_H = img_H // (2 ** n_upconv)
        self.feat_W = img_W // (2 ** n_upconv)
        self.nc_init = nc_init
        self.symmetric = symmetric
        self.num_sym_faces = num_sym_faces
        self.F = uv_sampler.size(1)
        self.T = uv_sampler.size(2)
        self.predict_flow = predict_flow
        # B x F x T x T x 2 --> B x F x T*T x 2
        self.uv_sampler = uv_sampler.view(-1, self.F, self.T * self.T, 2)

        self.enc = nb.fc_stack(nz_feat, self.nc_init * self.feat_H * self.feat_W, 2)
        if predict_flow:
            nc_final = 2
        else:
            nc_final = 3
        self.decoder = nb.decoder2d(n_upconv, None, nc_init, init_fc=False, nc_final=nc_final, use_deconv=opts.use_deconv, upconv_mode=opts.upconv_mode)

    def forward(self, feat):
        # pdb.set_trace()
        uvimage_pred = self.enc.forward(feat)
        uvimage_pred = uvimage_pred.view(uvimage_pred.size(0), self.nc_init, self.feat_H, self.feat_W)
        # B x 2 or 3 x H x W
        self.uvimage_pred = self.decoder.forward(uvimage_pred)
        if self.predict_flow:
            self.uvimage_pred = torch.tanh(self.uvimage_pred)
        else:
            self.uvimage_pred = torch.sigmoid(self.uvimage_pred)
        B = self.uvimage_pred.size(0)
        with torch.cuda.amp.autocast(enabled=False):
            tex_pred = torch.nn.functional.grid_sample(self.uvimage_pred.float(), self.uv_sampler[:B], align_corners=True)
        tex_pred = tex_pred.view(uvimage_pred.size(0), -1, self.F, self.T, self.T).permute(0, 2, 3, 4, 1)

        if self.symmetric:
            # Symmetrize.
            tex_left = tex_pred[:, -self.num_sym_faces:]
            tex_full = torch.cat([tex_pred, tex_left], 1)
            return tex_full
        else:
            # Contiguous Needed after the permute..
            return tex_pred.contiguous()


class ShapePredictor(nn.Module):
    """
    Outputs mesh deformations
    """

    def __init__(self, nz_feat, num_verts):
        super(ShapePredictor, self).__init__()
        # self.pred_layer = nb.fc(True, nz_feat, num_verts)
        self.pred_layer = nn.Linear(nz_feat, num_verts * 3)

        # Initialize pred_layer weights to be small so initial def aren't so big
        self.pred_layer.weight.data.normal_(0, 0.0001)

    def forward(self, feat):
        # pdb.set_trace()
        delta_v = self.pred_layer.forward(feat)
        # Make it B x num_verts x 3
        delta_v = delta_v.view(delta_v.size(0), -1, 3)
        # print('shape: ( Mean = {}, Var = {} )'.format(delta_v.mean().data[0], delta_v.var().data[0]))
        return delta_v


class QuatPredictor(nn.Module):

    def __init__(self, nz_feat, nz_rot=4, classify_rot=False):
        super(QuatPredictor, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, nz_rot)
        self.classify_rot = classify_rot

    def forward(self, feat):
        quat = self.pred_layer.forward(feat)
        if self.classify_rot:
            quat = torch.nn.functional.log_softmax(quat)
        else:
            quat = torch.nn.functional.normalize(quat)

        return quat

    def initialize_to_zero_rotation(self,):
        nb.net_init(self.pred_layer)
        self.pred_layer.bias = nn.Parameter(torch.FloatTensor([1, 0, 0, 0]).type(self.pred_layer.bias.data.type()))
        return

class QuatPredictorAzEle(nn.Module):

    def __init__(self, nz_feat, dataset='others'):
        super(QuatPredictorAzEle, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, 3)
        self.axis = torch.eye(3).float().cuda()
        self.dataset = dataset

    def forward(self, feat, return_az_ele=False):
        angles = 0.1 * self.pred_layer.forward(feat)
        angles = torch.tanh(feat)
        azimuth = np.pi / 6 * angles[..., 0]

        # # Birds
        if self.dataset == 'cub':
            elev = np.pi / 2 * (angles[..., 1])
            cyc_rot = np.pi / 3 * (angles[..., 2])
        else:
            # cars # Horse & Sheep
            elev = np.pi / 9 * (angles[..., 1])
            cyc_rot = np.pi / 9 * (angles[..., 2])
        if return_az_ele:
            return angles[..., 0], angles[..., 1]
        q_az = self.convert_ax_angle_to_quat(self.axis[1], azimuth)
        q_el = self.convert_ax_angle_to_quat(self.axis[0], elev)
        q_cr = self.convert_ax_angle_to_quat(self.axis[2], cyc_rot)
        quat = geom_utils.hamilton_product(q_el.unsqueeze(1), q_az.unsqueeze(1))
        quat = geom_utils.hamilton_product(q_cr.unsqueeze(1), quat)
        return quat.squeeze(1)

    def convert_ax_angle_to_quat(self, ax, ang):
        qw = torch.cos(ang / 2)
        qx = ax[0] * torch.sin(ang / 2)
        qy = ax[1] * torch.sin(ang / 2)
        qz = ax[2] * torch.sin(ang / 2)
        quat = torch.stack([qw, qx, qy, qz], dim=1)
        return quat

    def initialize_to_zero_rotation(self,):
        nb.net_init(self.pred_layer)
        return

class ScalePredictor(nn.Module):

    def __init__(self, nz, bias=1.0, lr=0.05):
        super(ScalePredictor, self).__init__()
        self.pred_layer = nn.Linear(nz, 1)
        self.lr = lr
        self.bias = bias

    def forward(self, feat):
        scale = self.lr * self.pred_layer.forward(feat) + self.bias  # b
        scale = torch.nn.functional.relu(scale) + 1E-12  # minimum scale is 0.0
        return scale


class TransPredictor(nn.Module):
    """
    Outputs [tx, ty] or [tx, ty, tz]
    """

    def __init__(self, nz, orth=True):
        super(TransPredictor, self).__init__()
        if orth:
            self.pred_layer = nn.Linear(nz, 2)
        else:
            self.pred_layer = nn.Linear(nz, 3)

    def forward(self, feat):
        trans = self.pred_layer.forward(feat)
        return trans

class Camera(nn.Module):

    def __init__(self, nz_input, az_ele_quat=False, scale_lr=0.05, scale_bias=1.0, dataset='others'):
        super(Camera, self).__init__()
        self.fc_layer = nb.fc_stack(nz_input, nz_input, 2)

        if az_ele_quat:
            self.quat_predictor = QuatPredictorAzEle(nz_input, dataset)
        else:
            self.quat_predictor = QuatPredictor(nz_input)

        self.prob_predictor = nn.Linear(nz_input, 1)
        self.scale_predictor = ScalePredictor(nz_input, lr=scale_lr, bias=scale_bias)
        self.trans_predictor = TransPredictor(nz_input)

    def forward(self, feat):
        feat = self.fc_layer(feat)
        quat_pred = self.quat_predictor.forward(feat)
        prob = self.prob_predictor(feat)
        scale = self.scale_predictor.forward(feat)
        trans = self.trans_predictor.forward(feat)
        return torch.cat([quat_pred, prob, scale, trans], dim=1)

    def init_quat_module(self,):
        self.quat_predictor.initialize_to_zero_rotation()


class MultiCamPredictor(nn.Module):

    def __init__(self, nc_input, ns_input, nz_channels, nz_feat=100, num_cams=8,
                 aze_ele_quat=False, scale_lr=0.05, scale_bias=1.0, dataset='others'):
        super(MultiCamPredictor, self).__init__()

        self.fc = nb.fc_stack(nz_feat, nz_feat, 2, use_bn=False)
        self.scale_predictor = ScalePredictor(nz_feat)
        nb.net_init(self.scale_predictor)
        self.trans_predictor = TransPredictor(nz_feat)
        nb.net_init(self.trans_predictor)
        self.prob_predictor = nn.Linear(nz_feat, num_cams)
        self.camera_predictor = nn.ModuleList([Camera(nz_feat, aze_ele_quat, scale_lr=scale_lr,
                                                      scale_bias=scale_bias, dataset=dataset) for i in range(num_cams)])

        nb.net_init(self)
        for cx in range(num_cams):
            self.camera_predictor[cx].init_quat_module()

        self.quat_predictor = QuatPredictor(nz_feat)
        self.quat_predictor.initialize_to_zero_rotation()
        self.num_cams = num_cams

        base_rotation = torch.FloatTensor([0.9239, 0, 0.3827, 0]).unsqueeze(0).unsqueeze(0)  # pi/4
        # base_rotation = torch.FloatTensor([ 0.7071,  0 , 0.7071,   0]).unsqueeze(0).unsqueeze(0) ## pi/2
        base_bias = torch.FloatTensor([0.7071, 0.7071, 0, 0]).unsqueeze(0).unsqueeze(0)
        self.cam_biases = [base_bias]
        for i in range(1, self.num_cams):
            self.cam_biases.append(geom_utils.hamilton_product(base_rotation, self.cam_biases[i - 1]))
        self.cam_biases = torch.stack(self.cam_biases).squeeze().cuda()
        return

    def forward(self, feat):
        feat = self.fc(feat)
        cameras = []
        for cx in range(self.num_cams):
            cameras.append(self.camera_predictor[cx].forward(feat))
        cameras = torch.stack(cameras, dim=1)
        quats = cameras[:, :, 0:4]
        prob_logits = cameras[:, :, 4]
        camera_probs = nn.functional.softmax(prob_logits, dim=1)

        scale = self.scale_predictor.forward(feat).unsqueeze(1).repeat(1, self.num_cams, 1)
        trans = self.trans_predictor.forward(feat).unsqueeze(1).repeat(1, self.num_cams, 1)
        scale = cameras[:, :, 5:6]
        trans = cameras[:, :, 6:8]

        bias_quats = self.cam_biases.unsqueeze(0).repeat(len(quats), 1, 1)
        new_quats = geom_utils.hamilton_product(quats, bias_quats)
        cam = torch.cat([scale, trans, new_quats, camera_probs.unsqueeze(-1)], dim=2)
        return self.sample(cam) + (quats,)

    def sample(self, cam):
        '''
            cams : B x num_cams x 8 Vector. Last column is probs.
        '''
        dist = torch.distributions.multinomial.Multinomial(probs=cam[:, :, 7])
        sample = dist.sample()
        sample_inds = torch.nonzero(sample)[:, None, 1]
        if cam.shape[0] > 1:
            sampled_cam = torch.gather(cam, dim=1, index=sample_inds.unsqueeze(-1).repeat(1, 1, 8)).squeeze()[:, 0:7]
        else:
            sampled_cam = cam
        return sampled_cam, sample_inds, cam[:, :, 7], cam[:, :, 0:7]

class TransformerCamera(nn.Module):

    def __init__(self, nz_feat=512, num_cams=8, aze_ele_quat=False, scale_lr=0.05, scale_bias=1.0, dataset='others',
                 num_heads=8, num_encoder_layers=6, num_decoder_layers=6):
        super(TransformerCamera, self).__init__()

        self.fc = nb.fc_stack(nz_feat, nz_feat, 2, use_bn=False)
        self.camera_predictor = Camera(nz_feat, aze_ele_quat, scale_lr=scale_lr, scale_bias=scale_bias, dataset=dataset)
        self.query_pos = nn.Embedding(num_cams, nz_feat)
        self.transformer = nn.Transformer(nz_feat, num_heads, num_encoder_layers, num_decoder_layers)
        self.row_embed = nn.Parameter(torch.rand(50, nz_feat // 2))
        self.col_embed = nn.Parameter(torch.rand(50, nz_feat // 2))

        nb.net_init(self)
        self.camera_predictor.init_quat_module()

        self.quat_predictor = QuatPredictor(nz_feat)
        self.quat_predictor.initialize_to_zero_rotation()
        self.num_cams = num_cams

        base_rotation = torch.FloatTensor([0.9239, 0, 0.3827, 0]).unsqueeze(0).unsqueeze(0)  # pi/4
        # base_rotation = torch.FloatTensor([ 0.7071,  0 , 0.7071,   0]).unsqueeze(0).unsqueeze(0) ## pi/2
        base_bias = torch.FloatTensor([0.7071, 0.7071, 0, 0]).unsqueeze(0).unsqueeze(0)
        self.cam_biases = [base_bias]
        for i in range(1, self.num_cams):
            self.cam_biases.append(geom_utils.hamilton_product(base_rotation, self.cam_biases[i - 1]))
        self.cam_biases = torch.stack(self.cam_biases).squeeze().cuda()
        return

    def forward(self, h):
        # construct positional encodings
        B, C, H, W = h.shape
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        trans_in1 = pos + 0.1 * h.flatten(2).permute(2, 0, 1)  # (HW x B x C)
        trans_in2 = self.query_pos.weight.unsqueeze(1).repeat(1, B, 1)  # (Nq x B x C) (Nq == Np)

        h = self.transformer(trans_in1, trans_in2).transpose(0, 1).contiguous().view(-1, C)  # (B*NCam x 512)
        cameras = self.camera_predictor.forward(h).view(B, self.num_cams, -1)  # (B x NCam x 8)
        quats = cameras[:, :, 0:4]
        prob_logits = cameras[:, :, 4]
        camera_probs = nn.functional.softmax(prob_logits, dim=1)

        scale = cameras[:, :, 5:6]
        trans = cameras[:, :, 6:8]

        bias_quats = self.cam_biases.unsqueeze(0).repeat(len(quats), 1, 1)
        new_quats = geom_utils.hamilton_product(quats, bias_quats)
        cam = torch.cat([scale, trans, new_quats, camera_probs.unsqueeze(-1)], dim=2)
        return self.sample(cam) + (quats,)

    def sample(self, cam):
        '''
            cams : B x num_cams x 8 Vector. Last column is probs.
        '''
        B, Nc, _ = cam.shape
        dist = torch.distributions.multinomial.Multinomial(probs=cam[:, :, 7])
        #sample = dist.sample()
        #sample_inds = torch.nonzero(sample)[:, None, 1]
        sample_inds = torch.zeros((B, 1)).long().cuda()
        if cam.shape[0] > 1:
            sampled_cam = torch.gather(cam, dim=1, index=sample_inds.unsqueeze(-1).repeat(1, 1, 8)).squeeze()[:, 0:7]
        else:
            sampled_cam = cam
        return sampled_cam, sample_inds, cam[:, :, 7], cam[:, :, 0:7]


class VertexPartPredictor(nn.Module):
    """
    Outputs Vertex Part Predictions
    """

    def __init__(self, num_verts, num_parts):
        super(VertexPartPredictor, self).__init__()
        self.num_parts = num_parts

        vert2part_init = torch.cuda.FloatTensor(1, num_verts, num_parts).uniform_()
        self.vert2part = nn.Parameter(nn.functional.softmax(vert2part_init, dim=-1))

    def forward(self, feat):
        R_map = self.vert2part
        return R_map.repeat(feat.shape[0], 1, 1)


class CodePredictor(nn.Module):
    def __init__(self, opts, nz_feat=100, num_verts=1000, num_parts=4, multi_cam=True, hidden_enc_dim=256):
        super(CodePredictor, self).__init__()
        self.opts = opts
        self.shape_predictor = ShapePredictor(nz_feat, num_verts=num_verts)
        self.multi_cam = multi_cam
        if multi_cam:
            if not opts.trans_cam:
                self.cam_predictor = MultiCamPredictor(512, 8, 128, nz_feat=nz_feat,
                                                       num_cams=opts.num_hypo_cams, aze_ele_quat=opts.az_ele_quat,
                                                       scale_lr=opts.scale_lr_decay, scale_bias=opts.scale_bias,
                                                       dataset=opts.dataset)
            else:
                self.cam_predictor = TransformerCamera(nz_feat=hidden_enc_dim, num_heads=opts.num_heads,
                                                       num_encoder_layers=opts.num_encoder_layers,
                                                       num_decoder_layers=opts.num_decoder_layers,
                                                       num_cams=opts.num_hypo_cams, aze_ele_quat=opts.az_ele_quat,
                                                       scale_lr=opts.scale_lr_decay, scale_bias=opts.scale_bias,
                                                       dataset=opts.dataset)
        else:
            self.cam_predictor = Camera(nz_feat,)
        if opts.seg_train:
            self.vertex_part_predictor = VertexPartPredictor(num_verts=num_verts, num_parts=num_parts)

    def forward(self, feat, wide_img_feat):
        return_stuff = {}
        shape_pred = self.shape_predictor.forward(feat)
        return_stuff['shape_pred'] = shape_pred
        if self.multi_cam:
            cam_sampled, sample_inds, cam_probs, all_cameras, base_quats = self.cam_predictor.forward(wide_img_feat)
            cam = cam_sampled
            return_stuff['cam_hypotheses'] = all_cameras
            return_stuff['base_quats'] = base_quats[:, 0]
        else:
            cam = self.cam_predictor.forward(img_feat)  # quat (0:4), prop(4:5), scale(5:6), trans(6:8)
            cam = torch.cat([cam[:, 5:6], cam[:, 6:8], cam[:, 0:4]], dim=1)
            sample_inds = torch.zeros(cam[:, None, 0].shape).long().cuda()
            cam_probs = sample_inds.float() + 1

        return_stuff['cam_sample_inds'] = sample_inds
        return_stuff['cam_probs'] = cam_probs
        return_stuff['cam'] = cam
        if self.opts.seg_train:
            return_stuff['vertex_part_pred'] = self.vertex_part_predictor.forward(feat)
        return return_stuff


class PartBasisGenerator(torch.nn.Module):
    def __init__(self, feature_dim, K, normalize=False):
        super(PartBasisGenerator, self).__init__()
        self.w = torch.nn.Parameter(torch.abs(torch.cuda.FloatTensor(K, feature_dim).normal_()))
        self.normalize = normalize

    def forward(self, x=None):
        # NOTE: Put relu
        out = torch.nn.functional.relu(self.w)
        # out = self.w  # NOTE: rm this when putting relu
        if self.normalize:
            return featureL2Norm(out)
        else:
            return out


#------------ Mesh Net ------------#
#----------------------------------#


class MeshNet(nn.Module):
    def __init__(self, input_shape, opts, nz_feat=100, num_kps=15, sfm_mean_shape=None, num_parts=7, query_pos=None,
                 num_encoder_layers=6, num_decoder_layers=6, predict_flow=True, hidden_dim=None, hidden_enc_dim=256):
        # Input shape is H x W of the image.
        super(MeshNet, self).__init__()
        self.opts = opts
        self.pred_texture = opts.texture
        self.predict_flow = predict_flow
        self.symmetric = opts.symmetric
        self.symmetric_texture = opts.symmetric_texture

        # Mean shape.
        #verts, faces = mesh.create_sphere(opts.subdivide)
        mean_mesh = sr.Mesh.from_obj(opts.csm_bird_template_path)
        verts = mean_mesh.vertices[0].cpu().numpy()
        faces = mean_mesh.faces[0].cpu().numpy()

        self.num_sym_faces = 624  # dummy value for non symmetric meshes

        if self.symmetric:
            verts, faces, num_indept, num_sym, num_indept_faces, num_sym_faces = mesh.make_symmetric(verts, faces)
            if sfm_mean_shape is not None:
                verts = geom_utils.project_verts_on_mesh(verts, sfm_mean_shape[0], sfm_mean_shape[1])
            num_verts = verts.shape[0]
            num_sym_output = num_indept + num_sym
            if opts.only_mean_sym:
                print('Only the mean shape is symmetric!')
                self.num_output = num_verts
            else:
                self.num_output = num_sym_output
            self.num_sym = num_sym
            self.num_indept = num_indept
            self.num_indept_faces = num_indept_faces
            self.num_sym_faces = num_sym_faces
            # mean shape is only half.
            self.mean_v = nn.Parameter(torch.Tensor(verts[:num_sym_output]))

            # Needed for symmetrizing..
            self.flip = Variable(torch.ones(1, 3).cuda(), requires_grad=False)
            self.flip[0, 0] = -1
        else:
            if sfm_mean_shape is not None:
                verts = geom_utils.project_verts_on_mesh(verts, sfm_mean_shape[0], sfm_mean_shape[1])
            self.mean_v = nn.Parameter(torch.Tensor(verts))
            self.num_output = num_verts

        verts_np = verts
        faces_np = faces
        self.faces = Variable(torch.LongTensor(faces).cuda(), requires_grad=False)
        self.edges2verts = mesh.compute_edges2verts(verts, faces)

        vert2kp_init = torch.Tensor(np.ones((num_kps, num_verts)) / float(num_verts))
        # Remember initial vert2kp (after softmax)
        self.vert2kp_init = torch.nn.functional.softmax(Variable(vert2kp_init.cuda(), requires_grad=False), dim=1)
        self.vert2kp = nn.Parameter(vert2kp_init)

        self.encoder = Encoder(input_shape, n_blocks=4, nz_feat=nz_feat, hidden_enc_dim=hidden_enc_dim)
        # self.encoder = Encoder(input_shape, query_pos=query_pos, n_blocks=4, nz_feat=nz_feat, num_parts=num_parts,
        #                       num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, hidden_dim=hidden_dim)
        self.code_predictor = CodePredictor(nz_feat=nz_feat, num_verts=self.num_output, num_parts=num_parts,
                                            multi_cam=opts.multiple_cam_hypo, opts=opts, hidden_enc_dim=hidden_enc_dim)

        if self.pred_texture:
            if self.symmetric_texture:
                num_faces = self.num_indept_faces + self.num_sym_faces
            else:
                num_faces = faces.shape[0]

            uv_sampler = mesh.compute_uvsampler(verts_np, faces_np[:num_faces], tex_size=opts.tex_size)
            # F' x T x T x 2
            uv_sampler = Variable(torch.FloatTensor(uv_sampler).cuda(), requires_grad=False)
            # B x F' x T x T x 2
            uv_sampler = uv_sampler.unsqueeze(0).repeat(2 * self.opts.batch_size, 1, 1, 1, 1)
            img_H = int(2**np.floor(np.log2(np.sqrt(num_faces) * opts.tex_size)))
            img_W = 2 * img_H
            if self.predict_flow:
                self.texture_predictor = TexturePredictorUV(
                    nz_feat, uv_sampler, opts, img_H=img_H, img_W=img_W, predict_flow=predict_flow, symmetric=opts.symmetric_texture, num_sym_faces=self.num_sym_faces)
                nb.net_init(self.texture_predictor)
            else:
                self.texture_predictor_new = TexturePredictor(uv_sampler, opts)

    def forward(self, img):
        img_feat, resnet_img_feat = self.encoder.forward(img)
        resnet_img_feat = nn.functional.interpolate(resnet_img_feat, (16, 16), mode='bilinear', align_corners=True)
        codes_pred = self.code_predictor.forward(img_feat, resnet_img_feat)
        if self.pred_texture:
            if self.predict_flow:
                texture_pred = self.texture_predictor.forward(img_feat)
            else:
                texture_pred = self.texture_predictor_new.forward(wide_img_feat)
            return codes_pred, texture_pred
        else:
            return codes_pred

    def symmetrize(self, V):
        """
        Takes num_indept+num_sym verts and makes it
        num_indept + num_sym + num_sym
        Is identity if model is not symmetric
        """
        if self.symmetric:
            if V.dim() == 2:
                # No batch
                V_left = self.flip * V[-self.num_sym:]
                return torch.cat([V, V_left], 0)
            else:
                # With batch
                V_left = self.flip * V[:, -self.num_sym:]
                return torch.cat([V, V_left], 1)
        else:
            return V

    def get_mean_shape(self):
        return self.symmetrize(self.mean_v)

    def get_query_basis(self, normalize=True):
        w = self.code_predictor.cam_predictor.query_pos.weight
        if normalize:
            w = featureL2Norm(w)
        return w

    '''def get_params(self):
        params = list(self.parameters())
        for cam in self.code_predictor.cameras:
            params += list(cam.parameters())

        return params'''
