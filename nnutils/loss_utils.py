"""
Loss Utils.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from . import geom_utils
import numpy as np
from chamferdist.chamferdist import ChamferDistance
import itertools

NC2_perm = list(itertools.permutations(range(8), 2))
NC2_perm = torch.LongTensor(list(zip(*NC2_perm))).cuda()


def semantic_consistency_loss_per_instance(features, pred_softmax, basis):
    """
    Semantic Consistency loss

    Args:
        features: B x C x H x W
        pred_softmax: B x H x W x Np (part segmented image)
        basis: Np X C
    Returns:
        loss: tensor of 1 element
    """
    B, H, W, Np = pred_softmax.shape
    num_feats = features.shape[1]
    flat_part_softmax = pred_softmax.contiguous().view(-1, Np)  # BHW x Np
    flat_features = features.permute(0, 2, 3, 1).contiguous().view(-1, num_feats)  # BHW x C
    loss = torch.nn.functional.mse_loss(torch.mm(flat_part_softmax, basis), flat_features, reduction='none')
    return loss.view(B, H, W, num_feats).mean(-1).sum([1, 2])


def semantic_consistency_loss(features, pred_softmax, basis, probs=None, num_cams=1):
    if probs == None:
        probs = torch.ones(pred_softmax.shape[0]).view(num_cams, -1) / num_cams  # Nc x B
        probs = probs.cuda()
    loss = semantic_consistency_loss_per_instance(features, pred_softmax, basis).view(num_cams, -1)  # Nc x B
    return (loss * probs).sum(0).mean()


def neg_entropy(cam_probs):
    '''
    cam_probs: Nc x B
    '''
    return -1 * (-torch.log(cam_probs + 1E-9) * cam_probs).sum(0).mean()


def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) +
                     epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)


def orthonormal_loss(w):
    """
    Orthonormal loss

    Args:
        w: Np X C
    Returns:
        loss: tensor of 1 element
    """
    K, C = w.shape
    w_norm = featureL2Norm(w)
    WWT = torch.matmul(w_norm, w_norm.transpose(0, 1))

    return torch.nn.functional.mse_loss(WWT - torch.eye(K).cuda(), torch.zeros(K, K).cuda(), reduction='sum')  # size_average=False)


def chamfer_loss(verts_org, verts_trans):
    """
    Chamfer loss

    Args:
        verts_org: B x Nv x 3
        verts_trans: B x Nv x 3
     Returns:
        loss: tensor of 1 element
    """
    chamferDist = ChamferDistance()
    dist1, dist2, idx1, idx2 = chamferDist(verts_org, verts_trans)
    loss = dist2
    return loss.mean()


def invariance_loss(R_input_softmax, R_target_softmax):
    """
    Invariance loss using KL Divergence

    Args:
        R_input: B x Nv x Np
        R_target: B x Nv x Np
    Returns:
        loss: tensor of 1 element
    """
    B, Nv, Np = R_input_softmax.shape
    loss = torch.nn.functional.kl_div(torch.log(R_input_softmax), R_target_softmax, reduction='batchmean')
    return loss / Nv


def batch_get_centers(R_pred_softmax, verts):
    """
    Gets part centroids of a batch

    Args:
        R_pred_softmax: B x Nv x Np
        verts: B x Nv x 3
    Returns:
        centroids: B x Np x 3
    """
    B, Nv, Np = R_pred_softmax.shape
    z = torch.sum(R_pred_softmax, 1).unsqueeze(2)  # (B x Nv x Np) --> (B x Np) --> (B x Np x 1)
    centroids = torch.bmm(R_pred_softmax.permute(0, 2, 1).contiguous(), verts)  # (B x Np x 3)
    return centroids / z


def concentration_loss(R_pred_softmax, verts):
    """
    Concentration Loss

    Args:
        R_pred_softmax: B x Nv x Np
        verts: B x Nv x 3
    Returns:
        loss: tensor of 1 element
    """
    B, Nv, Np = R_pred_softmax.shape

    loss = 0
    centers_all = batch_get_centers(R_pred_softmax, verts)  # (B x Np x 3)

    for k in range(Np):
        R_pred_k = R_pred_softmax[:, :, k:k + 1]  # (B x Nv x 1)
        z = torch.sum(R_pred_k, 1, keepdim=True)  # (B x 1 x 1)

        centres_k = centers_all[:, k:k + 1, :]  # (B x 1 x 3)
        norms = (verts - centres_k).pow(2)  # (B x Nv x 3)

        loss_per_part = torch.bmm(R_pred_k.permute(0, 2, 1).contiguous(), norms)  # (B x 1 x 3)
        loss_per_part = loss_per_part / z
        loss_per_part = loss_per_part.sum()  # (B x 1 x 3) --> loss per part for full batch
        loss = loss + loss_per_part

    return loss / B


def neg_iou_loss_per_instance(mask_pred, mask):
    """
    mask: B x H x W (binary mask)
    mask_pred: B x H x W (SoftRas predicted mask)

    Computes IoU loss over the masks
    """
    intersection = mask * mask_pred
    union = mask + mask_pred - intersection
    iou = intersection.sum(dim=(-1, -2)) / union.sum(dim=(-1, -2))
    return 1. - iou


def neg_iou_loss(mask_pred, mask, probs=None, num_cams=1):
    """
    mask: B x H x W (binary mask)
    mask_pred: B x H x W (SoftRas predicted mask)
    probs: Nc x B (Probability of each of `num_cams` camera)
    num_cams: number of cameras (Nc)

    Computes IoU loss over the masks
    """
    if probs == None:
        probs = torch.ones(mask.shape[0]).view(num_cams, -1) / num_cams  # Nc x B
        probs = probs.cuda()
    loss = neg_iou_loss_per_instance(mask_pred, mask).view(num_cams, -1)  # Nc x B
    return (loss * probs).sum(0).mean()


def mask_l2_loss_per_instance(mask_pred, mask):
    mask_pred = mask_pred.view(mask_pred.shape[0], -1)
    mask = mask.view(mask.shape[0], -1)
    mask_loss = (mask_pred - mask)
    mask_loss = (mask_loss * mask_loss).mean(-1)
    return mask_loss


def mask_l2_loss(mask_pred, mask, probs=None, num_cams=1):
    """
    mask: B x H x W (binary mask)
    mask_pred: B x H x W (SoftRas predicted mask)
    probs: Nc x B (Probability of each of `num_cams` camera)
    num_cams: number of cameras (Nc)

    Computes l2 loss over the masks
    """
    if probs == None:
        probs = torch.ones(mask.shape[0]).view(num_cams, -1) / num_cams  # Nc x B
        probs = probs.cuda()
    loss = mask_l2_loss_per_instance(mask_pred, mask).view(num_cams, -1)  # Nc x B
    return (loss * probs).sum(0).mean()


def mask_dt_loss_per_instance(proj_verts, dist_transf):
    """
    proj_verts: B x N x 2
    (In normalized coordinate [-1, 1])
    dist_transf: B x 1 x N x N

    Computes the distance transform at the points where vertices land.
    """
    # Reshape into B x 1 x N x 2
    sample_grid = proj_verts.unsqueeze(1)
    # B x 1 x 1 x N
    dist_transf = torch.nn.functional.grid_sample(dist_transf, sample_grid, padding_mode='border', align_corners=True)
    return dist_transf.mean([1, 2, 3])


def mask_dt_loss(proj_verts, dist_transf, probs=None, num_cams=1):
    """
    proj_verts: Nc*B x N x 2
    (In normalized coordinate [-1, 1])
    dist_transf: Nc*B x 1 x N x N
    probs: Nc x B (Probability of each of `num_cams` camera)
    num_cams: number of cameras (Nc)
    Computes the distance transform at the points where vertices land.
    """
    if probs == None:
        probs = torch.ones(proj_verts.shape[0]).view(num_cams, -1) / num_cams  # Nc x B
        probs = probs.cuda()
    loss = mask_dt_loss_per_instance(proj_verts, dist_transf).view(num_cams, -1)  # Nc x B
    return (loss * probs).sum(0).mean()


def cam_select_crit(mask_pred, mask, texture_pred, imgs, tex_loss):  # proj_verts, dist_transf):
    tl = tex_loss(texture_pred, imgs, mask_pred, mask, return_scalar=False)
    ml = mask_l2_loss_per_instance(mask_pred, mask)
    return 10 * ml + tl, ml, tl


def texture_dt_loss_per_instance(texture_flow, dist_transf, vis_rend=None, cams=None, verts=None, tex_pred=None):
    """
    texture_flow: B x F x T x T x 2
    (In normalized coordinate [-1, 1])
    dist_transf: B x 1 x N x N

    Similar to geom_utils.sample_textures
    But instead of sampling image, it samples dt values.
    """
    # Reshape into B x F x T*T x 2
    T = texture_flow.size(-2)
    F = texture_flow.size(1)
    flow_grid = texture_flow.view(-1, F, T * T, 2)
    # B x 1 x F x T*T
    dist_transf = torch.nn.functional.grid_sample(dist_transf, flow_grid, align_corners=True)

    if vis_rend is not None:
        # Visualize the error!
        # B x 3 x F x T*T
        dts = dist_transf.repeat(1, 3, 1, 1)
        # B x 3 x F x T x T
        dts = dts.view(-1, 3, F, T, T)
        # B x F x T x T x 3
        dts = dts.permute(0, 2, 3, 4, 1)
        dts = dts.unsqueeze(4).repeat(1, 1, 1, 1, T, 1) / dts.max()

        from utils import bird_vis
        for i in range(dist_transf.size(0)):
            rend_dt = vis_rend(verts[i], cams[i], dts[i])
            rend_img = bird_vis.tensor2im(tex_pred[i].data)
            import matplotlib.pyplot as plt
            plt.ion()
            fig = plt.figure(1)
            plt.clf()
            ax = fig.add_subplot(121)
            ax.imshow(rend_dt)
            ax = fig.add_subplot(122)
            ax.imshow(rend_img)
            import ipdb
            ipdb.set_trace()

    return dist_transf.mean([1, 2, 3])


def texture_dt_loss(texture_flow, dist_transf, vis_rend=None, cams=None, verts=None, tex_pred=None, probs=None, num_cams=1):
    """
    texture_flow: B x F x T x T x 2
    (In normalized coordinate [-1, 1])
    dist_transf: B x 1 x N x N
    probs: Nc x B (probability of camera)
    num_cams: number of cameras (Nc)

    Similar to geom_utils.sample_textures
    But instead of sampling image, it samples dt values.
    """
    if probs == None:
        probs = torch.ones(texture_flow.shape[0]).view(num_cams, -1) / num_cams  # Nc x B
        probs = probs.cuda()
    loss = texture_dt_loss_per_instance(texture_flow, dist_transf, vis_rend, cams, verts, tex_pred).view(num_cams, -1)  # Nc x B
    return (loss * probs).sum(0).mean()


def texture_loss_per_instance(img_pred, img_gt, mask_pred, mask_gt):
    """
    Input:
      img_pred, img_gt: B x 3 x H x W
      mask_pred, mask_gt: B x H x W
    """
    mask_pred = mask_pred.unsqueeze(1)
    mask_gt = mask_gt.unsqueeze(1)

    return torch.nn.functional_l1loss(img_pred * mask_pred, img_gt * mask_gt, reduction='none')


def texture_loss(img_pred, img_gt, mask_pred, mask_gt, probs=None, num_cams=1):
    """
    Input:
      img_pred, img_gt: Nc*B x 3 x H x W
      mask_pred, mask_gt: Nc*B x H x W
      probs: Probability of each of `num_cams` camera
      num_cams: number of cameras (Nc)
    """
    if probs == None:
        probs = torch.ones(img_pred.shape[0]).view(num_cams, -1) / num_cams  # Nc x B
        probs = probs.cuda()
    loss = texture_loss_per_instance(img_pred, img_gt, mask_pred, mask_gt).mean([1, 2, 3]).view(num_cams, -1)  # Nc x B
    return (loss * probs).sum(0).mean()


def camera_loss(cam_pred, cam_gt, margin):
    """
    cam_* are B x 7, [sc, tx, ty, quat]
    Losses are in similar magnitude so one margin is ok.
    """
    rot_pred = cam_pred[:, -4:]
    rot_gt = cam_gt[:, -4:]

    rot_loss = hinge_loss(quat_loss_geodesic(rot_pred, rot_gt), margin)
    # Scale and trans.
    st_loss = (cam_pred[:, :3] - cam_gt[:, :3])**2
    st_loss = hinge_loss(st_loss.view(-1), margin)

    return rot_loss.mean() + st_loss.mean()


def cam_ce_loss(probs, cam_ids):
    # probs: Nc x B
    # cam_ids: B
    return -torch.log(probs[cam_ids, torch.arange(probs.shape[1])]).mean()


def rotation_reg(all_cams, num_cams):
    quats = all_cams[:, :, 3:7]
    quats_x = torch.gather(quats, dim=1, index=NC2_perm[0].view(1, -1, 1).repeat(len(quats), 1, 4))
    quats_y = torch.gather(quats, dim=1, index=NC2_perm[1].view(1, -1, 1).repeat(len(quats), 1, 4))
    inter_quats = geom_utils.hamilton_product(quats_x, geom_utils.quat_conj(quats_y))
    quatAng = geom_utils.quat2ang(inter_quats).view(len(inter_quats), num_cams - 1, -1)
    quatAng = -1 * torch.nn.functional.max_pool1d(-1 * quatAng.permute(0, 2, 1), num_cams - 1, stride=1).squeeze()
    rot_mag = (np.pi - quatAng).mean()

    return rot_mag


def hinge_loss(loss, margin):
    # Only penalize if loss > margin
    zeros = torch.autograd.Variable(torch.zeros(1).cuda(), requires_grad=False)
    return torch.max(loss - margin, zeros)


def quat_loss_geodesic(q1, q2):
    '''
    Geodesic rotation loss.

    Args:
        q1: N X 4
        q2: N X 4
    Returns:
        loss : N x 1
    '''
    q1 = torch.unsqueeze(q1, 1)
    q2 = torch.unsqueeze(q2, 1)
    q2_conj = torch.cat([q2[:, :, [0]], -1 * q2[:, :, 1:4]], dim=-1)
    q_rel = geom_utils.hamilton_product(q1, q2_conj)
    q_loss = 1 - torch.abs(q_rel[:, :, 0])
    # we can also return q_loss*q_loss
    return q_loss


def quat_loss(q1, q2):
    '''
    Anti-podal squared L2 loss.

    Args:
        q1: N X 4
        q2: N X 4
    Returns:
        loss : N x 1
    '''
    q_diff_loss = (q1 - q2).pow(2).sum(1)
    q_sum_loss = (q1 + q2).pow(2).sum(1)
    q_loss, _ = torch.stack((q_diff_loss, q_sum_loss), dim=1).min(1)
    return q_loss


def triangle_loss(verts, edge2verts):
    """
    Encourages dihedral angle to be 180 degrees.

    Args:
        verts: B X N X 3
        edge2verts: B X E X 4
    Returns:
        loss : scalar
    """
    indices_repeat = torch.stack([edge2verts, edge2verts, edge2verts], dim=2)  # B X E X 3 X 4

    verts_A = torch.gather(verts, 1, indices_repeat[:, :, :, 0])
    verts_B = torch.gather(verts, 1, indices_repeat[:, :, :, 1])
    verts_C = torch.gather(verts, 1, indices_repeat[:, :, :, 2])
    verts_D = torch.gather(verts, 1, indices_repeat[:, :, :, 3])

    # n1 = cross(ad, ab)
    # n2 = cross(ab, ac)
    n1 = geom_utils.cross_product(verts_D - verts_A, verts_B - verts_A)
    n2 = geom_utils.cross_product(verts_B - verts_A, verts_C - verts_A)

    n1 = torch.nn.functional.normalize(n1, dim=2)
    n2 = torch.nn.functional.normalize(n2, dim=2)

    dot_p = (n1 * n2).sum(2)
    loss = ((1 - dot_p)**2).mean()
    return loss


def deform_l2reg(V):
    """
    l2 norm on V = B x N x 3
    """
    V = V.view(-1, V.size(2))
    return torch.mean(torch.norm(V, p=2, dim=1))


def entropy_loss(A):
    """
    Input is K x N
    Each column is a prob of vertices being the one for k-th keypoint.
    We want this to be sparse = low entropy.
    """
    entropy = -torch.sum(A * torch.log(A), 1)
    # Return avg entropy over
    return torch.mean(entropy)


def kp_l2_loss(kp_pred, kp_gt):
    """
    L2 loss between visible keypoints.

    Sum_i [0.5 * vis[i] * (kp_gt[i] - kp_pred[i])^2] / (|vis|)
    """
    criterion = torch.nn.MSELoss()

    vis = (kp_gt[:, :, 2, None] > 0).float()

    # This always has to be (output, target), not (target, output)
    return criterion(vis * kp_pred, vis * kp_gt[:, :, :2])


def lsgan_loss(score_real, score_fake):
    """
    DELETE ME.
    Label 0=fake, 1=real.
    score_real is B x 1, score for real samples
    score_fake is B x 1, score for fake samples

    Returns loss for discriminator and encoder.
    """

    disc_loss_real = torch.mean((score_real - 1)**2)
    disc_loss_fake = torch.mean((score_fake)**2)
    disc_loss = disc_loss_real + disc_loss_fake

    enc_loss = torch.mean((score_fake - 1)**2)

    return disc_loss, enc_loss


class EdgeLoss(object):
    """
    Edge length should not diverge from the original edge length.

    On initialization computes the current edge lengths.
    """

    def __init__(self, verts, edges2verts, margin=2, use_bad_edge=False, use_l2=False):
        # Input:
        #  verts: B x N x 3
        #  edeges2verts: B x E x 4
        #  (only using the first 2 columns)
        self.use_l2 = use_l2

        # B x E x 2
        edge_verts = edges2verts[:, :, :2]
        self.indices = torch.stack([edge_verts, edge_verts, edge_verts], dim=2)
        V_copy = torch.autograd.Variable(verts.data, requires_grad=False)
        if V_copy.dim() == 2:
            # N x 3 (mean shape) -> B x N x 3
            V_copy = V_copy.unsqueeze(0).repeat(edges2verts.size(0), 1, 1)

        if use_bad_edge:
            self.log_e0 = torch.log(self.compute_edgelength(V_copy))
        else:
            # e0 is the mean over all edge lengths!
            e0 = self.compute_edgelength(V_copy).mean(1).view(-1, 1)
            self.log_e0 = torch.log(e0)

        self.margin = np.log(margin)
        self.zeros = torch.autograd.Variable(torch.zeros(1).cuda(), requires_grad=False)

        # For visualization
        self.v1 = edges2verts[0, :, 0].data.cpu().numpy()
        self.v2 = edges2verts[0, :, 1].data.cpu().numpy()

    def __call__(self, verts):
        e1 = self.compute_edgelength(verts)
        if self.use_l2:
            dist = (torch.log(e1) - self.log_e0)**2
            self.dist = torch.max(dist - self.margin**2, self.zeros)
        else:
            dist = torch.abs(torch.log(e1) - self.log_e0)
            self.dist = torch.max(dist - self.margin, self.zeros)
        return self.dist.mean()

    def compute_edgelength(self, V):
        v1 = torch.gather(V, 1, self.indices[:, :, :, 0])
        v2 = torch.gather(V, 1, self.indices[:, :, :, 1])

        elengths = torch.sqrt(((v1 - v2)**2).sum(2))

        # B x E
        return elengths

    def visualize(self, verts, F_np, mv=None):
        from psbody.mesh import Mesh

        V = verts[0].data.cpu().numpy()
        mesh = Mesh(V, F_np)
        dist = self.dist[0].data.cpu().numpy()

        v_weights = np.zeros((V.shape[0]))
        for e_id, (v1_id, v2_id) in enumerate(zip(self.v1, self.v2)):
            v_weights[v1_id] += dist[e_id]
            v_weights[v2_id] += dist[e_id]

        mesh.set_vertex_colors_from_weights(v_weights)

        if mv is not None:
            mv.set_dynamic_meshes([mesh])
        else:
            mesh.show()
            import ipdb
            ipdb.set_trace()


class LaplacianLoss(object):
    """
    Encourages minimal mean curvature shapes.
    """

    def __init__(self, faces):
        # Input:
        #  faces: B x F x 3
        from nnutils.laplacian import LaplacianModule
        # V x V
        self.laplacian = LaplacianModule(faces)
        self.Lx = None

    def __call__(self, verts):
        self.Lx = self.laplacian(verts)
        # Reshape to BV x 3
        Lx = self.Lx.view(-1, self.Lx.size(2))
        loss = torch.norm(Lx, p=2, dim=1).mean()
        return loss

    def visualize(self, verts, mv=None):
        # Visualizes the laplacian.
        # Verts is B x N x 3 Variable
        Lx = self.Lx[0].data.cpu().numpy()

        V = verts[0].data.cpu().numpy()

        from psbody.mesh import Mesh
        F = self.laplacian.F_np[0]
        mesh = Mesh(V, F)

        weights = np.linalg.norm(Lx, axis=1)
        mesh.set_vertex_colors_from_weights(weights)

        if mv is not None:
            mv.set_dynamic_meshes([mesh])
        else:
            mesh.show()
            import ipdb
            ipdb.set_trace()


class PerceptualTextureLoss(object):
    def __init__(self):
        from nnutils.perceptual_loss import PerceptualLoss
        self.perceptual_loss = PerceptualLoss()

    def __call__(self, img_pred, img_gt, mask_pred, mask_gt, probs=None, num_cams=1, return_scalar=True):
        """
        Input:
          img_pred, img_gt: B x 3 x H x W
        mask_pred, mask_gt: B x H x W
        """
        mask_pred = mask_pred.unsqueeze(1)
        mask_gt = mask_gt.unsqueeze(1)

        # Only use mask_gt..

        dist = self.perceptual_loss(img_pred * mask_gt, img_gt * mask_gt)  # Nc*B
        if not return_scalar:
            return dist
        dist = dist.view(num_cams, -1)
        if probs == None:
            probs = torch.ones(img_pred.shape[0]).view(num_cams, -1) / num_cams  # Nc x B
            probs = probs.cuda()
        return (dist * probs).sum(0).mean()
