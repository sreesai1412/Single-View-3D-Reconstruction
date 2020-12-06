"""
Script for testing on CUB.

Sample usage:
python -m benchmark.evaluate --split val --name <model_name> --num_train_epoch <model_epoch>
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
import scipy.io as sio
import cv2
from matplotlib import pyplot as plt

from nnutils import test_utils
from data import cub as cub_data
from nnutils import predictor as pred_utils
from utils.bird_vis import convert2np
from nnutils import geom_utils

flags.DEFINE_boolean('visualize', False, 'if true visualizes things')

opts = flags.FLAGS


def make_heat_map(kps, H=256, W=256, sigma=3.):
    B, numkp = kps.shape[:2]
    x = torch.arange(W)
    y = torch.arange(H)
    grid_x, grid_y = torch.meshgrid(x, y)
    hms_grid = torch.stack([grid_y, grid_x], dim=0)
    hms_grid = hms_grid.view(1, 1, *hms_grid.shape)  # 1 x 1 x 2 x H x W
    hms_grid = hms_grid.repeat(B, numkp, 1, 1, 1).cuda()  # B x numkp x 2 x H x W

    hms = torch.zeros((B, numkp, H, W)).cuda()

    kp = (kps[:, :, :2] + 1) * 0.5 * W
    kp = torch.round(kp).long().cuda()

    for i in range(numkp):
        mu = kp[:, i].view(B, 2, 1, 1)
        norm = (hms_grid[:, i] - mu) / sigma
        hms[:, i] = (norm * norm).sum(1) / 2.

    hms = -hms.view(B, numkp, -1)
    hms = torch.nn.functional.softmax(hms, dim=-1)
    return hms.view(B, numkp, H, W)


class ShapeTester(test_utils.Tester):
    def define_model(self, kt=False):
        opts = self.opts

        self.predictor = pred_utils.MeshPredictor(opts)

        # for visualization
        self.renderer = self.predictor.vis_rend
        self.renderer.set_bgcolor([1., 1., 1.])
        self.renderer.renderer.renderer.image_size = 512
        self.renderer.set_light_dir([0, 1, -1], 0.38)
        self.num_figs = 0

    def init_dataset(self, kt=False):
        opts = self.opts
        self.data_module = cub_data

        opts = self.opts
        torch.manual_seed(0)
        if kt:
            self.dl_img1 = self.data_module.cub_test_pair_dataloader(opts, 1)
            self.dl_img2 = self.data_module.cub_test_pair_dataloader(opts, 2)
        else:
            self.dataloader = self.data_module.data_loader(opts)  # if not kt else self.data_module.cub_test_pair_dataloader(opts)

    def compute_pck(self, kps_pred, kps_gt):
        # Compute pck
        padding_frac = opts.padding_frac
        # The [-1,1] coordinate frame in which keypoints corresponds to:
        #    (1+2*padding_frac)*max_bbox_dim in image coords
        # pt_norm = 2* (pt_img - trans)/((1+2*pf)*max_bbox_dim)
        # err_pt = 2*err_img/((1+2*pf)*max_bbox_dim)
        # err_pck_norm = err_img/max_bbox_dim = err_pt*(1+2*pf)/2
        # so the keypoint error in the canonical fram should be multiplied by:
        err_scaling = (1 + 2 * padding_frac) / 2.0
        kps_vis = kps_gt[:, :, 2]
        kps_gt = kps_gt[:, :, 0:2]
        kps_err = kps_pred - kps_gt
        kps_err = np.sqrt(np.sum(kps_err * kps_err, axis=2)) * err_scaling

        return kps_err, kps_vis

    def keypoint_transfer_tex(self, kp_gt_source, kp_gt_tgt, tex_flow_source, tex_flow_tgt):
        # make heat map for every keypoint
        kp_gt_source_hm = make_heat_map(kp_gt_source)

        # sample from heatmap using predicted flow and take channelwise mean for each face.
        B, numkp = kp_gt_source_hm.shape[:2]
        F, T = tex_flow_source.shape[1:3]
        omega = geom_utils.sample_textures(tex_flow_source, kp_gt_source_hm).view(B, F, T, T, numkp)
        omega = omega.mean([-2, -3])  # B x F x numkp

        # then argmax to find face of every kp
        kp_face = omega.argmax(1).long()  # B x numkp

        # find locations that fall on face corresponding to each kp. take mean to find kp location in tgt
        xx = torch.arange(B).unsqueeze(-1).repeat(1, numkp).view(-1).long()
        kp_pred_tgt = tex_flow_tgt[xx, kp_face.view(-1)]
        kp_pred_tgt = kp_pred_tgt.view(B, numkp, T, T, 2).mean([-2, -3])  # B x numkp x 2

        kp_pred_tgt = kp_pred_tgt.detach().cpu().type_as(kp_gt_tgt).numpy()
        kp_gt_tgt = kp_gt_tgt.cpu().numpy()
        kps_err12, kps_vis12 = self.compute_pck(kp_pred_tgt, kp_gt_tgt)

        return kps_err12, kps_vis12

    def keypoint_transfer_cam(self, kp_gt_source, kp_gt_tgt, verts_source, verts_tgt, cam_pred_source, cam_pred_tgt):
        # Project vertices
        proj_verts_source = self.predictor.renderer.project_points(verts_source, cam_pred_source)  # B x Nv x 2
        B, numkp = kp_gt_source.shape[:2]
        kp_verts = []

        kp_gt_source = kp_gt_source.cuda()

        # Finding vertices closest to keypoints
        for i in range(numkp):
            err = (proj_verts_source - kp_gt_source[:, i:i + 1, :2])
            err = (err * err).sum(-1).sqrt()  # B x Nv
            kp_verts.append(err.argmin(1))

        kp_verts = torch.stack(kp_verts, 1)  # B x numkp

        # Finding predicted keypoints using vertices
        proj_verts_tgt = self.predictor.renderer.project_points(verts_tgt, cam_pred_tgt)  # B x Nv x 2
        batch_idxs = torch.arange(B).view(B, 1).repeat(1, numkp).view(-1)
        kp_pred_tgt = proj_verts_tgt[batch_idxs, kp_verts.view(-1)].view(B, numkp, 2)

        kp_pred_tgt = kp_pred_tgt.detach().cpu().type_as(kp_gt_tgt).numpy()
        kp_gt_tgt = kp_gt_tgt.detach().cpu().numpy()
        kps_err12, kps_vis12 = self.compute_pck(kp_pred_tgt, kp_gt_tgt)

        return kps_err12, kps_vis12

    def evaluate(self, outputs, batch):
        """
        Compute IOU and keypoint error
        """
        opts = self.opts
        bs = opts.batch_size

        # compute iou
        mask_gt = batch['mask'].view(bs, -1).numpy()
        mask_pred = outputs['mask_pred'].cpu().view(bs, -1).type_as(
            batch['mask']).numpy()
        intersection = mask_gt * mask_pred
        union = mask_gt + mask_pred - intersection
        iou = intersection.sum(1) / union.sum(1)

        kps_err, kps_vis = self.compute_pck(outputs['kp_pred'].cpu().type_as(batch['kp']).numpy(),
                                            batch['kp'].cpu().numpy())

        return iou, kps_err, kps_vis

    def evaluate_kt(self, outputs, batch):
        """
        Compute keypoint transfer error
        """
        opts = self.opts
        bs = opts.batch_size

        kp_gt = batch['kp']  # .cpu().numpy()
        kp_gt_1, kp_gt_2 = kp_gt.chunk(2)
        if 'tex_flow' in outputs.keys():
            tex_flow_1, tex_flow_2 = outputs['tex_flow'].chunk(2)  # .cpu().numpy()
        pred_cam_1, pred_cam_2 = outputs['cam_pred'].chunk(2)
        verts_1, verts_2 = outputs['verts'].chunk(2)

        # Keypoint transfer tex
        if 'tex_flow' in outputs.keys():
            kp_err12, kp_vis12 = self.keypoint_transfer_tex(kp_gt_1, kp_gt_2, tex_flow_1, tex_flow_2)
            kp_err21, kp_vis21 = self.keypoint_transfer_tex(kp_gt_2, kp_gt_1, tex_flow_2, tex_flow_1)

            kps_err_tex = np.concatenate([kp_err12, kp_err21])
            kps_vis_tex = np.concatenate([kp_vis12, kp_vis21])
        else:
            kps_err_tex, kps_vis_tex = None, None

        # Keypoint transfer camera
        kp_err12, kp_vis_cam12 = self.keypoint_transfer_cam(kp_gt_1, kp_gt_2, verts_1, verts_2, pred_cam_1, pred_cam_2)
        kp_err21, kp_vis_cam21 = self.keypoint_transfer_cam(kp_gt_2, kp_gt_1, verts_2, verts_1, pred_cam_2, pred_cam_1)

        kps_err_cam = np.concatenate([kp_err12, kp_err21])
        kps_vis_cam = np.concatenate([kp_vis_cam12, kp_vis_cam21])

        return kps_err_tex, kps_err_cam, kps_vis_cam

    def visualize(self, outputs, batch):
        self.num_figs += 1
        vert = outputs['verts'][0]
        cam = outputs['cam_pred'][0]
        texture = outputs['texture'][0]

        img_pred = self.renderer(vert, cam, texture=texture)
        aroundz = []
        aroundy = []
        # for deg in np.arange(0, 180, 30):
        for deg in np.arange(0, 150, 30):
            rendz = self.renderer.diff_vp(
                vert, cam, angle=-deg, axis=[1, 0, 0], texture=texture)
            rendy = self.renderer.diff_vp(
                vert, cam, angle=deg, axis=[0, 1, 0], texture=texture)
            aroundz.append(rendz)
            aroundy.append(rendy)

        aroundz = np.hstack(aroundz)
        aroundy = np.hstack(aroundy)
        vps = np.vstack((aroundz, aroundy))

        img = np.transpose(convert2np(batch['img'][0]), (1, 2, 0))
        import matplotlib.pyplot as plt
        plt.ion()
        fig = plt.figure(1)
        ax = fig.add_subplot(121)
        ax.imshow(img)
        ax.set_title('input')
        ax.axis('off')
        ax = fig.add_subplot(122)
        ax.imshow(img_pred)
        ax.set_title('pred_texture')
        ax.axis('off')
        plt.draw()

        fig = plt.figure(2)
        plt.imshow(vps)
        plt.axis('off')
        plt.draw()

    def get_azele_dist(self, outputs, batch):
        gt_cam = batch['sfm_pose']
        pred_cam = outputs['cam_pred']
        gt_az, gt_ele = gt_cam[:, 4].cpu().numpy(), gt_cam[:, 5].cpu().numpy()
        pred_az, pred_ele = pred_cam[:, 4].cpu().numpy(), pred_cam[:, 5].cpu().numpy()
        return gt_az, gt_ele, pred_az, pred_ele

    def save_azele_dist_graph(self, azele_gt, azele_pred):
        fig = plt.figure()
        plt.scatter(azele_gt[0], azele_gt[1])
        plt.xlabel('Elevation')
        plt.ylabel('Azimuth')
        plt.savefig('AzEle_gt.png')

        fig = plt.figure()
        plt.scatter(azele_pred[0], azele_pred[1])
        plt.xlabel('Elevation')
        plt.ylabel('Azimuth')
        plt.savefig('AzEle_pred.png')

    def test(self):
        opts = self.opts
        bench_stats = {'ious': [], 'kp_errs': [], 'kp_vis': []}

        if opts.ignore_pred_delta_v:
            result_path = osp.join(opts.results_dir, 'results_meanshape.mat')
        elif opts.use_sfm_ms:
            result_path = osp.join(opts.results_dir,
                                   'results_sfm_meanshape.mat')
        else:
            result_path = osp.join(opts.results_dir, 'results.mat')

        if opts.use_sfm_camera:
            result_path = result_path.replace('.mat', '_sfm_camera.mat')

        print('Writing to %s' % result_path)

        azele_gt, azele_pred = [[], []], [[], []]
        if not osp.exists(result_path):

            n_iter = len(self.dataloader)
            for i, batch in enumerate(self.dataloader):
                if i % 100 == 0:
                    print('{}/{} evaluation iterations.'.format(i, n_iter))
                if opts.max_eval_iter > 0 and (i >= opts.max_eval_iter):
                    break
                outputs = self.predictor.predict(batch)
                if opts.visualize:
                    self.visualize(outputs, batch)
                iou, kp_err, kp_vis = self.evaluate(outputs, batch)
                gt_az, gt_ele, pred_az, pred_ele = self.get_azele_dist(outputs, batch)
                azele_gt[0].append(gt_az)
                azele_gt[1].append(gt_ele)
                azele_pred[0].append(pred_az)
                azele_pred[1].append(pred_ele)
                bench_stats['ious'].append(iou)
                bench_stats['kp_errs'].append(kp_err)
                bench_stats['kp_vis'].append(kp_vis)

                if opts.save_visuals and (i % opts.visuals_freq == 0):
                    self.save_current_visuals(batch, outputs)

            bench_stats['kp_errs'] = np.concatenate(bench_stats['kp_errs'])
            bench_stats['kp_vis'] = np.concatenate(bench_stats['kp_vis'])

            bench_stats['ious'] = np.concatenate(bench_stats['ious'])
            sio.savemat(result_path, bench_stats)
        else:
            bench_stats = sio.loadmat(result_path)

        mean_iou = bench_stats['ious'].mean()

        n_vis_p = np.sum(bench_stats['kp_vis'], axis=0)
        n_correct_p_pt1 = np.sum(
            (bench_stats['kp_errs'] < 0.1) * bench_stats['kp_vis'], axis=0)
        n_correct_p_pt15 = np.sum(
            (bench_stats['kp_errs'] < 0.15) * bench_stats['kp_vis'], axis=0)
        pck1 = (n_correct_p_pt1 / n_vis_p).mean()
        pck15 = (n_correct_p_pt15 / n_vis_p).mean()
        print('%s mean iou %.3g, pck.1 %.3g, pck.15 %.3g' %
              (osp.basename(result_path), mean_iou, pck1, pck15))

    def test_kt(self):
        opts = self.opts
        bench_stats = {'kp_errs_tex': [], 'kp_errs_cam': [], 'kp_vis': []}

        if opts.ignore_pred_delta_v:
            result_path = osp.join(opts.results_dir, 'results_meanshape_kt.mat')
        elif opts.use_sfm_ms:
            result_path = osp.join(opts.results_dir,
                                   'results_sfm_meanshape_kt.mat')
        else:
            result_path = osp.join(opts.results_dir, 'results_kt.mat')

        if opts.use_sfm_camera:
            result_path = result_path.replace('.mat', '_sfm_camera_kt.mat')

        print('Writing to %s' % result_path)

        if not osp.exists(result_path):

            n_iter = len(self.dl_img1)
            for i, batch in enumerate(zip(self.dl_img1, self.dl_img2)):
                self.iter_index = i
                if i % 100 == 0:
                    print('{}/{} evaluation iterations.'.format(i, n_iter))
                if opts.max_eval_iter > 0 and (i >= opts.max_eval_iter):
                    break
                batch = {key: torch.cat([batch[i][key] for i in range(len(batch))])
                         for key in batch[0].keys() if key in ['img', 'kp', 'sfm_pose', 'mask']}
                outputs = self.predictor.predict(batch)
                if opts.visualize:
                    self.visualize(outputs, batch)
                kp_err_tex, kp_err_cam, kp_vis = self.evaluate_kt(outputs, batch)

                if kp_err_tex is not None:
                    bench_stats['kp_errs_tex'].append(kp_err_tex)
                bench_stats['kp_errs_cam'].append(kp_err_cam)
                bench_stats['kp_vis'].append(kp_vis)

                if opts.save_visuals and (i % opts.visuals_freq == 0):
                    self.save_current_visuals(batch, outputs)

            if len(bench_stats['kp_errs_tex']) > 0:
                bench_stats['kp_errs_tex'] = np.concatenate(bench_stats['kp_errs_tex'])
            bench_stats['kp_errs_cam'] = np.concatenate(bench_stats['kp_errs_cam'])
            bench_stats['kp_vis'] = np.concatenate(bench_stats['kp_vis'])

            #bench_stats['ious'] = np.concatenate(bench_stats['ious'])
            sio.savemat(result_path, bench_stats)
        else:
            bench_stats = sio.loadmat(result_path)

        # Report numbers for tex.
        if len(bench_stats['kp_errs_tex']) > 0:
            report_numbers(bench_stats, 'tex', result_path)
        report_numbers(bench_stats, 'cam', result_path)


def report_numbers(bench_stats, kt_type, result_path):
    n_vis_p = np.sum(bench_stats['kp_vis'], axis=0)
    n_correct_p_pt1 = np.sum(
        (bench_stats['kp_errs_%s' % kt_type] < 0.1) * bench_stats['kp_vis'], axis=0)
    n_correct_p_pt15 = np.sum(
        (bench_stats['kp_errs_%s' % kt_type] < 0.15) * bench_stats['kp_vis'], axis=0)
    pck1 = (n_correct_p_pt1 / n_vis_p).mean()
    pck15 = (n_correct_p_pt15 / n_vis_p).mean()
    print('Keypoint transfer %s: %s kt pck.1 %.3g, kt pck.15 %.3g' %
          (kt_type, osp.basename(result_path), pck1, pck15))


def main(_):
    opts.n_data_workers = 0
    opts.batch_size = 8

    opts.results_dir = osp.join(opts.results_dir_base, '%s' % (opts.split),
                                opts.name, 'epoch_%d' % opts.num_train_epoch)
    if not osp.exists(opts.results_dir):
        print('writing to %s' % opts.results_dir)
        os.makedirs(opts.results_dir)

    torch.manual_seed(0)
    tester = ShapeTester(opts)
    tester.init_testing()
    tester.test()

    opts.batch_size = 8
    tester_kt = ShapeTester(opts)
    tester_kt.init_testing(kt=True)
    tester_kt.test_kt()


if __name__ == '__main__':
    app.run(main)
