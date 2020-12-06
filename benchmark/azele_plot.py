"""
Script for testing on CUB.

Sample usage:
python -m benchmark.azele_plot --split val --name <model_name> --num_train_epoch <model_epoch>
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
        self.dataloader = self.data_module.data_loader(opts)  # if not kt else self.data_module.cub_test_pair_dataloader(opts)

    def test(self):
        result_path_prefix = opts.results_dir
        print('Writing to %s' % result_path_prefix)
        azs, elevs = [], []

        n_iter = len(self.dataloader)
        for i, batch in enumerate(self.dataloader):
            if i % 100 == 0:
                print('{}/{} evaluation iterations.'.format(i, n_iter))
            if opts.max_eval_iter > 0 and (i >= opts.max_eval_iter):
                break
            az, ele = self.predictor.predict_az_ele(batch)
            azs.append(az.detach().cpu().numpy())
            elevs.append(ele.detach().cpu().numpy())

        azs = np.vstack(azs)
        elevs = np.vstack(elevs)
        fig = plt.figure(figsize=(30, 15))
        for i in range(azs.shape[1]):
            plt.subplot(2, 4, 1 + i)
            plt.scatter(azs[:, i], elevs[:, i], s=2)
            plt.xlabel('Azimuth')
            plt.ylabel('Elevation')
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.title('Camera %d' % i)
            plt.plot()
        fig.tight_layout()
        plt.savefig(osp.join(result_path_prefix, 'azele_plot.png'))


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


if __name__ == '__main__':
    app.run(main)
