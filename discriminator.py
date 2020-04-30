import chainer
import chainer.functions as cf
import chainer.links as cl

import sn.sn_convolution_2d
import sn.sn_linear


def to_one_hot_vector(data, num_classes):
    xp = chainer.cuda.get_array_module(data)
    data2 = xp.zeros((data.size, num_classes), 'float32')
    data2[xp.arange(data.size), data] = 1
    return data2


class ShapeNetPatchDiscriminator(chainer.Chain):
    def __init__(self, dim_in=4):
        super(ShapeNetPatchDiscriminator, self).__init__()

        with self.init_scope():
            init = chainer.initializers.HeNormal()
            dims = [dim_in, 32, 64, 128, 256, 256, 1]
            Convolution2D = sn.sn_convolution_2d.SNConvolution2D
            Linear = sn.sn_linear.SNLinear
            self.conv1 = Convolution2D(dims[0], dims[1], 5, stride=2, pad=2, initialW=init, nobias=True)
            self.conv2 = Convolution2D(dims[1] * 2, dims[2], 5, stride=2, pad=2, initialW=init, nobias=True)
            self.conv3 = Convolution2D(dims[2], dims[3], 5, stride=2, pad=2, initialW=init, nobias=True)
            self.conv4 = Convolution2D(dims[3], dims[4], 5, stride=2, pad=2, initialW=init, nobias=True)
            self.conv5 = Convolution2D(dims[4], dims[5], 5, stride=2, pad=2, initialW=init, nobias=True)
            self.conv6 = Convolution2D(dims[5], dims[6], 5, stride=2, pad=2, initialW=init)
            self.linear_v = Linear(3, dims[1], initialW=init, nobias=True)
            self.linear_labels = Linear(13, dims[-2], nobias=True)

            self.conv1_bn = cl.BatchNormalization(dims[1], use_gamma=False)
            self.conv2_bn = cl.BatchNormalization(dims[2], use_gamma=False)
            self.conv3_bn = cl.BatchNormalization(dims[3], use_gamma=False)
            self.conv4_bn = cl.BatchNormalization(dims[4], use_gamma=False)
            self.conv5_bn = cl.BatchNormalization(dims[5], use_gamma=False)
            self.linear_v_bn = cl.BatchNormalization(dims[1], use_gamma=False)

    def __call__(self, x, v, labels=None):
        hi = cf.leaky_relu(self.conv1_bn(self.conv1(x)))  # [224 -> 112]
        hm = cf.leaky_relu(self.linear_v_bn(self.linear_v(v)))
        hm = cf.broadcast_to(hm[:, :, None, None], hi.shape)
        h = cf.concat((hi, hm), axis=1)

        h = cf.leaky_relu(self.conv2_bn(self.conv2(h)))  # [112 -> 56]
        h = cf.leaky_relu(self.conv3_bn(self.conv3(h)))  # [56 -> 28]
        h = cf.leaky_relu(self.conv4_bn(self.conv4(h)))  # [28 -> 14]
        h = cf.leaky_relu(self.conv5_bn(self.conv5(h)))  # [14 -> 7]
        h1 = self.conv6(h)  # [7 -> 4]

        if labels is not None:
            labels = to_one_hot_vector(labels, 13)
            h2 = self.linear_labels(labels)  # [bs, 256]
            h2 = cf.broadcast_to(h2[:, :, None, None], h.shape)  # [bs, 256, 7, 7]
            h2 = cf.sum(h2 * h, axis=1, keepdims=True)  # [bs, 1, 7, 7]
            h2 = h2[:, :, ::2, ::2]
            return h1 + h2
        else:
            return h1


def get_discriminator(name, dim_in):
    if name == 'shapenet_patch':
        return ShapeNetPatchDiscriminator(dim_in)
