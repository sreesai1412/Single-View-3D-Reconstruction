import chainer
import numpy as np
from chainer import cuda
from chainer.functions.array.broadcast import broadcast_to
from chainer.functions.connection import convolution_2d
from chainer.links.connection.convolution_2d import Convolution2D
from max_sv import max_singular_value


class SNConvolution2D(Convolution2D):
    """Two-dimensional convolutional layer with spectral normalization.

    Attributes:
        W (~chainer.Variable): Weight parameter.
        W_bar (~chainer.Variable): Spectrally normalized weight parameter.
        b (~chainer.Variable): Bias parameter.
        u (~numpy.array): Current estimation of the right largest singular vector of W.
        (optional) gamma (~chainer.Variable): the multiplier parameter.
        (optional) factor (float): constant factor to adjust spectral norm of W_bar.

    """

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None, use_gamma=False, Ip=1, factor=None):
        self.Ip = Ip
        self.use_gamma = use_gamma
        self.factor = factor
        super(SNConvolution2D, self).__init__(
            in_channels, out_channels, ksize, stride, pad,
            nobias, initialW, initial_bias)
        self.u = np.random.normal(size=(1, out_channels)).astype(dtype="f")
        self.register_persistent('u')

    @property
    def W_bar(self):
        """
        Spectrally Normalized Weight
        """
        W_mat = self.W.reshape(self.W.shape[0], -1)
        sigma, _u, _ = max_singular_value(W_mat, self.u, self.Ip)
        if self.factor:
            sigma = sigma / self.factor
        sigma = broadcast_to(sigma.reshape((1, 1, 1, 1)), self.W.shape)
        if chainer.config.train:
            # Update estimated 1st singular vector
            self.u[:] = _u
        if hasattr(self, 'gamma'):
            return broadcast_to(self.gamma, self.W.shape) * self.W / sigma
        else:
            return self.W / sigma

    def _initialize_params(self, in_size):
        super(SNConvolution2D, self)._initialize_params(in_size)
        if self.use_gamma:
            W_mat = self.W.data.reshape(self.W.shape[0], -1)
            _, s, _ = np.linalg.svd(W_mat)
            with self.init_scope():
                self.gamma = chainer.Parameter(s[0], (1, 1, 1, 1))

    def __call__(self, x):
        """Applies the convolution layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of the convolution.

        """
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return convolution_2d.convolution_2d(
            x, self.W_bar, self.b, self.stride, self.pad)
