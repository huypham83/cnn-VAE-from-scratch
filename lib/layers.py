import cupy as cp
from utils import *

class Activation():
    def __init__(self, func=None):
        match func:
            case 'relu':
                self.forward = self._relu_forward
                self.backward = self._relu_backward
            case 'leaky_relu':
                self.alpha = 0.2
                self.forward = self._lrelu_forward
                self.backward = self._lrelu_backward
            case 'sigmoid':
                self.forward = self._sigmoid_forward
                self.backward = self._sigmoid_backward
            case None:
                self.forward = self._id_forward
                self.backward = self._id_backward

    def _relu_forward(self, x):
        self.x = x
        return cp.maximum(0, x)
    def _relu_backward(self, out_grad):
        return (self.x > 0).astype(float) * out_grad
    
    def _lrelu_forward(self, x):
        self.x = x
        return cp.where(x > 0, x, self.alpha * x)
    def _lrelu_backward(self, out_grad):
        grad_mask = cp.where(self.x > 0, 1, self.alpha)
        return grad_mask * out_grad
    
    def _sigmoid_forward(self, x):
        self.output = 1 / (1 + cp.exp(-x))
        return self.output
    def _sigmoid_backward(self, out_grad):
        return self.output * (1 - self.output) * out_grad

    def _id_forward(self, x):
        return x
    def _id_backward(self, out_grad):
        return out_grad

class Flatten():
    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(self.x_shape[0], -1)

    def backward(self, out_grad):
        return out_grad.reshape(self.x_shape)

class Reshape():
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.N = x.shape[0]
        return x.reshape(self.N, *self.shape)

    def backward(self, out_grad):
        return out_grad.reshape(self.N, -1)

class MSELoss():
    def forward(self, y_pred, y_true):
        return cp.mean((y_pred - y_true) ** 2)
    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_pred.shape[0]

class VAELoss():
    def forward(self, y_pred, y_true, mean, log_var):
        batch_size = y_pred.shape[0]
        self.mse_loss = cp.sum((y_pred - y_true) ** 2) / batch_size
        self.kl_loss = cp.mean((-0.5 * cp.sum(1 + log_var - mean**2 - cp.exp(log_var), axis=1)))
        return self.mse_loss + self.kl_loss
    def backward(self, y_pred, y_true, mean, log_var):
        batch_size = y_pred.shape[0]
        mse_grad = 2 * (y_pred - y_true) / batch_size
        kl_grad_mean = mean / batch_size
        kl_grad_var = -0.5 * (1 - cp.exp(log_var)) / batch_size
        return mse_grad, kl_grad_mean, kl_grad_var
    
class BCELoss():
    def __init__(self):
        self.eps = 1e-7
    def forward(self, y_pred, y_true):
        y_pred = cp.clip(y_pred, self.eps, 1 - self.eps)

        self.loss = -cp.mean(y_true * cp.log(y_pred) + (1 - y_true) * cp.log(1 - y_pred))
        return self.loss
    def backward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]
        y_pred = cp.clip(y_pred, self.eps, 1 - self.eps)
        
        grad = -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
        return grad / batch_size

class Dense():
    def __init__(self, input_size, output_size, activation = None):
        self.output_size = output_size
        self.input_size = input_size
        std = 1 / cp.sqrt(input_size)
        self.W = cp.random.uniform(-std, std, (input_size, output_size))
        self.b = cp.zeros(output_size)
        self.act = Activation(activation)
    def forward(self, input):
        self.input = input
        self.z = self.input @ self.W + self.b
        return self.act.forward(self.z)
    def backward(self, out_grad):
        out_grad = self.act.backward(out_grad)
        self.dz = out_grad

        self.dW = self.input.T @ self.dz
        self.db = cp.sum(self.dz, axis = 0)
        self.inp_grad = self.dz @ self.W.T
        
        return self.inp_grad

    def get_params(self):
        return [('W', 'dW'), ('b', 'db')]
    
class Conv2D():
    def __init__(self, in_channel, out_channel, kernel_height, kernel_width, stride=1, activation = None):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.stride = stride
        self.padding = max(kernel_height, kernel_width) // 2

        std = 1 / cp.sqrt(in_channel * kernel_width * kernel_height)
        self.W = cp.random.uniform(-std, std, (out_channel, in_channel, kernel_height, kernel_width))
        self.b = cp.zeros((out_channel, 1))
        self.act = Activation(activation)

    def forward(self, x):
        self.x_shape = x.shape
        N, C, H, W = x.shape
        self.x_cols = im2col(x, self.kernel_height, self.kernel_width, self.padding, self.stride)
        self.w_cols = self.W.reshape(self.out_channel, -1)

        out = self.w_cols @ self.x_cols + self.b

        out_height = (H + 2 * self.padding - self.kernel_height) // self.stride + 1
        out_width = (W + 2 * self.padding - self.kernel_width) // self.stride + 1

        out = out.reshape(self.out_channel, out_height, out_width, N)
        out = out.transpose(3, 0, 1, 2)
        return self.act.forward(out)

    def backward(self, out_grad):
        out_grad = self.act.backward(out_grad)
        d_out = out_grad.transpose(1, 2, 3, 0).reshape(self.out_channel, -1)

        self.dW = d_out @ self.x_cols.T
        self.dW = self.dW.reshape(self.W.shape)

        self.db = cp.sum(d_out, axis=1, keepdims=True).reshape(self.out_channel, 1)

        d_cols = self.w_cols.T @ d_out

        inp_grad = col2im(d_cols, self.x_shape, self.kernel_height, self.kernel_width, self.padding, self.stride)

        return inp_grad
    
    def get_params(self):
        return [('W', 'dW'), ('b', 'db')]

# use even size kernel only bcz uhhh math
class TransposeConv2D():
    def __init__(self, in_channel, out_channel, kernel_height, kernel_width, stride=1, padding=-1, activation = None):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.stride = stride
        if padding == -1:
            self.padding = max(kernel_height, kernel_width) // 2
        else:
            self.padding = padding

        std = 1 / cp.sqrt(in_channel * kernel_width * kernel_height)
        self.W = cp.random.uniform(-std, std, (in_channel, out_channel, kernel_height, kernel_width))
        self.b = cp.zeros((out_channel, 1))
        self.act = Activation(activation)

    def forward(self, x):
        self.input = x
        self.x_shape = x.shape
        N, C, self.H_in, self.W_in = x.shape
        self.w_cols = self.W.reshape(self.in_channel, -1)
        self.x_cols = self.input.transpose(1, 2, 3, 0).reshape(self.in_channel, -1)

        out = self.w_cols.T @ self.x_cols

        out_height = (self.H_in - 1) * self.stride - 2 * self.padding + self.kernel_height
        out_width = (self.W_in - 1) * self.stride - 2 * self.padding + self.kernel_width

        out = col2im(out, (N, self.out_channel, out_height, out_width), self.kernel_height, self.kernel_width, self.padding, self.stride)
        out += self.b.reshape(1, -1, 1, 1)

        return self.act.forward(out)

    def backward(self, out_grad):
        out_grad = self.act.backward(out_grad)
        grad_cols = im2col(out_grad, self.kernel_height, self.kernel_width, self.padding, self.stride)
        self.dW = self.x_cols @ grad_cols.T
        self.dW = self.dW.reshape(self.in_channel, self.out_channel, self.kernel_height, self.kernel_width)
        self.inp_grad = self.w_cols @ grad_cols
        self.inp_grad = self.inp_grad.reshape(self.in_channel, self.H_in, self.W_in, -1).transpose(3, 0, 1, 2)
        self.db = cp.sum(out_grad, axis=(0, 2, 3), keepdims=True).reshape(self.out_channel, -1)

        return self.inp_grad
    
    def get_params(self):
        return [('W', 'dW'), ('b', 'db')]