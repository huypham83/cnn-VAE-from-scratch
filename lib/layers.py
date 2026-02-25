import cupy as cp
import numpy as np
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
            case 'tanh':
                self.forward = self._tanh_forward
                self.backward = self._tanh_backward
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
    
    def _tanh_forward(self, x):
        self.output = cp.tanh(x)
        return self.output
    
    def _tanh_backward(self, out_grad):
        return (1 - self.output**2) * out_grad

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
    
class CrossEntropyLoss():
    def forward(self, logits, targets):
        self.targets = targets
        batch_size, seq_len, vocab_size = logits.shape

        e_x = cp.exp(logits - cp.max(logits, axis=-1, keepdims=True))
        self.probs = e_x / cp.sum(e_x, axis=-1, keepdims=True)

        prob_flat = self.probs.reshape(-1, vocab_size)
        target_flat = targets.reshape(-1)

        correct_confidences = prob_flat[cp.arange(len(target_flat)), target_flat]
        
        loss = -cp.sum(cp.log(correct_confidences + 1e-7)) / (batch_size * seq_len)
        return loss

    def backward(self):
        batch_size, seq_len, vocab_size = self.probs.shape
        
        grad = self.probs.reshape(-1, vocab_size).copy()
        target_flat = self.targets.reshape(-1)
        grad[cp.arange(len(target_flat)), target_flat] -= 1
        
        grad /= (batch_size * seq_len)
        return grad.reshape(batch_size, seq_len, vocab_size)
    
class Embedding():
    def __init__(self, vocab_size, embed_dim, max_len, positional = False):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_embedding = cp.random.randn(vocab_size, embed_dim) * 0.01
        self.max_len = max_len
        self.pos_encoding = cp.zeros((1, max_len, embed_dim))
        self.embedding_grad = cp.zeros_like(self.token_embedding)
        if positional == True:
            self.pos_encoding = self.generate_pos_encoding()
    
    def generate_pos_encoding(self):
        pe = np.zeros((self.max_len, self.embed_dim))
        position = np.arange(0, self.max_len).reshape(-1, 1)

        div_term = np.exp(np.arange(0, self.embed_dim, 2) * -(np.log(10000.0) / self.embed_dim))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = cp.array(pe)
        return pe.reshape(1, self.max_len, self.embed_dim)

    def forward(self, input):
        self.input = input
        seq_len = input.shape[1]
        return self.token_embedding[input] + self.pos_encoding[:, :seq_len, :]

    def backward(self, out_grad):
        self.embedding_grad.fill(0)
        cp.add.at(self.embedding_grad, self.input, out_grad)
        return None
    
    def get_params(self):
        return [('token_embedding', 'embedding_grad')]
    
class SoftMax():
    def forward(self, x):
        e_x = cp.exp(x - cp.max(x, axis=-1, keepdims=True))
        self.output = e_x / cp.sum(e_x, axis=-1, keepdims=True)
        return self.output

    def backward(self, out_grad):
        u = out_grad * self.output
        v = self.output * cp.sum(u, axis=-1, keepdims=True)
        return u - v

class Dense():
    def __init__(self, input_size, output_size, activation = None):
        self.output_size = output_size
        self.input_size = input_size
        std = 1 / cp.sqrt(input_size)
        self.W = cp.random.uniform(-std, std, (input_size, output_size))
        self.b = cp.zeros(output_size)
        self.dW = cp.zeros_like(self.W)
        self.db = cp.zeros_like(self.b)
        self.act = Activation(activation)
    def forward(self, input):
        self.input = input
        self.z = self.input @ self.W + self.b
        return self.act.forward(self.z)
    def backward(self, out_grad):
        out_grad = self.act.backward(out_grad)
        self.dz = out_grad

        self.dW[:] = self.input.T @ self.dz
        self.db[:] = cp.sum(self.dz, axis = 0)
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
        self.dW = cp.zeros_like(self.W)
        self.db = cp.zeros_like(self.b)
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

        dW = d_out @ self.x_cols.T
        self.dW[:] = dW.reshape(self.W.shape)

        self.db[:] = cp.sum(d_out, axis=1, keepdims=True).reshape(self.out_channel, 1)

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
        self.dW = cp.zeros_like(self.W)
        self.db = cp.zeros_like(self.b)
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
        dW = self.x_cols @ grad_cols.T
        self.dW[:] = dW.reshape(self.in_channel, self.out_channel, self.kernel_height, self.kernel_width)
        self.inp_grad = self.w_cols @ grad_cols
        self.inp_grad = self.inp_grad.reshape(self.in_channel, self.H_in, self.W_in, -1).transpose(3, 0, 1, 2)
        self.db[:] = cp.sum(out_grad, axis=(0, 2, 3), keepdims=True).reshape(self.out_channel, -1)

        return self.inp_grad
    
    def get_params(self):
        return [('W', 'dW'), ('b', 'db')]
    
#RNN and LSTM

class RNN():
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        std = 1 / cp.sqrt(hidden_size) 

        self.Whh = cp.random.uniform(-std, std, (hidden_size, hidden_size))
        self.Wxh = cp.random.uniform(-std, std, (input_size, hidden_size))
        self.b = cp.zeros((1, hidden_size))
        self.dWhh = cp.zeros_like(self.Whh)
        self.dWxh = cp.zeros_like(self.Wxh)
        self.db = cp.zeros_like(self.b)

    def forward(self, input, prev_state = None):
        self.input = input
        batch_size, seq_len, _ = input.shape
        self.h = cp.zeros((batch_size, seq_len + 1, self.hidden_size))

        # Clip gradients to prevent exploding gradients during BPTT
        self.acts = [Activation('tanh') for _ in range(seq_len)]

        if prev_state is not None:
            h_prev = prev_state
            self.h[:, 0, :] = h_prev

        for t in range(seq_len):
            xt = input[:, t, :]
            h_prev = self.h[:, t, :]
            self.h[:, t + 1, :] = self.acts[t].forward(xt @ self.Wxh + h_prev @ self.Whh + self.b)

        return self.h[:, 1:, :]

    def backward(self, out_grad):
        batch_size, seq_len, _ = out_grad.shape

        inp_grad = cp.zeros_like(self.input)
        dh_next = cp.zeros((batch_size, self.hidden_size))

        for t in range(seq_len - 1, -1, -1):
            grad = out_grad[:, t, :] + (dh_next @ self.Whh.T)

            dz = self.acts[t].backward(grad)

            self.db += cp.sum(dz, axis=0)
            self.dWxh += self.input[:, t, :].T @ dz
            self.dWhh += self.h[:, t, :].T @ dz

            inp_grad[:, t, :] = dz @ self.Wxh.T
            dh_next = dz

        for param in [self.dWxh, self.dWhh, self.db]:
            cp.clip(param, -1, 1, out=param)

        return inp_grad
    
    def get_params(self):
        return [('Wxh', 'dWxh'), ('Whh', 'dWhh'), ('b', 'db')]
    
class LSTM():
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        std = 1 / cp.sqrt(hidden_size)

        self.W = cp.random.uniform(-std, std, (input_size + hidden_size, 4 * hidden_size))
        self.b = cp.zeros((1, 4 * hidden_size))
        self.b[:, :hidden_size] = 1.0
        
        self.dW = cp.zeros_like(self.W)
        self.db = cp.zeros_like(self.b)
    def forward(self, input, prev_state = None):
        self.input = input
        batch_size, seq_len, _ = input.shape
        self.h = cp.zeros((batch_size, seq_len + 1, self.hidden_size))
        self.c = cp.zeros((batch_size, seq_len + 1, self.hidden_size))

        self.f_acts = [Activation('sigmoid') for _ in range(seq_len)]
        self.i_acts = [Activation('sigmoid') for _ in range(seq_len)]
        self.g_acts = [Activation('tanh') for _ in range(seq_len)]
        self.o_acts = [Activation('sigmoid') for _ in range(seq_len)]
        self.c_acts = [Activation('tanh') for _ in range(seq_len)]

        if prev_state is not None:
            h_prev, c_prev = prev_state
            self.h[:, 0, :] = h_prev
            self.c[:, 0, :] = c_prev

        self.gates = cp.zeros((batch_size, seq_len, 4 * self.hidden_size))
        hd = self.hidden_size

        for t in range(seq_len):
            xt = self.input[:, t, :]
            h_prev = self.h[:, t, :]
            c_prev = self.c[:, t, :]

            xh = cp.hstack([xt, h_prev])

            gates = xh @ self.W + self.b

            f = self.f_acts[t].forward(gates[:, :hd])
            i = self.i_acts[t].forward(gates[:, hd:2 * hd])
            g = self.g_acts[t].forward(gates[:, 2 * hd:3 * hd])
            o = self.o_acts[t].forward(gates[:, 3 * hd:])

            self.c[:, t + 1, :] = (c_prev * f) + (g * i)
            self.h[:, t + 1, :] = o * self.c_acts[t].forward(self.c[:, t + 1, :])

        last_h = self.h[:, -1, :]
        last_c = self.c[:, -1, :]

        return self.h[:, 1:, :], (last_h, last_c)
    
    def backward(self, out_grad):
        batch_size, seq_len, _ = out_grad.shape

        inp_grad = cp.zeros_like(self.input)

        dh_next = cp.zeros((batch_size, self.hidden_size))
        dc_next = cp.zeros((batch_size, self.hidden_size))

        for t in range(seq_len - 1, -1, -1):
            dh = out_grad[:, t, :] + dh_next
            
            o = self.o_acts[t].output
            tanh_c = self.c_acts[t].output
            f = self.f_acts[t].output
            i = self.i_acts[t].output
            g = self.g_acts[t].output
            c_prev = self.c[:, t, :]
            
            do = self.o_acts[t].backward(dh * tanh_c)
            dc = self.c_acts[t].backward(dh * o) + dc_next
            
            df = self.f_acts[t].backward(dc * c_prev)
            di = self.i_acts[t].backward(dc * g)
            dg = self.g_acts[t].backward(dc * i)
            
            dgates = cp.hstack([df, di, dg, do])
            
            xh = cp.hstack([self.input[:, t, :], self.h[:, t, :]])
            self.dW += xh.T @ dgates
            self.db += cp.sum(dgates, axis=0, keepdims=True)
            
            dxh = dgates @ self.W.T
            inp_grad[:, t, :] = dxh[:, :self.input_size]
            dh_next = dxh[:, self.input_size:]
            dc_next = dc * f

        for param in [self.dW, self.db]: 
            cp.clip(param, -1, 1, out=param)
            
        return inp_grad

    def get_params(self):
        return [('W', 'dW'), ('b', 'db')]