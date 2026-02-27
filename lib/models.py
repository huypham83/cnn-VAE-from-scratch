import cupy as cp
from lib.layers import *

class AutoEncoder():
    def __init__(self, input_dim, hidden_dim, latent_dim):
        self.enc1 = Dense(input_dim, hidden_dim, activation='relu')
        self.enc2 = Dense(hidden_dim, latent_dim, activation='relu')

        self.dec1 = Dense(latent_dim, hidden_dim, activation='relu')
        self.dec2 = Dense(hidden_dim, input_dim, activation='sigmoid')

    def forward(self, input):
        out = self.enc1.forward(input)
        self.latent = self.enc2.forward(out)
        out = self.dec1.forward(self.latent)
        self.reconstruction = self.dec2.forward(out)
        return self.reconstruction

    def backward(self, out_grad):
        grad = out_grad
        grad = self.dec2.backward(grad)
        grad = self.dec1.backward(grad)

        grad = self.enc2.backward(grad)
        grad = self.enc1.backward(grad)

        return grad
    
    def get_layer(self):
        return [self.enc1, self.enc2, self.dec1, self.dec2]

class VariationalAutoEncoder():
    def __init__(self, latent_dim, image_height, image_width, image_channel):
        self.H = image_height
        self.W = image_width
        self.C = image_channel

        self.conv1 = Conv2D(in_channel=self.C, out_channel=32, kernel_height=3, kernel_width=3, stride=2, activation='relu')
        self.conv2 = Conv2D(in_channel=32, out_channel=64, kernel_height=3, kernel_width=3, stride=2, activation='relu')
        self.flatten = Flatten()

        c_H = self.H // 4
        c_W = self.W // 4

        self.enc_mean = Dense(64 * c_H * c_W, latent_dim)
        self.enc_var = Dense(64 * c_H * c_W, latent_dim)

        self.dec = Dense(latent_dim, 64 * c_H * c_W, activation='relu')
        self.reshape = Reshape((64, c_H, c_W))

        self.tconv1 = TransposeConv2D(in_channel=64, out_channel=32, kernel_height=4, kernel_width=4, stride=2, padding=1, activation='relu')

        self.tconv2 = TransposeConv2D(in_channel=32, out_channel=self.C, kernel_height=4, kernel_width=4, stride=2, padding=1, activation='sigmoid')

        self.f = Flatten()

    def sampling(self, mean, log_var):
        self.std = cp.exp(0.5 * log_var)
        self.eps = cp.random.normal(0, 1, mean.shape)
        return mean + (self.eps * self.std)

    def forward(self, input):
        img = input.reshape(-1, self.C, self.H, self.W)

        out = self.conv1.forward(img)
        out = self.conv2.forward(out)
        out = self.flatten.forward(out)

        self.mean = self.enc_mean.forward(out)
        self.var = self.enc_var.forward(out)
        self.z = self.sampling(self.mean, self.var)

        out = self.dec.forward(self.z)
        out = self.reshape.forward(out)
        out = self.tconv1.forward(out)
        out = self.tconv2.forward(out)
        self.reconstruction = self.f.forward(out)
        return self.reconstruction, self.mean, self.var

    def backward(self, mse_grad, kl_grad_mean, kl_grad_var):
        grad = self.f.backward(mse_grad)
        grad = self.tconv2.backward(grad)
        grad = self.tconv1.backward(grad)
        grad = self.reshape.backward(grad)
        grad = self.dec.backward(grad)

        dmean = grad + kl_grad_mean
        dvar = (grad * self.eps * 0.5 * self.std) + kl_grad_var

        d_enc_mean = self.enc_mean.backward(dmean)
        d_enc_var = self.enc_var.backward(dvar)
        grad = d_enc_mean + d_enc_var

        grad = self.flatten.backward(grad)
        grad = self.conv2.backward(grad)
        grad = self.conv1.backward(grad)

        return grad

    def get_layer(self):
        return [self.conv1, self.conv2, self.enc_mean, self.enc_var, self.dec, self.tconv1, self.tconv2]
    
# Dead code from above, dont touch 

class SequentialBase():
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x, is_training=True):
        out = x
        for layer in self.layers:
            if type(layer).__name__ == 'BatchNormalization2D':
                out = layer.forward(out, is_training=is_training)
            else:
                out = layer.forward(out)
        return out
    
    def backward(self, out_grad):
        grad = out_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def get_layer(self):
        return list(self.layers)

# GAN starts here

class Generator():
    def __init__(self, latent_dim, image_height=28, image_width=28, image_channel=1):
        self.H = image_height
        self.W = image_width
        self.C = image_channel
        self.latent_dim = latent_dim
        
        c_C = 64
        self.model = SequentialBase(
            Dense(latent_dim, c_C * 7 * 7),
            Reshape((c_C, 7, 7)),
            BatchNormalization2D(c_C),
            Activation('relu'),

            UpSample2D(scale=2),
            Conv2D(in_channel=c_C, out_channel=32, kernel_height=3, kernel_width=3),
            BatchNormalization2D(32),
            Activation('relu'),

            UpSample2D(scale=2),
            Conv2D(in_channel=32, out_channel=16, kernel_height=3, kernel_width=3),
            BatchNormalization2D(16),
            Activation('relu'),
            
            Conv2D(in_channel=16, out_channel=self.C, kernel_height=3, kernel_width=3, activation='sigmoid'),
            Flatten()
        )

    def sampling(self, batch_size):
        return cp.random.normal(0, 1, (batch_size, self.latent_dim))
    
    def forward(self, batch_size, is_training=True):
        z = self.sampling(batch_size)
        return self.model.forward(z, is_training)
    
    def backward(self, out_grad):
        return self.model.backward(out_grad)
    
    def get_layer(self):
        return self.model.get_layer()

class Discriminator():
    def __init__(self, image_height=28, image_width=28, image_channel=1):
        self.H = image_height
        self.W = image_width
        self.C = image_channel
        
        self.model = SequentialBase(
            Reshape((self.C, self.H, self.W)),
            
            Conv2D(in_channel=self.C, out_channel=16, kernel_height=3, kernel_width=3, stride=1),
            Activation('leaky_relu'),
            MaxPooling2D(scale=2),
            
            Conv2D(in_channel=16, out_channel=32, kernel_height=3, kernel_width=3, stride=1),
            Activation('leaky_relu'),
            MaxPooling2D(scale=2),
            
            Flatten(),
            Dense(32 * 7 * 7, 1, activation='sigmoid')
        )

    def forward(self, input, is_training=True):
        return self.model.forward(input, is_training=is_training)
    
    def backward(self, out_grad):
        return self.model.backward(out_grad)
    
    def get_layer(self):
        return self.model.get_layer()