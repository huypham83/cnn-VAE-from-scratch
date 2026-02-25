import pickle
import os
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from lib.models import VariationalAutoEncoder
from lib.layers import VAELoss
from lib.data import CIFAR10DataLoader
from lib.optimizer import SGD
from lib.utils import *

input_dim = 784
hidden_dim = 128
latent_dim = 128
batch_size = 32
lr = 0.005
epochs = 100
train = False

file_name = 'weight/vae_weights.pkl'
model = VariationalAutoEncoder(latent_dim, image_height=32, image_width=32, image_channel=3)
optimizer = SGD(model.get_layer(), lr=0.005)

if os.path.exists(file_name):
    print(f"Found existing weights at {file_name}. Loading model...")
    with open(file_name, 'rb') as f:
        trained_weights = pickle.load(f)
        apply_state(model, trained_weights['model'])

    print("Loaded model")

else:
    print("No existing weights found. Training from scratch...")
    train = True

if train == True:
    print(f"Starting training sequence...")

    loader = CIFAR10DataLoader('./dataset/CIFAR-10', batch_size)
    loss_func = VAELoss()
    for epoch in range(epochs):
        loss = 0
        batches = 0
        for x_batch, _ in loader.next_batch():
            reconstruction, mean, var = model.forward(x_batch)
            loss += loss_func.forward(reconstruction, x_batch, mean, var)
            batches += 1
            mse_grad, kl_grad_mean, kl_grad_var = loss_func.backward(reconstruction, x_batch, mean, var)
            model.backward(mse_grad, kl_grad_mean, kl_grad_var)
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss / batches:.6f}")

    print("Training complete")

    model_weights = {
        'model':extract_state(model)
    }
    with open(file_name, 'wb') as f:
        pickle.dump(model_weights, f)

    print(f"Model weights are saved to {file_name}.")

num_samples = 20
cols = 5
rows = (num_samples + cols - 1) // cols

random_z = cp.random.normal(0, 1, (num_samples, latent_dim))

out = model.dec.forward(random_z)
out = model.reshape.forward(out)
out = model.tconv1.forward(out)
out = model.tconv2.forward(out)
generated_images = model.f.forward(out)

generated_images_np = cp.asnumpy(generated_images)

plt.figure(figsize=(cols * 2.5, rows * 2.5))
for i in range(num_samples):
    ax = plt.subplot(rows, cols, i + 1)

    plt.imshow(generated_images_np[i].reshape(3, 32, 32).transpose(1, 2, 0))
    plt.axis('off')
plt.suptitle("Generated images", fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()