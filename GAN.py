import pickle
import os
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from lib.models import Generator, Discriminator
from lib.layers import BCELoss
from lib.data import EMNISTDataLoader
from lib.optimizer import SGD, Momentum, Adam
from lib.utils import *

latent_dim = 128
batch_size = 32
lr = 0.0002
epochs = 50
train = False

file_name = 'weight/gan_weights.pkl'
os.makedirs(os.path.dirname(file_name), exist_ok=True)
generator = Generator(latent_dim=latent_dim, image_height=28, image_width=28, image_channel=1)
discriminator = Discriminator(image_height=28, image_width=28, image_channel=1)
g_optimizer = Adam(generator.get_layer(), lr=lr, beta1=0.5, beta2=0.999)
d_optimizer = Adam(discriminator.get_layer(), lr=lr, beta1=0.5, beta2=0.999)

note = """
    Generator (Lightweight): 
    - Dense: 128 -> 64  * 7 * 7
    - Reshape: 64, 7, 7 -> BN(64) -> ReLU
    - UpSample(2) -> Conv2D(c=32) -> BN(32) -> ReLU
    - UpSample(2) -> Conv2D(c=16) -> BN(16) -> ReLU
    - Conv2D(c=1) -> Sigmoid -> Flatten
    Discriminator (Lightweight):
    - Reshape: 1, 28, 28
    - Conv2D(c=16) -> lReLU -> MaxPool(2)
    - Conv2D(c=32) -> lReLU -> MaxPool(2)
    - Flatten -> Dense: 32 * 7 * 7 -> 1 -> Sigmoid
"""

if os.path.exists(file_name):
    print(f"Found existing weights at {file_name}. Loading model...")
    with open(file_name, 'rb') as f:
        trained_weights = pickle.load(f)
        start_epoch = trained_weights.get('epoch', 0)
        if 'architecture' in trained_weights:
            print(trained_weights['architecture'])
        apply_state(generator, trained_weights['generator'])
        apply_state(discriminator, trained_weights['discriminator'])

        if 'g_optim' in trained_weights:
            g_optimizer.load_state(trained_weights['g_optim'])
            d_optimizer.load_state(trained_weights['d_optim'])
            
    print("Loaded model successfully!")

else:
    print("No existing weights found. Training from scratch...")
    train = True

if train == True:
    print(f"Starting training sequence...")

    loader = EMNISTDataLoader('./dataset/EMNIST', batch_size)
    loss_func = BCELoss()
    for epoch in range(start_epoch, epochs):
        d_loss_total = 0
        g_loss_total = 0
        batches = 0
        for x_batch, _ in loader.next_batch():
            current_batch_size = x_batch.shape[0]
            
            real_labels = cp.full((current_batch_size, 1), 0.9)
            fake_labels = cp.zeros((current_batch_size, 1))

            fakes = generator.forward(current_batch_size)
            x = cp.vstack((x_batch, fakes))
            y = cp.vstack((real_labels, fake_labels))

            d_pred = discriminator.forward(x, is_training=True)
            d_loss = loss_func.forward(d_pred, y)
            
            grad_combined = loss_func.backward(d_pred, y)
            discriminator.backward(grad_combined)
            d_optimizer.step()

            d_real_score = cp.mean(d_pred[:current_batch_size])
            d_fake_score = cp.mean(d_pred[current_batch_size:])
            d_loss_total += d_loss

            # Train G
            new_fakes = generator.forward(current_batch_size)
            
            d_pred_new = discriminator.forward(new_fakes)
            
            g_loss = loss_func.forward(d_pred_new, real_labels)
            g_loss_total += g_loss

            grad = loss_func.backward(d_pred_new, real_labels)
            grad = discriminator.backward(grad)
            generator.backward(grad)
            g_optimizer.step()

            batches += 1
            
        print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss_total/batches:.4f} | G Loss: {g_loss_total/batches:.4f} | D(x): {d_real_score:.4f} | D(G(z)): {d_fake_score:.4f}")

        if (epoch + 1) % 5 == 0:
            model_state = {
                'epoch': epoch + 1,
                'architecture': note,
                'generator': extract_state(generator),
                'discriminator': extract_state(discriminator),
                'g_optim':g_optimizer.save_state(),
                'd_optim':d_optimizer.save_state()
            }

            with open(file_name, 'wb') as f:
                pickle.dump(model_state, f)

            print(f"Model weights are saved to {file_name}.")

    print("Training complete")

model_state = {
    'epoch': epoch + 1,
    'architecture': note,
    'generator': extract_state(generator),
    'discriminator': extract_state(discriminator),
    'g_optim':g_optimizer.save_state(),
    'd_optim':d_optimizer.save_state()
}

with open(file_name, 'wb') as f:
    pickle.dump(model_state, f)

print(f"Model weights are saved to {file_name}.")

num_samples = 20
cols = 5
rows = (num_samples + cols - 1) // cols

generated_images_np = cp.asnumpy(generator.forward(num_samples, is_training=False))

plt.figure(figsize=(cols * 2.5, rows * 2.5))
for i in range(num_samples):
    ax = plt.subplot(rows, cols, i + 1)

    plt.imshow(generated_images_np[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.suptitle("Generated images", fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()