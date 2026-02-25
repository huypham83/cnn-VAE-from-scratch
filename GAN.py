import pickle
import os
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from lib.models import Generator, Discriminator
from lib.layers import BCELoss
from lib.data import CIFAR10DataLoader
from lib.optimizer import SGD, Momentum, Adam
from lib.utils import *

latent_dim = 128
batch_size = 32
lr = 0.0002
epochs = 50
train = False

file_name = 'Weight/gan_weights.pkl'
generator = Generator(latent_dim=latent_dim, image_height=32, image_width=32, image_channel=3)
discriminator = Discriminator(image_height=32, image_width=32, image_channel=3)
g_optimizer = Adam(generator.get_layer(), lr=lr, beta1=0.5, beta2=0.999)
d_optimizer = Adam(discriminator.get_layer(), lr=lr, beta1=0.5, beta2=0.999)

if os.path.exists(file_name):
    print(f"Found existing weights at {file_name}. Loading model...")
    with open(file_name, 'rb') as f:
        trained_weights = pickle.load(f)

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

    loader = CIFAR10DataLoader('./dataset/CIFAR-10', batch_size)
    loss_func = BCELoss()
    for epoch in range(epochs):
        d_loss_total = 0
        g_loss_total = 0
        batches = 0
        for x_batch, _ in loader.next_batch():
            current_batch_size = x_batch.shape[0]
            
            real_labels = cp.full((current_batch_size, 1), 0.9)
            fake_labels = cp.zeros((current_batch_size, 1))

            # Train D with real
            d_pred_real = discriminator.forward(x_batch)
            d_loss_real = loss_func.forward(d_pred_real, real_labels)
            grad = loss_func.backward(d_pred_real, real_labels)
            discriminator.backward(grad)
            d_optimizer.step()
            d_real_score = cp.mean(d_pred_real)

            # Train D with fake
            fakes = generator.forward(current_batch_size)
            d_pred_fake = discriminator.forward(fakes)
            d_loss_fake = loss_func.forward(d_pred_fake, fake_labels)
            grad = loss_func.backward(d_pred_fake, fake_labels)
            discriminator.backward(grad)
            d_optimizer.step()
            d_fake_score = cp.mean(d_pred_fake)

            d_loss_total += (d_loss_real + d_loss_fake) / 2

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
            model_weights = {
                'generator': extract_state(generator),
                'discriminator': extract_state(discriminator),
                'g_optim':g_optimizer.save_state(),
                'd_optim':d_optimizer.save_state()
            }

            with open(file_name, 'wb') as f:
                pickle.dump(model_weights, f)

            print(f"Model weights are saved to {file_name}.")

    print("Training complete")

num_samples = 20
cols = 5
rows = (num_samples + cols - 1) // cols

generated_images_np = cp.asnumpy(generator.forward(num_samples))

plt.figure(figsize=(cols * 2.5, rows * 2.5))
for i in range(num_samples):
    ax = plt.subplot(rows, cols, i + 1)

    plt.imshow(generated_images_np[i].reshape(3, 32, 32).transpose(1, 2, 0))
    plt.axis('off')
plt.suptitle("Generated images", fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()