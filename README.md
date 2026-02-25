# Generative Models from scratch

This is a personal project I built for fun and to learn the math. Instead of relying on PyTorch or TensorFlow, I wrote a neural network engine entirely from scratch using [CuPy](https://cupy.dev/) (NumPy but written to run on the GPU).

This project does have the help of Gemini, but I still wrote most of the code myself.

## What's Inside

* **No high-level ML libraries:** Just 67 hours of pure suffering. (Why am I doing this to myself?)
* **PyTorch-like OOP Architecture:** So it turns out I do need to learn OOP to be able to code a somewhat readable codebase.
* **Custom Layers:** Hand-written `Conv2D`, `TransposeConv2D`, and `Dense` layers with custom forward and backward passes.
* **Optimized Convolutions:** Implemented `im2col` to accelerate Convolution operations.
* **Custom Optimizers:** Built `SGD`, `Momentum`, and `Adam` optimizer. (Because wow GAN is just that unstable to train with SGD alone)
* **The Models:**
   * Variational Autoencoder (VAE)
   * Generative Adversarial Network (GAN)
   * more in the future...

## Files

* `utils.py`: Math helper functions and model state save/load mechanics.
* `data.py`: CIFAR-10 data loader class (and EMNIST data loader class too but teehee).
* `layers.py`: Neural layer classes and loss calculation classes.
* `optimizer.py`: The gradient descent engines (`SGD`, `Momentum`, `Adam`).
* `models.py`: The architecture for the `VariationalAutoEncoder`, `Generator`, and `Discriminator`.

## Training Scripts

* `VAE.py`: Trains the Variational Autoencoder.
* `GAN.py`: Trains the Generative Adversarial Network.
* `vae_weights.pkl` / `gan_weights.pkl`: Pre-trained model states so you don't need to train the model yourself to see the result.

## Results

* **VAE:**
  
<img width="1218" height="985" alt="image" src="https://github.com/user-attachments/assets/db2b76d1-7ca3-4822-9b96-65fb3d23b85a" />

* **GAN:**

[![Screenshot-from-2026-02-24-09-39-50.png](https://i.postimg.cc/Pr4qXJzK/Screenshot-from-2026-02-24-09-39-50.png)](https://postimg.cc/fJVs5wP0)

Yeah these looks like abstract art with striping on the GAN model, not too proud of it but the models did work.
## How to Run it

**Note:** I ran this code on Ubuntu 22.04.4 LTS. You will need an NVIDIA GPU with CUDA installed to run CuPy. Alternatively, you can modify the imports to `import numpy as cp` to run on the CPU (though it will be significantly slower).

1. **Install dependencies:**
   ```bash
   pip install cupy-cuda11x numpy pandas matplotlib
   ```
3. **Download the CIFAR-10 dataset:**
   Run these commands in your terminal to fetch and extract the official Python dataset directly into the project folder:
   ```bash
   wget [https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
   tar -xzf cifar-10-python.tar.gz
   ```
4. **Run the training loops:**
   ```bash
   python3 VAE.py
   # OR
   python3 GAN.py
   ```
   
## Log
* 21/2/2026: Oh the VAE worked great, I should upload this to GitHub and look for a new model.
* 23/2/2026:
   - Oh I don't actually know anything. People just kinda swept all the math under the rug unless you look for it very hard. Well back to learning grad level Statistics and Probalibity ig. Also my code is terrible, you should not copy my way of doing OOP at all. I guess I still upload this GAN model regardless because after it did work well enough.
   - Knowing how it works is one thing. Proving why they works is another entire thing. Which it's kinda on me for not even suspecting some of the equations I used along the way. Like you know how the loss function of Linear Regression can seem straightforward at first, but deriving it from probalibity is not so much so.
