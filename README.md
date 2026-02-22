# CuPy VAE From Scratch

This is a personal project I built to understand the math and behind deep learning. Instead of relying on PyTorch or TensorFlow, I wrote a custom neural network engine entirely from absolute scratch using [CuPy](https://cupy.dev/) (NumPy but written to run on GPU).

This project does have the help of Gemini, I still write most of the code however.

## What's Inside

* **No high-level ML libraries:** Just pure math and 67 hours of bashing my head against the wall.
* **Custom Layers:** Hand-written `Conv2D`, `TransposeConv2D`, and `Dense` layers.
* **Optimized Convolutions:** Implemented `im2col` and `col2im` to accelerate Convolution operations.
* **The Model:** A fully functional Convolutional Variational Autoencoder (VAE) with a KL-Divergence loss function.

## Files

* utils.py: Math helper functions
* data.py: Data loader classes
* layers.py: Neural layers classes (Conv2D, TransposeConv2D, Dense, ...)
* models.py: The VAE class.
* main.py: Training loop, model loader and image visualization
* vae_weights.pkl: A pre-trained model so that you don't need to train to see the results.

## Results

Here is what the custom VAE learnt to generate after training on CIFAR-10 for 100 epoches:

<img width="1218" height="985" alt="image" src="https://github.com/user-attachments/assets/db2b76d1-7ca3-4822-9b96-65fb3d23b85a" />

Yeah it looks more like abstract art than anything, but at least it does have some kinds of structure to it so nice ig.
## How to Run it

**Note:** You will need an NVIDIA GPU with CUDA installed to run CuPy. Or you can modify the import to "import numpy as cp", that works too.

1. Install dependencies: 
   ```bash
   pip install cupy numpy pandas matplotlib
   ```
2. Download the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) python dataset and extract it inside the same folder as the repo.

3. Run:
   ```bash
   python3 main.py
   ```
