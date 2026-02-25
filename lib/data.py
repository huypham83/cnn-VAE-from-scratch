import cupy as cp
import numpy as np
import pandas as pd
import os
import pickle

class DataLoader():
    def __init__(self, file_path, batch_size):
        self.batch_size = batch_size
        df = pd.read_csv(file_path, header=None, dtype=np.float32)
        print('--Done Loading--')
        data = df.values
        self.labels = data[:, 0]
        self.pixels = data[:, 1:]
        self.pixels /= 255.0
        self.data = cp.array(self.pixels)

    def next_batch(self):
        indices = cp.random.permutation(len(self.data))
        shuffled_data = self.data[indices]

        num_batches = len(self.data) // self.batch_size

        for i in range(num_batches):
            start = i * self.batch_size
            end = start + self.batch_size

            batch_x = shuffled_data[start:end]

            yield batch_x, batch_x

class CIFAR10DataLoader():
    def __init__(self, folder_path, batch_size):
        self.batch_size = batch_size
        self.data = []

        print("Loading CIFAR-10 batches...")
        for i in range(1, 6):
            file_path = os.path.join(folder_path, f'data_batch_{i}')

            with open(file_path, 'rb') as fo:
                batch_dict = pickle.load(fo, encoding='latin1')
                self.data.append(batch_dict['data'])

        self.data = np.vstack(self.data).astype(np.float32)

        self.data /= 255.0

        self.data = cp.array(self.data)

        print(f"--Done Loading-- Total shape: {self.data.shape}")

    def next_batch(self):
        indices = cp.random.permutation(len(self.data))
        shuffled_data = self.data[indices]

        num_batches = len(self.data) // self.batch_size

        for i in range(num_batches):
            start = i * self.batch_size
            end = start + self.batch_size

            batch_x = shuffled_data[start:end]

            yield batch_x, batch_x
            
