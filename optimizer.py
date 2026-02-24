import cupy as cp
import numpy as np

class SGD():
    def __init__(self, layers, lr=0.001):
        self.layers = layers
        self.lr = lr

    def step(self):
        for layer in self.layers:
            if hasattr(layer, 'get_params'):
                for p, dp in layer.get_params():
                    param = getattr(layer, p)
                    grad = getattr(layer, dp)

                    param -= self.lr * grad

class Momentum():
    def __init__(self, layers, lr=0.0002, beta=0.9):
        self.layers = layers
        self.lr = lr
        self.beta = beta
        
        self.m = {}

        for layer in self.layers:
            if hasattr(layer, 'get_params'):
                l_id = id(layer)
                self.m[l_id] = {}
                for p, _ in layer.get_params():
                    param = getattr(layer, p)
                    self.m[l_id][p] = cp.zeros_like(param)

    def step(self):
        for layer in self.layers:
            if hasattr(layer, 'get_params'):
                l_id = id(layer)
                for p, dp in layer.get_params():
                    param = getattr(layer, p)
                    grad = getattr(layer, dp)

                    self.m[l_id][p] = self.beta * self.m[l_id][p] + (1 - self.beta) * grad
                    
                    param -= self.lr * self.m[l_id][p]

    def save_state(self):
        state = {'m': []}
        
        for layer in self.layers:
            if hasattr(layer, 'get_params'):
                l_id = id(layer)
                layer_m = {}
                for p, _ in layer.get_params():
                    layer_m[p] = cp.asnumpy(self.m[l_id][p])
                state['m'].append(layer_m)
                
        return state

    def load_state(self, state):
        layer_idx = 0
        
        for layer in self.layers:
            if hasattr(layer, 'get_params'):
                l_id = id(layer)
                for p, _ in layer.get_params():
                    self.m[l_id][p] = cp.array(state['m'][layer_idx][p])
                layer_idx += 1

class Adam():
    def __init__(self, layers, lr=0.0002, beta1=0.9, beta2=0.999, eps=1e-7):
        self.layers = layers
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        self.m = {}
        self.v = {}

        for layer in self.layers:
            if hasattr(layer, 'get_params'):
                l_id = id(layer)
                self.m[l_id] = {}
                self.v[l_id] = {}
                for p, _ in layer.get_params():
                    param = getattr(layer, p)
                    self.m[l_id][p] = cp.zeros_like(param)
                    self.v[l_id][p] = cp.zeros_like(param)

    def step(self):
        self.t += 1
        for layer in self.layers:
            if hasattr(layer, 'get_params'):
                l_id = id(layer)
                for p, dp in layer.get_params():
                    param = getattr(layer, p)
                    grad = getattr(layer, dp)
                    
                    self.m[l_id][p] = self.beta1 * self.m[l_id][p] + (1 - self.beta1) * grad
                    self.v[l_id][p] = self.beta2 * self.v[l_id][p] + (1 - self.beta2) * (grad ** 2)

                    m_hat =  self.m[l_id][p] / (1 - self.beta1 ** self.t)
                    v_hat =  self.v[l_id][p] / (1 - self.beta2 ** self.t)

                    param -= self.lr * m_hat / (cp.sqrt(v_hat) + self.eps)

    def save_state(self):
        state = {'t': self.t, 'm': [], 'v': []}
        
        for layer in self.layers:
            if hasattr(layer, 'get_params'):
                l_id = id(layer)
                layer_m = {}
                layer_v = {}
                for p, _ in layer.get_params():
                    layer_m[p] = cp.asnumpy(self.m[l_id][p])
                    layer_v[p] = cp.asnumpy(self.v[l_id][p])
                state['m'].append(layer_m)
                state['v'].append(layer_v)
                
        return state

    def load_state(self, state):
        self.t = state['t']
        layer_idx = 0
        
        for layer in self.layers:
            if hasattr(layer, 'get_params'):
                l_id = id(layer)
                for p, _ in layer.get_params():
                    self.m[l_id][p] = cp.array(state['m'][layer_idx][p])
                    self.v[l_id][p] = cp.array(state['v'][layer_idx][p])
                layer_idx += 1