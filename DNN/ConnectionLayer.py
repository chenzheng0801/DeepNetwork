import numpy as np
from numba import jit


class FullConnection(object):
    def __init__(self, input_size, output_size, activator, derivative, input_vec=None):
        self.activator = activator
        self.input_size = input_size
        self.output_size = output_size
        self.derivative = derivative
        l = -np.sqrt(6.0/(output_size+input_size))
        r = np.sqrt(6.0/(output_size+input_size))
        # l = -0.8
        # r = 0.8
        self.weight = np.random.uniform(l, r, (input_size, output_size))
        self.b = np.random.uniform(l, r, (1, output_size))
        self.acc_grad_weight = 0
        self.acc_grad_b = 0
        self.input = input_vec

    @jit(forceobj=True)
    def do_inference(self, input_vec):
        output = np.add(np.dot(input_vec, self.weight), self.b)
        output = self.activator(output)
        self.input = input_vec
        return output

    @jit(forceobj=True)
    def foward(self, input_vec):
        output = np.add(np.dot(input_vec, self.weight), self.b)
        output = self.activator(output)
        self.input = input_vec
        return output

    @jit(forceobj=True)
    def backward(self, delta, derivative, learning_rate, batch):
        ret_delta = derivative(self.input) * np.dot(delta, self.weight.T)

        batch_weight = np.einsum("Bi,Bj->Bij", self.input, delta)
        self.b += learning_rate * (np.sum(delta, axis=0, keepdims=True) / batch)
        self.weight += learning_rate * (np.sum(batch_weight, axis=0) / batch)
        return ret_delta

    @jit(forceobj=True)
    def momentum_backward(self, delta, derivative, learning_rate, batch, momentum):
        ret_delta = derivative(self.input) * np.dot(delta, self.weight.T)
        batch_weight = np.einsum("Bi,Bj->Bij", self.input, delta)

        self.acc_grad_weight = momentum * self.acc_grad_weight\
                               + learning_rate * (np.sum(batch_weight, axis=0) / batch)
        self.acc_grad_b = momentum * self.acc_grad_b\
                          + learning_rate * (np.sum(delta, axis=0, keepdims=True) / batch)
        self.b += self.acc_grad_b
        self.weight += self.acc_grad_weight
        return ret_delta


class BatchNorm(object):
    def __init__(self, input_size, output_size, activator, derivative, input_vec=None):
        self.activator = activator
        self.input_size = input_size
        self.output_size = output_size
        self.derivative = derivative
        l = -np.sqrt(6.0/(output_size+input_size))
        r = np.sqrt(6.0/(output_size+input_size))
        self.weight = np.random.uniform(l, r, (input_size, output_size))
        self.b = np.random.uniform(l, r, (1, output_size))
        self.runing_mean = 0.0
        self.runing_var = 0.0
        self.sample_mean = 0.0
        self.sample_var_eps = 0.0
        self.std_inp = 0.0
        self.beta = np.ones((1, output_size))
        self.gamma = np.zeros((1, output_size))
        self.inp = input_vec

    @jit(forceobj=True)
    def foward(self, input_vec):
        eps = 1e-5
        momentum = 0.9
        linear_output = np.add(np.dot(input_vec, self.weight), self.b)

        sample_mean = np.mean(linear_output, axis=0, keepdims=True)
        sample_var = np.var(linear_output, axis=0, keepdims=True)
        self.runing_mean = momentum * self.runing_mean + (1 - momentum) * sample_mean
        self.runing_var = momentum * self.runing_var + (1 - momentum) * sample_var
        self.sample_var_eps = sample_var + eps
        self.sample_mean = sample_mean

        self.std_inp = (linear_output - sample_mean) / np.sqrt(sample_var+eps)
        final_inp = self.beta * self.std_inp + self.gamma
        output = self.activator(final_inp)
        self.inp = input_vec
        return output

    @jit(forceobj=True)
    def do_inference(self, input_vec):
        eps = 1e-5
        linear_output = np.add(np.dot(input_vec, self.weight), self.b)
        std_inp = (linear_output - self.runing_mean) / np.sqrt(self.runing_var + eps)
        final_inp = self.beta * std_inp + self.gamma
        output = self.activator(final_inp)
        return output

    @jit(forceobj=True)
    def backward(self, delta, derivative, learning_rate, batch):
        eps = 1e-5
        delta_gamma = np.sum(delta, axis=0, keepdims=True)
        delta_beta = np.sum(self.std_inp * delta, axis=0, keepdims=True)
        self.gamma += learning_rate*delta_gamma
        self.beta += learning_rate*delta_beta
        delta_std_x = self.beta * delta

        delta_x = batch * delta_std_x - np.sum(delta_std_x, axis=0, keepdims=True)\
                  - self.std_inp*np.sum(delta_std_x * self.std_inp, axis=0, keepdims=True)
        delta_x *= (1.0 / batch) / np.sqrt(self.sample_var_eps)

        ret_delta = derivative(self.inp) * np.dot(delta_x, self.weight.T)
        batch_weight = np.einsum("Bi,Bj->Bij", self.inp, delta_x)
        self.b += learning_rate * (np.sum(delta, axis=0, keepdims=True) / batch)
        self.weight += learning_rate * (np.sum(batch_weight, axis=0) / batch)
        return ret_delta
