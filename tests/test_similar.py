from unittest import TestCase

import torch
import numpy as np

from keras_radam.backend import keras
from keras_radam.backend import backend as K
from keras_radam import RAdam

from .official import RAdam as OfficialRAdam


class TestSimilar(TestCase):

    @staticmethod
    def gen_torch_linear(w, b):
        linear = torch.nn.Linear(3, 5)
        linear.weight = torch.nn.Parameter(torch.Tensor(w.transpose().tolist()))
        linear.bias = torch.nn.Parameter(torch.Tensor(b.tolist()))
        return linear

    @staticmethod
    def gen_keras_linear(w, b, weight_decay=0.):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(input_shape=(3,), units=5, name='Dense'))
        model.get_layer('Dense').set_weights([w, b])
        model.compile(optimizer=RAdam(
            lr=1e-3,
            weight_decay=weight_decay,
        ), loss='mse')
        return model

    @staticmethod
    def gen_random_weights():
        return np.random.standard_normal((3, 5)), np.random.standard_normal((5,))

    def test_same(self):
        w, b = self.gen_random_weights()
        torch_linear = self.gen_torch_linear(w, b)
        keras_linear = self.gen_keras_linear(w, b, weight_decay=1e-3)
        w, b = self.gen_random_weights()
        criterion = torch.nn.MSELoss()
        optimizer = OfficialRAdam(torch_linear.parameters(), lr=1e-3, weight_decay=1e-3, eps=K.epsilon())
        for i in range(500):
            x = np.random.standard_normal((1, 3))
            y = np.dot(x, w) + b
            optimizer.zero_grad()
            y_hat = torch_linear(torch.Tensor(x.tolist()))
            loss = criterion(y_hat, torch.Tensor(y.tolist()))
            torch_loss = loss.tolist()
            loss.backward()
            optimizer.step()
            keras_loss = keras_linear.train_on_batch(x, y).tolist()
            print(i, torch_loss, keras_loss)
        self.assertLess(abs(torch_loss - keras_loss), 0.1)
        self.assertTrue(np.allclose(
            torch_linear.weight.detach().numpy().transpose(),
            keras_linear.get_weights()[0],
            atol=1e-2,
        ))
        self.assertTrue(np.allclose(
            torch_linear.bias.detach().numpy(),
            keras_linear.get_weights()[1],
            atol=1e-2,
        ))
