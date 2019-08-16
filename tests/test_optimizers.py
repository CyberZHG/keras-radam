import os
import tempfile
from unittest import TestCase

import numpy as np

from keras_radam.backend import keras
from keras_radam import RAdam


class TestRAdam(TestCase):

    @staticmethod
    def gen_linear_model() -> keras.models.Model:
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(
            input_shape=(17,),
            units=3,
            bias_constraint=keras.constraints.max_norm(),
            name='Dense',
        ))
        model.compile(RAdam(decay=1e-4, weight_decay=1e-4), loss='mse')
        return model

    @staticmethod
    def gen_linear_data(w=None) -> (np.ndarray, np.ndarray):
        np.random.seed(0xcafe)
        x = np.random.standard_normal((4096 * np.random.randint(20, 30), 17))
        if w is None:
            w = np.random.standard_normal((17, 3))
        y = np.dot(x, w)
        return x, y, w

    def test_fit(self):
        x, y, w = self.gen_linear_data()
        model = self.gen_linear_model()

        model_path = os.path.join(tempfile.gettempdir(), 'test_accumulation_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'RAdam': RAdam})

        model.fit(x, y,
                  epochs=100,
                  callbacks=[
                      keras.callbacks.ReduceLROnPlateau(monitor='loss', min_lr=1e-6, patience=2, verbose=True),
                      keras.callbacks.EarlyStopping(monitor='loss', patience=3),
                  ])

        model_path = os.path.join(tempfile.gettempdir(), 'test_accumulation_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'RAdam': RAdam})

        x, y, w = self.gen_linear_data(w)
        predicted = model.predict(x)
        self.assertLess(np.max(np.abs(predicted - y)), 1e-3)
