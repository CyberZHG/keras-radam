import os
import tempfile
from unittest import TestCase

import numpy as np
from tensorflow import keras

from keras_radam import RAdam


class TestRAdam(TestCase):

    @staticmethod
    def gen_linear_model(optimizer) -> keras.models.Model:
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(
            input_shape=(17,),
            units=3,
            bias_constraint=keras.constraints.max_norm(),
            name='Dense',
        ))
        model.compile(optimizer, loss='mse')
        return model

    @staticmethod
    def gen_linear_data(w=None) -> (np.ndarray, np.ndarray):
        np.random.seed(0xcafe)
        x = np.random.standard_normal((4096 * 30, 17))
        if w is None:
            w = np.random.standard_normal((17, 3))
        y = np.dot(x, w)
        return x, y, w

    def _test_fit(self, optimizer, atol=1e-2):
        x, y, w = self.gen_linear_data()
        model = self.gen_linear_model(optimizer)

        callbacks = [keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=1e-8)]
        if isinstance(optimizer, RAdam):
            model_path = os.path.join(tempfile.gettempdir(), 'test_accumulation_%f.h5' % np.random.random())
            model.save(model_path)
            from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
            with CustomObjectScope({'RAdam': RAdam}):  # Workaround for incorrect global variable used in keras
                model = keras.models.load_model(model_path, custom_objects={'RAdam': RAdam})
            callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='loss', min_lr=1e-8, patience=2, verbose=True))

        model.fit(x, y,
                  epochs=100,
                  batch_size=32,
                  callbacks=callbacks)

        model_path = os.path.join(tempfile.gettempdir(), 'test_accumulation_%f.h5' % np.random.random())
        model.save(model_path)
        from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
        with CustomObjectScope({'RAdam': RAdam}):  # Workaround for incorrect global variable used in keras
            model = keras.models.load_model(model_path, custom_objects={'RAdam': RAdam})

        x, y, w = self.gen_linear_data(w)
        predicted = model.predict(x)
        self.assertLess(np.max(np.abs(predicted - y)), atol)

    def test_amsgrad(self):
        self._test_fit(RAdam(amsgrad=True))

    def test_decay(self):
        self._test_fit(RAdam(decay=1e-4, weight_decay=1e-6), atol=0.1)

    def test_warmup(self):
        self._test_fit(RAdam(total_steps=38400, warmup_proportion=0.1, min_lr=1e-6))

    def test_fit_embed(self):
        optimizers = [RAdam]
        for optimizer in optimizers:
            for amsgrad in [False, True]:
                model = keras.models.Sequential()
                model.add(keras.layers.Embedding(
                    input_shape=(None,),
                    input_dim=5,
                    output_dim=16,
                    mask_zero=True,
                ))
                model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=8)))
                model.add(keras.layers.Dense(units=2, activation='softmax'))
                model.compile(optimizer(
                    total_steps=38400,
                    warmup_proportion=0.1,
                    min_lr=1e-6,
                    weight_decay=1e-6,
                    amsgrad=amsgrad,
                ), loss='sparse_categorical_crossentropy')

                x = np.random.randint(0, 5, (64, 3))
                y = []
                for i in range(x.shape[0]):
                    if 2 in x[i]:
                        y.append(1)
                    else:
                        y.append(0)
                y = np.array(y)
                model.fit(x, y, epochs=10)

                model_path = os.path.join(tempfile.gettempdir(), 'test_accumulation_%f.h5' % np.random.random())
                model.save(model_path)
                from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
                with CustomObjectScope({'RAdam': RAdam}):  # Workaround for incorrect global variable used in keras
                    keras.models.load_model(model_path, custom_objects={'RAdam': RAdam})
