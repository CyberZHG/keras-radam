# Keras RAdam

[![Version](https://img.shields.io/pypi/v/keras-rectified-adam.svg)](https://pypi.org/project/keras-rectified-adam/)
![License](https://img.shields.io/pypi/l/keras-rectified-adam.svg)

\[[中文](https://github.com/CyberZHG/keras-radam/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-radam/blob/master/README.md)\]

Unofficial implementation of [RAdam](https://arxiv.org/pdf/1908.03265v1.pdf) in Keras. 

## Install

```bash
pip install keras-rectified-adam
```

## External Link

- [tensorflow/addons:RectifiedAdam](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/rectified_adam.py)

## Usage

```python
from tensorflow import keras
import numpy as np
from keras_radam import RAdam

# Build toy model with RAdam optimizer
model = keras.models.Sequential()
model.add(keras.layers.Dense(input_shape=(17,), units=3))
model.compile(RAdam(), loss='mse')

# Generate toy data
x = np.random.standard_normal((4096 * 30, 17))
w = np.random.standard_normal((17, 3))
y = np.dot(x, w)

# Fit
model.fit(x, y, epochs=5)
```

### Use Warmup

```python
from keras_radam import RAdam

RAdam(total_steps=10000, warmup_proportion=0.1, min_lr=1e-5)
```
