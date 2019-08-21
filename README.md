# Keras RAdam

[![Travis](https://travis-ci.org/CyberZHG/keras-radam.svg)](https://travis-ci.org/CyberZHG/keras-radam)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-radam/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-radam)
[![Version](https://img.shields.io/pypi/v/keras-rectified-adam.svg)](https://pypi.org/project/keras-rectified-adam/)
![Downloads](https://img.shields.io/pypi/dm/keras-rectified-adam.svg)
![License](https://img.shields.io/pypi/l/keras-rectified-adam.svg)

![](https://img.shields.io/badge/keras-tensorflow-blue.svg)
![](https://img.shields.io/badge/keras-theano-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras/eager-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras/2.0_beta-blue.svg)

\[[中文](https://github.com/CyberZHG/keras-radam/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-radam/blob/master/README.md)\]

Unofficial implementation of [RAdam](https://arxiv.org/pdf/1908.03265v1.pdf) in Keras and TensorFlow. 

## Install

```bash
pip install keras-rectified-adam
```

## Usage

```python
import keras
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

### TensorFlow without Keras

```python
from keras_radam.training import RAdamOptimizer

RAdamOptimizer(learning_rate=1e-3)
```

### Use Warmup

```python
from keras_radam import RAdam

RAdam(total_steps=10000, warmup_proportion=0.1, min_lr=1e-5)
```

## Q & A

### About Correctness

The optimizer produces similar losses and weights to the official optimizer after 500 steps.

### Use `tf.keras` or `tf-2.0`

Add `TF_KERAS=1` to environment variables to use `tensorflow.python.keras`.

### Use `theano` Backend

Add `KERAS_BACKEND=theano` to environment variables to enable `theano` backend.
