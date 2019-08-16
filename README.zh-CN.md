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

[RAdam](https://arxiv.org/pdf/1908.03265v1.pdf)的非官方实现。

## 安装

```bash
pip install keras-rectified-adam
```

## 使用

```python
import keras
import numpy as np
from keras_radam import RAdam

# 构建一个使用RAdam优化器的简单模型
model = keras.models.Sequential()
model.add(keras.layers.Dense(input_shape=(17,), units=3))
model.compile(RAdam(), loss='mse')

# 构建简单数据
x = np.random.standard_normal((4096 * 30, 17))
w = np.random.standard_normal((17, 3))
y = np.dot(x, w)

# 开始训练
model.fit(x, y, epochs=5)
```
