#!/usr/bin/env python

import tensorflow as tf

from model import Model

# pylint: disable=arguments-differ


class Simple128Convolution(Model):
  """
  Creates 128x128 images based on noise input.
  """

  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Simple128Convolution.Generator, self).__init__()

      self.fc1 = tf.keras.layers.Dense(16*16*64, use_bias=False)
      self.batchnorm0 = tf.keras.layers.BatchNormalization()

      initial_filters = 512

      self.conv0 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**0),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      self.conv1 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**1),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm2 = tf.keras.layers.BatchNormalization()

      self.conv2 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**2),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm3 = tf.keras.layers.BatchNormalization()

      self.conv3 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**3),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm4 = tf.keras.layers.BatchNormalization()

      self.conv4 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**4),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm5 = tf.keras.layers.BatchNormalization()

      self.conv5 = tf.keras.layers.Conv2DTranspose(3 if config.has_colored_target else 1,
          (5, 5), strides=(1, 1), padding="same", use_bias=False)

    def call(self, x, training=True):
      x = self.fc1(x)
      x = self.batchnorm0(x, training=training)
      x = tf.nn.relu(x)

      x = tf.reshape(x, shape=(-1, 16, 16, 64))

      x = self.conv0(x)
      x = self.batchnorm1(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv1(x)
      x = self.batchnorm2(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv2(x)
      x = self.batchnorm3(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv3(x)
      x = self.batchnorm4(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv4(x)
      x = self.batchnorm5(x, training=training)
      x = tf.nn.relu(x)

      x = tf.nn.tanh(self.conv5(x))
      return x

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(Simple128Convolution.Discriminator, self).__init__()

      initial_filters = 32

      self.conv1 = tf.keras.layers.Conv2D(int(initial_filters*2**0),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      self.conv2 = tf.keras.layers.Conv2D(int(initial_filters*2**1),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm2 = tf.keras.layers.BatchNormalization()

      self.conv3 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm3 = tf.keras.layers.BatchNormalization()

      self.conv4 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm4 = tf.keras.layers.BatchNormalization()

      self.dropout = tf.keras.layers.Dropout(0.3)
      self.flatten = tf.keras.layers.Flatten()
      self.fc1 = tf.keras.layers.Dense(config.discriminator_classes)

    def call(self, x, training=True):
      x = self.conv1(x)
      x = self.batchnorm1(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv2(x)
      x = self.batchnorm2(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv3(x)
      x = self.batchnorm3(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv4(x)
      x = self.batchnorm4(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.flatten(x)
      x = self.fc1(x)
      return x


class Simple256Convolution(Model):
  """
  Creates 256x256 images based on noise input.
  """

  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Simple256Convolution.Generator, self).__init__()

      self.fc1 = tf.keras.layers.Dense(16*16*64, use_bias=False)
      self.batchnorm0 = tf.keras.layers.BatchNormalization()

      initial_filters = 512

      self.conv0 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**0),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      self.conv1 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**1),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm2 = tf.keras.layers.BatchNormalization()

      self.conv2 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**2),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm3 = tf.keras.layers.BatchNormalization()

      self.conv3 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**3),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm4 = tf.keras.layers.BatchNormalization()

      self.conv4 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**4),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm5 = tf.keras.layers.BatchNormalization()

      self.conv5 = tf.keras.layers.Conv2DTranspose(3 if config.has_colored_target else 1,
          (5, 5), strides=(1, 1), padding="same", use_bias=False)

    def call(self, x, training=True):
      x = self.fc1(x)
      x = self.batchnorm0(x, training=training)
      x = tf.nn.relu(x)

      x = tf.reshape(x, shape=(-1, 16, 16, 64))

      x = self.conv0(x)
      x = self.batchnorm1(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv1(x)
      x = self.batchnorm2(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv2(x)
      x = self.batchnorm3(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv3(x)
      x = self.batchnorm4(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv4(x)
      x = self.batchnorm5(x, training=training)
      x = tf.nn.relu(x)

      x = tf.nn.tanh(self.conv5(x))
      return x

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(Simple256Convolution.Discriminator, self).__init__()

      initial_filters = 32

      self.conv1 = tf.keras.layers.Conv2D(int(initial_filters*2**0),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      self.conv2 = tf.keras.layers.Conv2D(int(initial_filters*2**1),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm2 = tf.keras.layers.BatchNormalization()

      self.conv3 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm3 = tf.keras.layers.BatchNormalization()

      self.conv4 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm4 = tf.keras.layers.BatchNormalization()

      self.dropout = tf.keras.layers.Dropout(0.3)
      self.flatten = tf.keras.layers.Flatten()
      self.fc1 = tf.keras.layers.Dense(config.discriminator_classes)

    def call(self, x, training=True):
      x = self.conv1(x)
      x = self.batchnorm1(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv2(x)
      x = self.batchnorm2(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv3(x)
      x = self.batchnorm3(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv4(x)
      x = self.batchnorm4(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.flatten(x)
      x = self.fc1(x)
      return x


class Simple480pConvolution(Model):
  """
  Creates 640x480 images based on noise input.
  """

  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Simple480pConvolution.Generator, self).__init__()

      self.fc1 = tf.keras.layers.Dense(15*20*64, use_bias=False)
      self.batchnorm0 = tf.keras.layers.BatchNormalization()

      initial_filters = 512

      self.conv0 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**0),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      self.conv1 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**1),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm2 = tf.keras.layers.BatchNormalization()

      self.conv2 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**2),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm3 = tf.keras.layers.BatchNormalization()

      self.conv3 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**3),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm4 = tf.keras.layers.BatchNormalization()

      self.conv4 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**4),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm5 = tf.keras.layers.BatchNormalization()

      self.conv5 = tf.keras.layers.Conv2DTranspose(3 if config.has_colored_target else 1,
          (5, 5), strides=(1, 1), padding="same", use_bias=False)

    def call(self, x, training=True):
      x = self.fc1(x)
      x = self.batchnorm0(x, training=training)
      x = tf.nn.relu(x)

      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      x = self.conv0(x)
      x = self.batchnorm1(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv1(x)
      x = self.batchnorm2(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv2(x)
      x = self.batchnorm3(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv3(x)
      x = self.batchnorm4(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv4(x)
      x = self.batchnorm5(x, training=training)
      x = tf.nn.relu(x)

      x = tf.nn.tanh(self.conv5(x))
      return x

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(Simple480pConvolution.Discriminator, self).__init__()

      initial_filters = 32

      self.conv1 = tf.keras.layers.Conv2D(int(initial_filters*2**0),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      self.conv2 = tf.keras.layers.Conv2D(int(initial_filters*2**1),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm2 = tf.keras.layers.BatchNormalization()

      self.conv3 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm3 = tf.keras.layers.BatchNormalization()

      self.conv4 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm4 = tf.keras.layers.BatchNormalization()

      self.dropout = tf.keras.layers.Dropout(0.3)
      self.flatten = tf.keras.layers.Flatten()
      self.fc1 = tf.keras.layers.Dense(config.discriminator_classes)

    def call(self, x, training=True):
      x = self.conv1(x)
      x = self.batchnorm1(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv2(x)
      x = self.batchnorm2(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv3(x)
      x = self.batchnorm3(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv4(x)
      x = self.batchnorm4(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.flatten(x)
      x = self.fc1(x)
      return x


class Dummy480p(Model):
  """
  Dummy model for 480p images.
  """

  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Dummy480p.Generator, self).__init__()

      self.fc1 = tf.keras.layers.Dense(15*20*1, use_bias=False)
      self.batchnorm0 = tf.keras.layers.BatchNormalization()

      initial_filters = 16

      self.conv0 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**0),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      self.conv1 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**1),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm2 = tf.keras.layers.BatchNormalization()

      self.conv2 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**2),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm3 = tf.keras.layers.BatchNormalization()

      self.conv3 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**3),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm4 = tf.keras.layers.BatchNormalization()

      self.conv4 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**4),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm5 = tf.keras.layers.BatchNormalization()

      self.conv5 = tf.keras.layers.Conv2DTranspose(3 if config.has_colored_target else 1,
          (5, 5), strides=(1, 1), padding="same", use_bias=False)

    def call(self, x, training=True):
      x = self.fc1(x)
      x = self.batchnorm0(x, training=training)
      x = tf.nn.relu(x)

      x = tf.reshape(x, shape=(-1, 15, 20, 1))

      x = self.conv0(x)
      x = self.batchnorm1(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv1(x)
      x = self.batchnorm2(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv2(x)
      x = self.batchnorm3(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv3(x)
      x = self.batchnorm4(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv4(x)
      x = self.batchnorm5(x, training=training)
      x = tf.nn.relu(x)

      x = tf.nn.tanh(self.conv5(x))
      return x

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(Dummy480p.Discriminator, self).__init__()

      initial_filters = 1

      self.conv1 = tf.keras.layers.Conv2D(int(initial_filters*2**0),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      self.conv2 = tf.keras.layers.Conv2D(int(initial_filters*2**1),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm2 = tf.keras.layers.BatchNormalization()

      self.conv3 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm3 = tf.keras.layers.BatchNormalization()

      self.conv4 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm4 = tf.keras.layers.BatchNormalization()

      self.dropout = tf.keras.layers.Dropout(0.3)
      self.flatten = tf.keras.layers.Flatten()
      self.fc1 = tf.keras.layers.Dense(config.discriminator_classes, use_bias=False)

    def call(self, x, training=True):
      x = self.conv1(x)
      x = self.batchnorm1(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv2(x)
      x = self.batchnorm2(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv3(x)
      x = self.batchnorm3(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv4(x)
      x = self.batchnorm4(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.flatten(x)
      x = self.fc1(x)
      return x


class Veggie480Convolution(Model):
  """
  Creates 480x480 images based on noise input.
  """

  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Veggie480Convolution.Generator, self).__init__()

      self.fc1 = tf.keras.layers.Dense(15*15*64, use_bias=False)
      self.batchnorm0 = tf.keras.layers.BatchNormalization()

      initial_filters = 512

      self.conv0 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**0),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      self.conv1 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**1),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm2 = tf.keras.layers.BatchNormalization()

      self.conv2 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**2),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm3 = tf.keras.layers.BatchNormalization()

      self.conv3 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**3),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm4 = tf.keras.layers.BatchNormalization()

      self.conv4 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**4),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm5 = tf.keras.layers.BatchNormalization()

      self.conv5 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**5),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm6 = tf.keras.layers.BatchNormalization()

      self.conv6 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**6),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm7 = tf.keras.layers.BatchNormalization()

      self.conv7 = tf.keras.layers.Conv2DTranspose(3 if config.has_colored_target else 1,
          (5, 5), strides=(1, 1), padding="same", use_bias=False)

    def call(self, x, training=True):
      x = self.fc1(x)
      x = self.batchnorm0(x, training=training)
      x = tf.nn.relu(x)

      x = tf.reshape(x, shape=(-1, 15, 15, 64))

      x = self.conv0(x)
      x = self.batchnorm1(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv1(x)
      x = self.batchnorm2(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv2(x)
      x = self.batchnorm3(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv3(x)
      x = self.batchnorm4(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv4(x)
      x = self.batchnorm5(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv5(x)
      x = self.batchnorm6(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv6(x)
      x = self.batchnorm7(x, training=training)
      x = tf.nn.relu(x)

      x = tf.nn.tanh(self.conv7(x))
      return x

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(Veggie480Convolution.Discriminator, self).__init__()

      initial_filters = 32

      self.conv1 = tf.keras.layers.Conv2D(int(initial_filters*2**0),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      self.conv2 = tf.keras.layers.Conv2D(int(initial_filters*2**1),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm2 = tf.keras.layers.BatchNormalization()

      self.conv3 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm3 = tf.keras.layers.BatchNormalization()

      self.conv4 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm4 = tf.keras.layers.BatchNormalization()

      self.dropout = tf.keras.layers.Dropout(0.3)
      self.flatten = tf.keras.layers.Flatten()
      self.fc1 = tf.keras.layers.Dense(config.discriminator_classes)

    def call(self, x, training=True):
      x = self.conv1(x)
      x = self.batchnorm1(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv2(x)
      x = self.batchnorm2(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv3(x)
      x = self.batchnorm3(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv4(x)
      x = self.batchnorm4(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.flatten(x)
      x = self.fc1(x)
      return x
