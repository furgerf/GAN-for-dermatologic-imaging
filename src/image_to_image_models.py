#!/usr/bin/env python

import tensorflow as tf

from model import Model

# pylint: disable=arguments-differ,too-many-instance-attributes,too-many-lines,too-many-statements


class FromImage256Convolution(Model):
  """
  Translates between arbitrarily-sized images (so the name is selected rather poorly).
  """

  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(FromImage256Convolution.Generator, self).__init__()

      initial_filters = 512

      self.conv0 = tf.keras.layers.Conv2D(int(initial_filters/2**0),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      self.conv1 = tf.keras.layers.Conv2D(int(initial_filters/2**1),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm2 = tf.keras.layers.BatchNormalization()


      self.conv2 = tf.keras.layers.Conv2D(int(initial_filters/2**2),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm3 = tf.keras.layers.BatchNormalization()


      self.conv3 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**1),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm4 = tf.keras.layers.BatchNormalization()

      self.conv4 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**0),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm5 = tf.keras.layers.BatchNormalization()

      self.conv5 = tf.keras.layers.Conv2D(3 if config.has_colored_target else 1,
          (5, 5), strides=(1, 1), padding="same", use_bias=False)

    def call(self, x, training=True):
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
      super(FromImage256Convolution.Discriminator, self).__init__()

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


class OneStride(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(OneStride.Generator, self).__init__()

      initial_filters = 64

      self.conv0 = tf.keras.layers.Conv2D(int(initial_filters*2**0),
          (7, 7), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm0 = tf.keras.layers.BatchNormalization()

      self.conv1 = tf.keras.layers.Conv2D(int(initial_filters*2**1),
          (3, 3), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm1 = tf.keras.layers.BatchNormalization()


      self.conv2 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm2 = tf.keras.layers.BatchNormalization()

      self.conv3 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm3 = tf.keras.layers.BatchNormalization()

      self.conv4 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm4 = tf.keras.layers.BatchNormalization()

      self.conv5 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm5 = tf.keras.layers.BatchNormalization()

      self.conv6 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm6 = tf.keras.layers.BatchNormalization()

      self.conv7 = tf.keras.layers.Conv2DTranspose(int(initial_filters*2**2),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm7 = tf.keras.layers.BatchNormalization()

      self.conv8 = tf.keras.layers.Conv2DTranspose(int(initial_filters*2**2),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm8 = tf.keras.layers.BatchNormalization()


      self.conv9 = tf.keras.layers.Conv2DTranspose(int(initial_filters*2**1),
          (3, 3), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm9 = tf.keras.layers.BatchNormalization()

      self.conv10 = tf.keras.layers.Conv2D(3 if config.has_colored_target else 1,
          (7, 7), strides=(1, 1), padding="same", use_bias=False)

    def call(self, x, training=True):
      x = self.conv0(x)
      x = self.batchnorm0(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv1(x)
      x = self.batchnorm1(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv2(x)
      x = self.batchnorm2(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv3(x)
      x = self.batchnorm3(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv4(x)
      x = self.batchnorm4(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv5(x)
      x = self.batchnorm5(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv6(x)
      x = self.batchnorm6(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv7(x)
      x = self.batchnorm7(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv8(x)
      x = self.batchnorm8(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv9(x)
      x = self.batchnorm9(x, training=training)
      x = tf.nn.relu(x)

      x = tf.nn.tanh(self.conv10(x))
      return x

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(OneStride.Discriminator, self).__init__()

      # NOTE: this model without "better disc" had 32 initial filters and one less layer of conv/bn

      initial_filters = 64

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

      self.conv5 = tf.keras.layers.Conv2D(int(initial_filters*2**4),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm5 = tf.keras.layers.BatchNormalization()

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

      x = self.conv5(x)
      x = self.batchnorm5(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.flatten(x)
      x = self.fc1(x)
      return x


class ThreeStrides(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(ThreeStrides.Generator, self).__init__()

      initial_filters = 64

      self.conv0 = tf.keras.layers.Conv2D(int(initial_filters*2**0),
          (7, 7), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm0 = tf.keras.layers.BatchNormalization()

      self.conv1 = tf.keras.layers.Conv2D(int(initial_filters*2**1),
          (3, 3), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      self.conv2 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (3, 3), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm2 = tf.keras.layers.BatchNormalization()

      self.conv3 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (3, 3), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm3 = tf.keras.layers.BatchNormalization()


      self.conv4 = tf.keras.layers.Conv2D(int(initial_filters*2**4),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm4 = tf.keras.layers.BatchNormalization()

      self.conv5 = tf.keras.layers.Conv2D(int(initial_filters*2**4),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm5 = tf.keras.layers.BatchNormalization()

      self.conv6 = tf.keras.layers.Conv2D(int(initial_filters*2**4),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm6 = tf.keras.layers.BatchNormalization()


      self.conv7 = tf.keras.layers.Conv2DTranspose(int(initial_filters*2**3),
          (3, 3), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm7 = tf.keras.layers.BatchNormalization()

      self.conv8 = tf.keras.layers.Conv2DTranspose(int(initial_filters*2**2),
          (3, 3), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm8 = tf.keras.layers.BatchNormalization()

      self.conv9 = tf.keras.layers.Conv2DTranspose(int(initial_filters*2**1),
          (3, 3), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm9 = tf.keras.layers.BatchNormalization()

      self.conv10 = tf.keras.layers.Conv2D(3 if config.has_colored_target else 1,
          (7, 7), strides=(1, 1), padding="same", use_bias=False)

    def call(self, x, training=True):
      x = self.conv0(x)
      x = self.batchnorm0(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv1(x)
      x = self.batchnorm1(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv2(x)
      x = self.batchnorm2(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv3(x)
      x = self.batchnorm3(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv4(x)
      x = self.batchnorm4(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv5(x)
      x = self.batchnorm5(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv6(x)
      x = self.batchnorm6(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv7(x)
      x = self.batchnorm7(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv8(x)
      x = self.batchnorm8(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv9(x)
      x = self.batchnorm9(x, training=training)
      x = tf.nn.relu(x)

      x = tf.nn.tanh(self.conv10(x))
      return x

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(ThreeStrides.Discriminator, self).__init__()

      # NOTE: this model without "better disc" had 32 initial filters and one less layer of conv/bn

      initial_filters = 64

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

      self.conv5 = tf.keras.layers.Conv2D(int(initial_filters*2**4),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm5 = tf.keras.layers.BatchNormalization()

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

      x = self.conv5(x)
      x = self.batchnorm5(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.flatten(x)
      x = self.fc1(x)
      return x


class ThreeByThreeFilters(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(ThreeByThreeFilters.Generator, self).__init__()

      initial_filters = 512

      self.conv0 = tf.keras.layers.Conv2D(int(initial_filters/2**0),
          (3, 3), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      self.conv1 = tf.keras.layers.Conv2D(int(initial_filters/2**1),
          (3, 3), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm2 = tf.keras.layers.BatchNormalization()


      self.conv2 = tf.keras.layers.Conv2D(int(initial_filters/2**2),
          (7, 7), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm3 = tf.keras.layers.BatchNormalization()


      self.conv3 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**1),
          (7, 7), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm4 = tf.keras.layers.BatchNormalization()

      self.conv4 = tf.keras.layers.Conv2DTranspose(int(initial_filters/2**0),
          (3, 3), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm5 = tf.keras.layers.BatchNormalization()

      self.conv5 = tf.keras.layers.Conv2D(3 if config.has_colored_target else 1,
          (3, 3), strides=(1, 1), padding="same", use_bias=False)

    def call(self, x, training=True):
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
      super(ThreeByThreeFilters.Discriminator, self).__init__()

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


class DeeperFewerFilters(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(DeeperFewerFilters.Generator, self).__init__()

      initial_filters = 64

      self.conv0 = tf.keras.layers.Conv2D(int(initial_filters*2**0),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      self.conv1 = tf.keras.layers.Conv2D(int(initial_filters*2**1),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm2 = tf.keras.layers.BatchNormalization()

      self.conv2 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm3 = tf.keras.layers.BatchNormalization()


      self.conv3 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (3, 3), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm4 = tf.keras.layers.BatchNormalization()

      self.conv4 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (3, 3), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm5 = tf.keras.layers.BatchNormalization()

      self.conv5 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (3, 3), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm6 = tf.keras.layers.BatchNormalization()

      self.conv6 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (3, 3), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm7 = tf.keras.layers.BatchNormalization()

      self.conv7 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (3, 3), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm8 = tf.keras.layers.BatchNormalization()


      self.conv8 = tf.keras.layers.Conv2DTranspose(int(initial_filters*2**2),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm9 = tf.keras.layers.BatchNormalization()

      self.conv9 = tf.keras.layers.Conv2DTranspose(int(initial_filters*2**1),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm10 = tf.keras.layers.BatchNormalization()

      self.conv10 = tf.keras.layers.Conv2D(3 if config.has_colored_target else 1,
          (5, 5), strides=(1, 1), padding="same", use_bias=False)

    def call(self, x, training=True):
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

      x = self.conv7(x)
      x = self.batchnorm8(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv8(x)
      x = self.batchnorm9(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv9(x)
      x = self.batchnorm10(x, training=training)
      x = tf.nn.relu(x)

      x = tf.nn.tanh(self.conv10(x))
      return x

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(DeeperFewerFilters.Discriminator, self).__init__()

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


class DeepOneStrides(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(DeepOneStrides.Generator, self).__init__()

      initial_filters = 64

      self.conv0 = tf.keras.layers.Conv2D(int(initial_filters*2**0),
          (7, 7), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm0 = tf.keras.layers.BatchNormalization()


      self.conv1 = tf.keras.layers.Conv2D(int(initial_filters*2**1),
          (3, 3), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm1 = tf.keras.layers.BatchNormalization()


      self.conv2 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (3, 3), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm2 = tf.keras.layers.BatchNormalization()

      self.conv3 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (3, 3), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm3 = tf.keras.layers.BatchNormalization()

      self.conv4 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (3, 3), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm4 = tf.keras.layers.BatchNormalization()

      self.conv5 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (3, 3), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm5 = tf.keras.layers.BatchNormalization()

      self.conv6 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (3, 3), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm6 = tf.keras.layers.BatchNormalization()

      self.conv7 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (3, 3), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm7 = tf.keras.layers.BatchNormalization()

      self.conv8 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (3, 3), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm8 = tf.keras.layers.BatchNormalization()

      self.conv9 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (3, 3), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm9 = tf.keras.layers.BatchNormalization()


      self.conv10 = tf.keras.layers.Conv2DTranspose(int(initial_filters*2**1),
          (3, 3), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm10 = tf.keras.layers.BatchNormalization()


      self.conv11 = tf.keras.layers.Conv2D(3 if config.has_colored_target else 1,
          (7, 7), strides=(1, 1), padding="same", use_bias=False)

    def call(self, x, training=True):
      x = self.conv0(x)
      x = self.batchnorm0(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv1(x)
      x = self.batchnorm1(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv2(x)
      x = self.batchnorm2(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv3(x)
      x = self.batchnorm3(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv4(x)
      x = self.batchnorm4(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv5(x)
      x = self.batchnorm5(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv6(x)
      x = self.batchnorm6(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv7(x)
      x = self.batchnorm7(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv8(x)
      x = self.batchnorm8(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv9(x)
      x = self.batchnorm9(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv10(x)
      x = self.batchnorm10(x, training=training)
      x = tf.nn.relu(x)

      x = tf.nn.tanh(self.conv11(x))
      return x

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(DeepOneStrides.Discriminator, self).__init__()

      initial_filters = 64

      self.conv1 = tf.keras.layers.Conv2D(int(initial_filters*2**0),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      self.conv2 = tf.keras.layers.Conv2D(int(initial_filters*2**0),
          (5, 5), strides=(1, 1), padding="same")
      self.batchnorm2 = tf.keras.layers.BatchNormalization()

      self.conv3 = tf.keras.layers.Conv2D(int(initial_filters*2**1),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm3 = tf.keras.layers.BatchNormalization()

      self.conv4 = tf.keras.layers.Conv2D(int(initial_filters*2**1),
          (5, 5), strides=(1, 1), padding="same")
      self.batchnorm4 = tf.keras.layers.BatchNormalization()

      self.conv5 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm5 = tf.keras.layers.BatchNormalization()

      self.conv6 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (5, 5), strides=(1, 1), padding="same")
      self.batchnorm6 = tf.keras.layers.BatchNormalization()

      self.conv7 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm7 = tf.keras.layers.BatchNormalization()

      self.conv8 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(1, 1), padding="same")
      self.batchnorm8 = tf.keras.layers.BatchNormalization()

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

      x = self.conv5(x)
      x = self.batchnorm5(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv6(x)
      x = self.batchnorm6(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv7(x)
      x = self.batchnorm7(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv8(x)
      x = self.batchnorm8(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.flatten(x)
      x = self.fc1(x)
      return x


class DeepTwoStrides(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(DeepTwoStrides.Generator, self).__init__()

      initial_filters = 64

      self.conv0 = tf.keras.layers.Conv2D(int(initial_filters*2**0),
          (7, 7), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm0 = tf.keras.layers.BatchNormalization()


      self.conv1 = tf.keras.layers.Conv2D(int(initial_filters*2**1),
          (3, 3), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      self.conv2 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (3, 3), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm2 = tf.keras.layers.BatchNormalization()


      self.conv3 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm3 = tf.keras.layers.BatchNormalization()

      self.conv4 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm4 = tf.keras.layers.BatchNormalization()

      self.conv5 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm5 = tf.keras.layers.BatchNormalization()

      self.conv6 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm6 = tf.keras.layers.BatchNormalization()

      self.conv7 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm7 = tf.keras.layers.BatchNormalization()

      self.conv8 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm8 = tf.keras.layers.BatchNormalization()


      self.conv9 = tf.keras.layers.Conv2DTranspose(int(initial_filters*2**2),
          (3, 3), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm9 = tf.keras.layers.BatchNormalization()

      self.conv10 = tf.keras.layers.Conv2DTranspose(int(initial_filters*2**1),
          (3, 3), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm10 = tf.keras.layers.BatchNormalization()


      self.conv11 = tf.keras.layers.Conv2D(3 if config.has_colored_target else 1,
          (7, 7), strides=(1, 1), padding="same", use_bias=False)

    def call(self, x, training=True):
      x = self.conv0(x)
      x = self.batchnorm0(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv1(x)
      x = self.batchnorm1(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv2(x)
      x = self.batchnorm2(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv3(x)
      x = self.batchnorm3(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv4(x)
      x = self.batchnorm4(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv5(x)
      x = self.batchnorm5(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv6(x)
      x = self.batchnorm6(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv7(x)
      x = self.batchnorm7(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv8(x)
      x = self.batchnorm8(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv9(x)
      x = self.batchnorm9(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv10(x)
      x = self.batchnorm10(x, training=training)
      x = tf.nn.relu(x)

      x = tf.nn.tanh(self.conv11(x))
      return x

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(DeepTwoStrides.Discriminator, self).__init__()

      initial_filters = 64

      self.conv1 = tf.keras.layers.Conv2D(int(initial_filters*2**0),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      self.conv2 = tf.keras.layers.Conv2D(int(initial_filters*2**0),
          (5, 5), strides=(1, 1), padding="same")
      self.batchnorm2 = tf.keras.layers.BatchNormalization()

      self.conv3 = tf.keras.layers.Conv2D(int(initial_filters*2**1),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm3 = tf.keras.layers.BatchNormalization()

      self.conv4 = tf.keras.layers.Conv2D(int(initial_filters*2**1),
          (5, 5), strides=(1, 1), padding="same")
      self.batchnorm4 = tf.keras.layers.BatchNormalization()

      self.conv5 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm5 = tf.keras.layers.BatchNormalization()

      self.conv6 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (5, 5), strides=(1, 1), padding="same")
      self.batchnorm6 = tf.keras.layers.BatchNormalization()

      self.conv7 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm7 = tf.keras.layers.BatchNormalization()

      self.conv8 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(1, 1), padding="same")
      self.batchnorm8 = tf.keras.layers.BatchNormalization()

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

      x = self.conv5(x)
      x = self.batchnorm5(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv6(x)
      x = self.batchnorm6(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv7(x)
      x = self.batchnorm7(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv8(x)
      x = self.batchnorm8(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.flatten(x)
      x = self.fc1(x)
      return x


class DeepThreeStrides(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(DeepThreeStrides.Generator, self).__init__()

      initial_filters = 64

      self.conv0 = tf.keras.layers.Conv2D(int(initial_filters*2**0),
          (7, 7), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm0 = tf.keras.layers.BatchNormalization()


      self.conv1 = tf.keras.layers.Conv2D(int(initial_filters*2**1),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      self.conv2 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm2 = tf.keras.layers.BatchNormalization()

      self.conv3 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm3 = tf.keras.layers.BatchNormalization()


      self.conv4 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm4 = tf.keras.layers.BatchNormalization()

      self.conv5 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm5 = tf.keras.layers.BatchNormalization()

      self.conv6 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm6 = tf.keras.layers.BatchNormalization()

      self.conv7 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm7 = tf.keras.layers.BatchNormalization()


      self.conv8 = tf.keras.layers.Conv2DTranspose(int(initial_filters*2**3),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm8 = tf.keras.layers.BatchNormalization()

      self.conv9 = tf.keras.layers.Conv2DTranspose(int(initial_filters*2**2),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm9 = tf.keras.layers.BatchNormalization()

      self.conv10 = tf.keras.layers.Conv2DTranspose(int(initial_filters*2**1),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm10 = tf.keras.layers.BatchNormalization()


      self.conv11 = tf.keras.layers.Conv2D(3 if config.has_colored_target else 1,
          (7, 7), strides=(1, 1), padding="same", use_bias=False)

    def call(self, x, training=True):
      x = self.conv0(x)
      x = self.batchnorm0(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv1(x)
      x = self.batchnorm1(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv2(x)
      x = self.batchnorm2(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv3(x)
      x = self.batchnorm3(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv4(x)
      x = self.batchnorm4(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv5(x)
      x = self.batchnorm5(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv6(x)
      x = self.batchnorm6(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv7(x)
      x = self.batchnorm7(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv8(x)
      x = self.batchnorm8(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv9(x)
      x = self.batchnorm9(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv10(x)
      x = self.batchnorm10(x, training=training)
      x = tf.nn.relu(x)

      x = tf.nn.tanh(self.conv11(x))
      return x

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(DeepThreeStrides.Discriminator, self).__init__()

      initial_filters = 64

      self.conv1 = tf.keras.layers.Conv2D(int(initial_filters*2**0),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      self.conv2 = tf.keras.layers.Conv2D(int(initial_filters*2**0),
          (5, 5), strides=(1, 1), padding="same")
      self.batchnorm2 = tf.keras.layers.BatchNormalization()

      self.conv3 = tf.keras.layers.Conv2D(int(initial_filters*2**1),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm3 = tf.keras.layers.BatchNormalization()

      self.conv4 = tf.keras.layers.Conv2D(int(initial_filters*2**1),
          (5, 5), strides=(1, 1), padding="same")
      self.batchnorm4 = tf.keras.layers.BatchNormalization()

      self.conv5 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm5 = tf.keras.layers.BatchNormalization()

      self.conv6 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (5, 5), strides=(1, 1), padding="same")
      self.batchnorm6 = tf.keras.layers.BatchNormalization()

      self.conv7 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm7 = tf.keras.layers.BatchNormalization()

      self.conv8 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(1, 1), padding="same")
      self.batchnorm8 = tf.keras.layers.BatchNormalization()

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

      x = self.conv5(x)
      x = self.batchnorm5(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv6(x)
      x = self.batchnorm6(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv7(x)
      x = self.batchnorm7(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv8(x)
      x = self.batchnorm8(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.flatten(x)
      x = self.fc1(x)
      return x


class DeepFourStrides(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(DeepFourStrides.Generator, self).__init__()

      initial_filters = 64

      self.conv0 = tf.keras.layers.Conv2D(int(initial_filters*2**0),
          (7, 7), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm0 = tf.keras.layers.BatchNormalization()


      self.conv1 = tf.keras.layers.Conv2D(int(initial_filters*2**1),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      self.conv2 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm2 = tf.keras.layers.BatchNormalization()

      self.conv3 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm3 = tf.keras.layers.BatchNormalization()

      self.conv4 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm4 = tf.keras.layers.BatchNormalization()


      self.conv5 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm5 = tf.keras.layers.BatchNormalization()

      self.conv6 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm6 = tf.keras.layers.BatchNormalization()


      self.conv7 = tf.keras.layers.Conv2DTranspose(int(initial_filters*2**3),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm7 = tf.keras.layers.BatchNormalization()

      self.conv8 = tf.keras.layers.Conv2DTranspose(int(initial_filters*2**3),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm8 = tf.keras.layers.BatchNormalization()

      self.conv9 = tf.keras.layers.Conv2DTranspose(int(initial_filters*2**2),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm9 = tf.keras.layers.BatchNormalization()

      self.conv10 = tf.keras.layers.Conv2DTranspose(int(initial_filters*2**1),
          (5, 5), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm10 = tf.keras.layers.BatchNormalization()


      self.conv11 = tf.keras.layers.Conv2D(3 if config.has_colored_target else 1,
          (7, 7), strides=(1, 1), padding="same", use_bias=False)

    def call(self, x, training=True):
      x = self.conv0(x)
      x = self.batchnorm0(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv1(x)
      x = self.batchnorm1(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv2(x)
      x = self.batchnorm2(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv3(x)
      x = self.batchnorm3(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv4(x)
      x = self.batchnorm4(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv5(x)
      x = self.batchnorm5(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv6(x)
      x = self.batchnorm6(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv7(x)
      x = self.batchnorm7(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv8(x)
      x = self.batchnorm8(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv9(x)
      x = self.batchnorm9(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv10(x)
      x = self.batchnorm10(x, training=training)
      x = tf.nn.relu(x)

      x = tf.nn.tanh(self.conv11(x))
      return x

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(DeepFourStrides.Discriminator, self).__init__()

      initial_filters = 64

      self.conv1 = tf.keras.layers.Conv2D(int(initial_filters*2**0),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      self.conv2 = tf.keras.layers.Conv2D(int(initial_filters*2**0),
          (5, 5), strides=(1, 1), padding="same")
      self.batchnorm2 = tf.keras.layers.BatchNormalization()

      self.conv3 = tf.keras.layers.Conv2D(int(initial_filters*2**1),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm3 = tf.keras.layers.BatchNormalization()

      self.conv4 = tf.keras.layers.Conv2D(int(initial_filters*2**1),
          (5, 5), strides=(1, 1), padding="same")
      self.batchnorm4 = tf.keras.layers.BatchNormalization()

      self.conv5 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm5 = tf.keras.layers.BatchNormalization()

      self.conv6 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (5, 5), strides=(1, 1), padding="same")
      self.batchnorm6 = tf.keras.layers.BatchNormalization()

      self.conv7 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm7 = tf.keras.layers.BatchNormalization()

      self.conv8 = tf.keras.layers.Conv2D(int(initial_filters*2**3),
          (5, 5), strides=(1, 1), padding="same")
      self.batchnorm8 = tf.keras.layers.BatchNormalization()

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

      x = self.conv5(x)
      x = self.batchnorm5(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv6(x)
      x = self.batchnorm6(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv7(x)
      x = self.batchnorm7(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.conv8(x)
      x = self.batchnorm8(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.dropout(x, training=training)

      x = self.flatten(x)
      x = self.fc1(x)
      return x


class DummyImageToImage(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(DummyImageToImage.Generator, self).__init__()

      initial_filters = 2

      self.conv0 = tf.keras.layers.Conv2D(int(initial_filters*2**0),
          (7, 7), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm0 = tf.keras.layers.BatchNormalization()

      self.conv1 = tf.keras.layers.Conv2D(int(initial_filters*2**1),
          (3, 3), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm1 = tf.keras.layers.BatchNormalization()


      self.conv2 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (5, 5), strides=(1, 1), padding="same", use_bias=False)
      self.batchnorm2 = tf.keras.layers.BatchNormalization()


      self.conv9 = tf.keras.layers.Conv2DTranspose(int(initial_filters*2**1),
          (3, 3), strides=(2, 2), padding="same", use_bias=False)
      self.batchnorm9 = tf.keras.layers.BatchNormalization()

      self.conv10 = tf.keras.layers.Conv2D(3 if config.has_colored_target else 1,
          (7, 7), strides=(1, 1), padding="same", use_bias=False)

    def call(self, x, training=True):
      x = self.conv0(x)
      x = self.batchnorm0(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv1(x)
      x = self.batchnorm1(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv2(x)
      x = self.batchnorm2(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv9(x)
      x = self.batchnorm9(x, training=training)
      x = tf.nn.relu(x)

      x = tf.nn.tanh(self.conv10(x))
      return x

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(DummyImageToImage.Discriminator, self).__init__()

      initial_filters = 2

      self.conv1 = tf.keras.layers.Conv2D(int(initial_filters*2**0),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      self.conv2 = tf.keras.layers.Conv2D(int(initial_filters*2**1),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm2 = tf.keras.layers.BatchNormalization()

      self.conv3 = tf.keras.layers.Conv2D(int(initial_filters*2**2),
          (5, 5), strides=(2, 2), padding="same")
      self.batchnorm3 = tf.keras.layers.BatchNormalization()

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

      x = self.flatten(x)
      x = self.fc1(x)
      return x
