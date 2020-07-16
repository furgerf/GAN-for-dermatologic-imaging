#!/usr/bin/env python

# pylint: disable=too-many-locals,arguments-differ,unused-import

import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                     Flatten, SpatialDropout2D)
from tensorflow.nn import leaky_relu, relu, tanh

from deep_model_blocks import (BottleneckResidualBlock, Conv, ConvBlock,
                               Deconv, DeconvBlock, ResidualBlock, ResizeBlock,
                               ReverseBottleneckResidualBlock,
                               ReverseResidualBlock, UBlock)
from model import Model


class Deep512NoiseMsDiscS2S1(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep512NoiseMsDiscS2S1.Generator, self).__init__()

      initial_filters = int(512/32)

      self.fc = tf.keras.layers.Dense(16*16*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          # default
          DeconvBlock(initial_filters*32, 5, 2),
          # ConvBlock(initial_filters*16, 5, 1),

          DeconvBlock(initial_filters*16, 5, 2),
          ConvBlock(initial_filters*8, 5, 1),

          DeconvBlock(initial_filters*8, 5, 2),
          ConvBlock(initial_filters*4, 5, 1),

          DeconvBlock(initial_filters*4, 5, 2),
          ConvBlock(initial_filters*2, 5, 1),

          DeconvBlock(initial_filters*2, 5, 2),
          # ConvBlock(initial_filters*1, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 16, 16, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout):
        super(Deep512NoiseMsDiscS2S1.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(512 * scaling_factor)
          size_y = int(512 * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32//2

        self.blocks = [
            ConvBlock(initial_filters*2, 5, 2),
            ConvBlock(initial_filters*1, 5, 1),

            ConvBlock(initial_filters*4, 5, 2),
            ConvBlock(initial_filters*2, 5, 1),

            ConvBlock(initial_filters*8, 5, 2),
            ConvBlock(initial_filters*4, 5, 1),

            ConvBlock(initial_filters*16, 5, 2),
            ConvBlock(initial_filters*8, 5, 1),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        self.fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        x = self.resize(x)
        for block in self.blocks:
          x = block(x, training=training)
          x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def __init__(self, config):
      super(Deep512NoiseMsDiscS2S1.Discriminator, self).__init__()
      self.discriminators = [Deep512NoiseMsDiscS2S1.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3)) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(Deep512NoiseMsDiscS2S1.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class Deep512NoiseMsDiscS2(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep512NoiseMsDiscS2.Generator, self).__init__()

      initial_filters = int(512/32)*1

      self.fc = tf.keras.layers.Dense(8*8*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          # default
          DeconvBlock(initial_filters*32, 5, 2),
          # ConvBlock(initial_filters*16, 5, 1),

          DeconvBlock(initial_filters*16, 5, 2),
          # ConvBlock(initial_filters*8, 5, 1),

          DeconvBlock(initial_filters*8, 5, 2),
          # ConvBlock(initial_filters*4, 5, 1),

          DeconvBlock(initial_filters*4, 5, 2),
          # ConvBlock(initial_filters*2, 5, 1),

          DeconvBlock(initial_filters*2, 5, 2),
          # ConvBlock(initial_filters*1, 5, 1),

          DeconvBlock(initial_filters*1, 5, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 8, 8, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout):
        super(Deep512NoiseMsDiscS2.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(512 * scaling_factor)
          size_y = int(512 * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32*1

        self.blocks = [
            ConvBlock(initial_filters*1, 4, 2),
            ConvBlock(initial_filters*2, 4, 2),
            ConvBlock(initial_filters*4, 4, 2),
            ConvBlock(initial_filters*8, 4, 2),
            ConvBlock(initial_filters*16, 4, 2),
            ConvBlock(initial_filters*32, 4, 2),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        self.fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        x = self.resize(x)
        for block in self.blocks:
          x = block(x, training=training)
          x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def __init__(self, config):
      super(Deep512NoiseMsDiscS2.Discriminator, self).__init__()
      self.discriminators = [Deep512NoiseMsDiscS2.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3)) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(Deep512NoiseMsDiscS2.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)

class Deep512NoiseMsDiscS2Alternative(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep512NoiseMsDiscS2Alternative.Generator, self).__init__()

      initial_filters = int(512/32)*1

      self.fc = tf.keras.layers.Dense(16*16*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          # default
          DeconvBlock(initial_filters*32, 5, 2),
          # ConvBlock(initial_filters*16, 5, 1),

          DeconvBlock(initial_filters*16, 5, 2),
          # ConvBlock(initial_filters*8, 5, 1),

          DeconvBlock(initial_filters*8, 5, 2),
          # ConvBlock(initial_filters*4, 5, 1),

          DeconvBlock(initial_filters*4, 5, 2),
          # ConvBlock(initial_filters*2, 5, 1),

          DeconvBlock(initial_filters*2, 5, 2),
          # ConvBlock(initial_filters*1, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 16, 16, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout):
        super(Deep512NoiseMsDiscS2Alternative.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(512 * scaling_factor)
          size_y = int(512 * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32*1

        self.blocks = [
            ConvBlock(initial_filters*1, 4, 2),
            ConvBlock(initial_filters*2, 4, 2),
            ConvBlock(initial_filters*4, 4, 2),
            ConvBlock(initial_filters*8, 4, 2),
            ConvBlock(initial_filters*16, 4, 2),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        self.fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        x = self.resize(x)
        for block in self.blocks:
          x = block(x, training=training)
          x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def __init__(self, config):
      super(Deep512NoiseMsDiscS2Alternative.Discriminator, self).__init__()
      self.discriminators = [Deep512NoiseMsDiscS2Alternative.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3)) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(Deep512NoiseMsDiscS2Alternative.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)

class Deep512NoiseS2(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep512NoiseS2.Generator, self).__init__()

      initial_filters = int(512/32)*1

      self.fc = tf.keras.layers.Dense(16*16*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          DeconvBlock(initial_filters*32, 5, 2),
          # ConvBlock(initial_filters*16, 5, 1),

          DeconvBlock(initial_filters*16, 5, 2),
          # ConvBlock(initial_filters*8, 5, 1),

          DeconvBlock(initial_filters*8, 5, 2),
          # ConvBlock(initial_filters*4, 5, 1),

          DeconvBlock(initial_filters*4, 5, 2),
          # ConvBlock(initial_filters*2, 5, 1),

          DeconvBlock(initial_filters*2, 5, 2),
          # ConvBlock(initial_filters*1, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 16, 16, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(Deep512NoiseS2.Discriminator, self).__init__()

      initial_filters = 32*1

      self.blocks = [
          ConvBlock(initial_filters*2, 5, 2),

          ConvBlock(initial_filters*4, 5, 2),

          ConvBlock(initial_filters*8, 5, 2),

          ConvBlock(initial_filters*16, 5, 2),

          ConvBlock(initial_filters*32, 5, 2),
          ]

      self.dropout = Dropout(0.3)
      self.flatten = Flatten()
      self.fc = Dense(config.discriminator_classes, use_bias=False)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
        x = self.dropout(x, training=training)
      x = self.flatten(x)
      x = self.fc(x)
      return x


class Deep256NoiseMsDiscS2S1(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep256NoiseMsDiscS2S1.Generator, self).__init__()

      initial_filters = int(512/32)

      self.fc = tf.keras.layers.Dense(8*8*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          DeconvBlock(initial_filters*32, 5, 2),
          # ConvBlock(initial_filters*16, 5, 1),

          DeconvBlock(initial_filters*16, 5, 2),
          ConvBlock(initial_filters*8, 5, 1),

          DeconvBlock(initial_filters*8, 5, 2),
          ConvBlock(initial_filters*4, 5, 1),

          DeconvBlock(initial_filters*4, 5, 2),
          ConvBlock(initial_filters*2, 5, 1),

          DeconvBlock(initial_filters*2, 5, 2),
          # ConvBlock(initial_filters*1, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 8, 8, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout):
        super(Deep256NoiseMsDiscS2S1.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(256 * scaling_factor)
          size_y = int(256 * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32//2

        self.blocks = [
            ConvBlock(initial_filters*2, 5, 2),
            ConvBlock(initial_filters*1, 5, 1),

            ConvBlock(initial_filters*4, 5, 2),
            ConvBlock(initial_filters*2, 5, 1),

            ConvBlock(initial_filters*8, 5, 2),
            ConvBlock(initial_filters*4, 5, 1),

            ConvBlock(initial_filters*16, 5, 2),
            ConvBlock(initial_filters*8, 5, 1),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        self.fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        x = self.resize(x)
        for block in self.blocks:
          x = block(x, training=training)
          x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def __init__(self, config):
      super(Deep256NoiseMsDiscS2S1.Discriminator, self).__init__()
      self.discriminators = [Deep256NoiseMsDiscS2S1.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3)) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(Deep256NoiseMsDiscS2S1.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class Deep256NoiseMsDiscS2(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep256NoiseMsDiscS2.Generator, self).__init__()

      initial_filters = int(512/32)*1

      self.fc = tf.keras.layers.Dense(8*8*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          DeconvBlock(initial_filters*32, 5, 2),
          # ConvBlock(initial_filters*32, 5, 1),

          DeconvBlock(initial_filters*16, 5, 2),
          # ConvBlock(initial_filters*16, 5, 1),

          DeconvBlock(initial_filters*8, 5, 2),
          # ConvBlock(initial_filters*8, 5, 1),

          DeconvBlock(initial_filters*4, 5, 2),
          # ConvBlock(initial_filters*4, 5, 1),

          DeconvBlock(initial_filters*2, 5, 2),
          # ConvBlock(initial_filters*2, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 8, 8, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout):
        super(Deep256NoiseMsDiscS2.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(256 * scaling_factor)
          size_y = int(256 * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32*1

        self.blocks = [
            ConvBlock(initial_filters*1, 4, 2),
            ConvBlock(initial_filters*2, 4, 2),
            ConvBlock(initial_filters*4, 4, 2),
            ConvBlock(initial_filters*8, 4, 2),
            ConvBlock(initial_filters*16, 4, 2),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        self.fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        x = self.resize(x)
        for block in self.blocks:
          x = block(x, training=training)
          x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def __init__(self, config):
      super(Deep256NoiseMsDiscS2.Discriminator, self).__init__()
      self.discriminators = [Deep256NoiseMsDiscS2.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3)) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(Deep256NoiseMsDiscS2.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class Deep256NoiseMsDiscS2Alternative(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep256NoiseMsDiscS2Alternative.Generator, self).__init__()

      initial_filters = int(512/32)*1

      self.fc = tf.keras.layers.Dense(8*8*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          DeconvBlock(initial_filters*32, 5, 2),
          # ConvBlock(initial_filters*16, 5, 1),

          DeconvBlock(initial_filters*16, 5, 2),
          # ConvBlock(initial_filters*8, 5, 1),

          DeconvBlock(initial_filters*8, 5, 2),
          # ConvBlock(initial_filters*4, 5, 1),

          DeconvBlock(initial_filters*4, 5, 2),
          # ConvBlock(initial_filters*2, 5, 1),

          DeconvBlock(initial_filters*2, 5, 2),
          # ConvBlock(initial_filters*1, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 8, 8, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout):
        super(Deep256NoiseMsDiscS2Alternative.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(256 * scaling_factor)
          size_y = int(256 * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32*1

        self.blocks = [
            ConvBlock(initial_filters*1, 4, 2),
            ConvBlock(initial_filters*2, 4, 2),
            ConvBlock(initial_filters*4, 4, 2),
            ConvBlock(initial_filters*8, 4, 2),
            ConvBlock(initial_filters*16, 4, 2),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        self.fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        x = self.resize(x)
        for block in self.blocks:
          x = block(x, training=training)
          x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def __init__(self, config):
      super(Deep256NoiseMsDiscS2Alternative.Discriminator, self).__init__()
      self.discriminators = [Deep256NoiseMsDiscS2Alternative.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3)) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(Deep256NoiseMsDiscS2Alternative.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class Deep256NoiseS2(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep256NoiseS2.Generator, self).__init__()

      initial_filters = int(512/32)*1

      self.fc = tf.keras.layers.Dense(8*8*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          DeconvBlock(initial_filters*32, 5, 2),
          # ConvBlock(initial_filters*16, 5, 1),

          DeconvBlock(initial_filters*16, 5, 2),
          # ConvBlock(initial_filters*8, 5, 1),

          DeconvBlock(initial_filters*8, 5, 2),
          # ConvBlock(initial_filters*4, 5, 1),

          DeconvBlock(initial_filters*4, 5, 2),
          # ConvBlock(initial_filters*2, 5, 1),

          DeconvBlock(initial_filters*2, 5, 2),
          # ConvBlock(initial_filters*1, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 8, 8, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(Deep256NoiseS2.Discriminator, self).__init__()

      initial_filters = 32*1

      self.blocks = [
          ConvBlock(initial_filters*2, 5, 2),

          ConvBlock(initial_filters*4, 5, 2),

          ConvBlock(initial_filters*8, 5, 2),

          ConvBlock(initial_filters*16, 5, 2),

          ConvBlock(initial_filters*32, 5, 2),
          ]

      self.dropout = Dropout(0.3)
      self.flatten = Flatten()
      self.fc = Dense(config.discriminator_classes, use_bias=False)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
        x = self.dropout(x, training=training)
      x = self.flatten(x)
      x = self.fc(x)
      return x


class Deep128NoiseMsDiscS2(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep128NoiseMsDiscS2.Generator, self).__init__()

      initial_filters = int(512/32)*1

      self.fc = tf.keras.layers.Dense(8*8*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          DeconvBlock(initial_filters*32, 5, 2),
          # ConvBlock(initial_filters*16*2, 5, 1),

          DeconvBlock(initial_filters*16, 5, 2),
          # ConvBlock(initial_filters*8*2, 5, 1),

          DeconvBlock(initial_filters*8, 5, 2),
          # ConvBlock(initial_filters*4*2, 5, 1),

          DeconvBlock(initial_filters*4, 5, 2),
          # ConvBlock(initial_filters*2*2, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 8, 8, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout):
        super(Deep128NoiseMsDiscS2.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(128 * scaling_factor)
          size_y = int(128 * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32*1

        self.blocks = [
            ConvBlock(initial_filters*1, 4, 2),
            ConvBlock(initial_filters*2, 4, 2),
            ConvBlock(initial_filters*4, 4, 2),
            ConvBlock(initial_filters*8, 4, 2),

            # ResidualBlock(initial_filters*1, 3, 2, project_shortcut=True),
            # ResidualBlock(initial_filters*2, 3, 2, project_shortcut=True),
            # ResidualBlock(initial_filters*4, 3, 2, project_shortcut=True),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        # self.pre_fc = Dense(1000, use_bias=False)
        self.fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        x = self.resize(x)
        for block in self.blocks:
          x = block(x, training=training)
          x = self.dropout(x, training=training)
        x = self.flatten(x)
        # x = self.pre_fc(x)
        x = self.fc(x)
        return x

    def __init__(self, config):
      super(Deep128NoiseMsDiscS2.Discriminator, self).__init__()
      self.discriminators = [Deep128NoiseMsDiscS2.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3)) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(Deep128NoiseMsDiscS2.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)

class Deep128NoiseMsDiscS2Alternative(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep128NoiseMsDiscS2Alternative.Generator, self).__init__()

      initial_filters = int(512/32)*1

      self.fc = tf.keras.layers.Dense(8*8*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          DeconvBlock(initial_filters*32, 5, 2),
          # ConvBlock(initial_filters*16*2, 5, 1),

          DeconvBlock(initial_filters*16, 5, 2),
          # ConvBlock(initial_filters*8*2, 5, 1),

          DeconvBlock(initial_filters*8, 5, 2),
          # ConvBlock(initial_filters*4*2, 5, 1),

          DeconvBlock(initial_filters*4, 5, 2),
          # ConvBlock(initial_filters*2*2, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 8, 8, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout):
        super(Deep128NoiseMsDiscS2Alternative.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(128 * scaling_factor)
          size_y = int(128 * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32*1

        self.blocks = [
            ConvBlock(initial_filters*1, 4, 2),
            ConvBlock(initial_filters*2, 4, 2),
            ConvBlock(initial_filters*4, 4, 2),
            ConvBlock(initial_filters*8, 4, 2),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        # self.pre_fc = Dense(1000, use_bias=False)
        self.fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        x = self.resize(x)
        for block in self.blocks:
          x = block(x, training=training)
          x = self.dropout(x, training=training)
        x = self.flatten(x)
        # x = self.pre_fc(x)
        x = self.fc(x)
        return x

    def __init__(self, config):
      super(Deep128NoiseMsDiscS2Alternative.Discriminator, self).__init__()
      self.discriminators = [Deep128NoiseMsDiscS2Alternative.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3)) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(Deep128NoiseMsDiscS2Alternative.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)

class Deep480NoiseS2(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480NoiseS2.Generator, self).__init__()

      initial_filters = int(512/32)*1

      self.fc = tf.keras.layers.Dense(15*15*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          DeconvBlock(initial_filters*32, 5, 2),
          ConvBlock(initial_filters*16, 5, 1),

          DeconvBlock(initial_filters*16, 5, 2),
          ConvBlock(initial_filters*8, 5, 1),

          DeconvBlock(initial_filters*8, 5, 2),
          ConvBlock(initial_filters*4, 5, 1),

          DeconvBlock(initial_filters*4, 5, 2),
          ConvBlock(initial_filters*2, 5, 1),

          DeconvBlock(initial_filters*2, 5, 2),
          ConvBlock(initial_filters*1, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 15, 15, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480NoiseS2.Discriminator, self).__init__()

      initial_filters = 32//2*1

      self.blocks = [
          ConvBlock(initial_filters*2, 4, 2),

          ConvBlock(initial_filters*4, 4, 2),

          ConvBlock(initial_filters*8, 4, 2),

          ConvBlock(initial_filters*16, 4, 2),

          ConvBlock(initial_filters*32, 4, 2),
          ]

      self.dropout = Dropout(0.3)
      self.flatten = Flatten()
      self.fc = Dense(config.discriminator_classes, use_bias=False)

    def call(self, x, training):
      for block in self.blocks:
        x = block(x, training=training)
        x = self.dropout(x, training=training)
      x = self.flatten(x)
      x = self.fc(x)
      return x

class Deep480NoiseMsDiscS2(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480NoiseMsDiscS2.Generator, self).__init__()

      initial_filters = int(512/32)*1

      self.fc = tf.keras.layers.Dense(15*15*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          DeconvBlock(initial_filters*32, 5, 2),
          ConvBlock(initial_filters*16, 5, 1),

          DeconvBlock(initial_filters*16, 5, 2),
          ConvBlock(initial_filters*8, 5, 1),

          DeconvBlock(initial_filters*8, 5, 2),
          ConvBlock(initial_filters*4, 5, 1),

          DeconvBlock(initial_filters*4, 5, 2),
          ConvBlock(initial_filters*2, 5, 1),

          DeconvBlock(initial_filters*2, 5, 2),
          ConvBlock(initial_filters*1, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 15, 15, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout):
        super(Deep480NoiseMsDiscS2.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(480 * scaling_factor)
          size_y = int(480 * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32//2*1

        self.blocks = [
            ConvBlock(initial_filters*2, 4, 2),

            ConvBlock(initial_filters*4, 4, 2),

            ConvBlock(initial_filters*8, 4, 2),

            ConvBlock(initial_filters*16, 4, 2),

            ConvBlock(initial_filters*32, 4, 2),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        self.fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        x = self.resize(x)
        for block in self.blocks:
          x = block(x, training=training)
          x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def __init__(self, config):
      super(Deep480NoiseMsDiscS2.Discriminator, self).__init__()
      self.discriminators = [Deep480NoiseMsDiscS2.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3)) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(Deep480NoiseMsDiscS2.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)
