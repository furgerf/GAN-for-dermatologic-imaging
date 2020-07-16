#!/usr/bin/env python

# pylint: disable=too-many-locals,arguments-differ,unused-import

import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                     Flatten, MaxPooling2D, SpatialDropout2D,
                                     add)
from tensorflow.nn import leaky_relu, relu, tanh

from deep_model_blocks import (BottleneckResidualBlock, Conv, ConvBlock,
                               Deconv, DeconvBlock, ResidualBlock, ResizeBlock,
                               ReverseBottleneckResidualBlock,
                               ReverseResidualBlock, UBlock)
from model import Model


class Deep480pNoise(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoise.Generator, self).__init__()

      initial_filters = int(512/32)

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
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
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoise.Discriminator, self).__init__()

      initial_filters = 32

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


class Deep480pNoiseFancyFilters(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoiseFancyFilters.Generator, self).__init__()

      initial_filters = int(512/32)

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          DeconvBlock(initial_filters*32, 3, 2),
          # ConvBlock(initial_filters*16, 3, 1),

          DeconvBlock(initial_filters*16, 3, 2),
          ConvBlock(initial_filters*8, 3, 1),

          DeconvBlock(initial_filters*8, 7, 2),
          ConvBlock(initial_filters*4, 7, 1),

          DeconvBlock(initial_filters*4, 7, 2),
          ConvBlock(initial_filters*2, 7, 1),

          DeconvBlock(initial_filters*2, 7, 2),
          # ConvBlock(initial_filters*1, 7, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 7, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoiseFancyFilters.Discriminator, self).__init__()

      initial_filters = 32

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


class Deep480pNoiseThreeSteps(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoiseThreeSteps.Generator, self).__init__()

      initial_filters = 32

      self.fc_shape = (60, 80, 16)

      self.fc = tf.keras.layers.Dense(self.fc_shape[0]*self.fc_shape[1]*self.fc_shape[2], use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          DeconvBlock(initial_filters*4, 7, 2),
          ConvBlock(initial_filters*2, 3, 1),
          ConvBlock(initial_filters*2, 3, 1),

          DeconvBlock(initial_filters*2, 7, 2),
          ConvBlock(initial_filters*1, 3, 1),
          ConvBlock(initial_filters*1, 3, 1),

          DeconvBlock(initial_filters*1, 7, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 7, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, self.fc_shape[0], self.fc_shape[1], self.fc_shape[2]))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoiseThreeSteps.Discriminator, self).__init__()

      initial_filters = 32

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

class Deep480pNoiseSmallerGenLayer(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoiseSmallerGenLayer.Generator, self).__init__()

      initial_filters = int(512/32)

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          DeconvBlock(initial_filters*32, 5, 2),
          # ConvBlock(initial_filters*8, 5, 1),
          # ConvBlock(initial_filters*8, 5, 1),

          DeconvBlock(initial_filters*16, 5, 2),
          ConvBlock(initial_filters*4, 5, 1),
          ConvBlock(initial_filters*4, 5, 1),

          DeconvBlock(initial_filters*8, 5, 2),
          ConvBlock(initial_filters*2, 5, 1),
          ConvBlock(initial_filters*2, 5, 1),

          DeconvBlock(initial_filters*4, 5, 2),
          ConvBlock(initial_filters*1, 5, 1),
          ConvBlock(initial_filters*1, 5, 1),

          DeconvBlock(initial_filters*2, 5, 2),
          # ConvBlock(initial_filters*1, 5, 1),
          # ConvBlock(initial_filters*1, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoiseSmallerGenLayer.Discriminator, self).__init__()

      initial_filters = 32

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

class Deep480pNoiseSmallerGenLayerFancyFilters(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoiseSmallerGenLayerFancyFilters.Generator, self).__init__()

      initial_filters = int(512/32)

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          DeconvBlock(initial_filters*32, 7, 2),
          # ConvBlock(initial_filters*8, 3, 1),
          # ConvBlock(initial_filters*8, 3, 1),

          DeconvBlock(initial_filters*16, 7, 2),
          ConvBlock(initial_filters*4, 3, 1),
          ConvBlock(initial_filters*4, 3, 1),

          DeconvBlock(initial_filters*8, 7, 2),
          ConvBlock(initial_filters*2, 3, 1),
          ConvBlock(initial_filters*2, 3, 1),

          DeconvBlock(initial_filters*4, 7, 2),
          ConvBlock(initial_filters*1, 3, 1),
          ConvBlock(initial_filters*1, 3, 1),

          DeconvBlock(initial_filters*2, 7, 2),
          # ConvBlock(initial_filters*1, 3, 1),
          # ConvBlock(initial_filters*1, 3, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 7, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoiseSmallerGenLayerFancyFilters.Discriminator, self).__init__()

      initial_filters = 32

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


class Deep480pNoiseNoDeconv(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoiseNoDeconv.Generator, self).__init__()

      initial_filters = int(512/32)

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          ResizeBlock((30, 40), initial_filters*32, 5),
          # ConvBlock(initial_filters*16, 5, 1),

          ResizeBlock((60, 80), initial_filters*16, 5),
          ConvBlock(initial_filters*8, 5, 1),

          ResizeBlock((120, 160), initial_filters*8, 5),
          ConvBlock(initial_filters*4, 5, 1),

          ResizeBlock((240, 320), initial_filters*4, 5),
          ConvBlock(initial_filters*2, 5, 1),

          ResizeBlock((480, 640), initial_filters*2, 5),
          # ConvBlock(initial_filters*1, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoiseNoDeconv.Discriminator, self).__init__()

      initial_filters = 32

      self.blocks = [
          ConvBlock(initial_filters*2, 4, 2),
          ConvBlock(initial_filters*1, 5, 1),

          ConvBlock(initial_filters*4, 4, 2),
          ConvBlock(initial_filters*2, 5, 1),

          ConvBlock(initial_filters*8, 4, 2),
          ConvBlock(initial_filters*4, 5, 1),

          ConvBlock(initial_filters*16, 4, 2),
          ConvBlock(initial_filters*8, 5, 1),
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


class Deep480pNoiseResidual(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoiseResidual.Generator, self).__init__()

      initial_filters = int(512/32)//2

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          ReverseResidualBlock(initial_filters*32, 5, 2),
          ConvBlock(initial_filters*2*16, 5, 1),

          ReverseResidualBlock(initial_filters*16, 5, 2),
          ConvBlock(initial_filters*2*8, 5, 1),

          ReverseResidualBlock(initial_filters*8, 5, 2),
          ConvBlock(initial_filters*2*4, 5, 1),

          ReverseResidualBlock(initial_filters*4, 5, 2),
          ConvBlock(initial_filters*2*2, 5, 1),

          ReverseResidualBlock(initial_filters*2, 5, 2),
          ConvBlock(initial_filters*2*1, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoiseResidual.Discriminator, self).__init__()

      initial_filters = 32

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


class Deep480pNoiseMultiscaleDisc(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoiseMultiscaleDisc.Generator, self).__init__()

      initial_filters = int(512/32)

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
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


          # # more filters in deconv
          # DeconvBlock(initial_filters*32*2, 5, 2),
          # ConvBlock(initial_filters*16, 5, 1),

          # DeconvBlock(initial_filters*16*2, 5, 2),
          # ConvBlock(initial_filters*8, 5, 1),

          # DeconvBlock(initial_filters*8*2, 5, 2),
          # ConvBlock(initial_filters*4, 5, 1),

          # DeconvBlock(initial_filters*4*2, 5, 2),
          # ConvBlock(initial_filters*2, 5, 1),

          # DeconvBlock(initial_filters*2, 5, 2),
          # # ConvBlock(initial_filters*1, 5, 1),


          # # more filters in conv
          # DeconvBlock(initial_filters*32, 5, 2),
          # ConvBlock(initial_filters*16*2, 5, 1),

          # DeconvBlock(initial_filters*16, 5, 2),
          # ConvBlock(initial_filters*8*2, 5, 1),

          # DeconvBlock(initial_filters*8, 5, 2),
          # ConvBlock(initial_filters*4*2, 5, 1),

          # DeconvBlock(initial_filters*4, 5, 2),
          # ConvBlock(initial_filters*2*2, 5, 1),

          # DeconvBlock(initial_filters*2, 5, 2),
          # ConvBlock(initial_filters*1*1, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout):
        super(Deep480pNoiseMultiscaleDisc.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(640 * scaling_factor)
          size_y = int(480 * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32//2

        self.blocks = [
            # default
            ConvBlock(initial_filters*2, 4, 2),
            ConvBlock(initial_filters*1, 4, 1),

            ConvBlock(initial_filters*4, 4, 2),
            ConvBlock(initial_filters*2, 4, 1),

            ConvBlock(initial_filters*8, 4, 2),
            ConvBlock(initial_filters*4, 4, 1),

            ConvBlock(initial_filters*16, 4, 2),
            ConvBlock(initial_filters*8, 4, 1),

            # NOTE: keep track of image resizing+conv!
            ConvBlock(initial_filters*32, 4, 2),
            ConvBlock(initial_filters*16, 4, 1),

            # # more filters in unstrided
            # ConvBlock(initial_filters*2, 4, 2),
            # ConvBlock(initial_filters*1*2, 5, 1),

            # ConvBlock(initial_filters*4, 4, 2),
            # ConvBlock(initial_filters*2*2, 5, 1),

            # ConvBlock(initial_filters*8, 4, 2),
            # ConvBlock(initial_filters*4*2, 5, 1),

            # # NOTE: keep track of image resizing+conv!
            # ConvBlock(initial_filters*16, 4, 2),
            # ConvBlock(initial_filters*8*2, 5, 1),
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
      super(Deep480pNoiseMultiscaleDisc.Discriminator, self).__init__()
      self.discriminators = [Deep480pNoiseMultiscaleDisc.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3)) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(Deep480pNoiseMultiscaleDisc.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class Deep480pNoiseMultiscaleDiscGenLarge(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoiseMultiscaleDiscGenLarge.Generator, self).__init__()

      initial_filters = int(512/32/2)

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
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

          DeconvBlock(initial_filters*1, 5, 2),
          ConvBlock(initial_filters*1, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      x = tanh(self.final_conv(x))
      return tf.image.resize_nearest_neighbor(x, (480, 640))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout):
        super(Deep480pNoiseMultiscaleDiscGenLarge.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(640 * scaling_factor)
          size_y = int(480 * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32//2

        self.blocks = [
            # default
            ConvBlock(initial_filters*2, 4, 2),
            ConvBlock(initial_filters*1, 5, 1),

            ConvBlock(initial_filters*4, 4, 2),
            ConvBlock(initial_filters*2, 5, 1),

            ConvBlock(initial_filters*8, 4, 2),
            ConvBlock(initial_filters*4, 5, 1),

            # NOTE: keep track of image resizing+conv!
            ConvBlock(initial_filters*16, 4, 2),
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
      super(Deep480pNoiseMultiscaleDiscGenLarge.Discriminator, self).__init__()
      self.discriminators = [Deep480pNoiseMultiscaleDiscGenLarge.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3)) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(Deep480pNoiseMultiscaleDiscGenLarge.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class Deep480pNoiseMultiscaleDiscShallow(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoiseMultiscaleDiscShallow.Generator, self).__init__()

      initial_filters = int(512/32)

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          DeconvBlock(initial_filters*32, 5, 2),
          DeconvBlock(initial_filters*16, 5, 2),
          DeconvBlock(initial_filters*8, 5, 2),
          DeconvBlock(initial_filters*4, 5, 2),
          DeconvBlock(initial_filters*2, 5, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout):
        super(Deep480pNoiseMultiscaleDiscShallow.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        self.scaling_factor = scaling_factor
        self.resize = None

        initial_filters = 32//2

        self.blocks = [
            ConvBlock(initial_filters*2, 5, 2),
            ConvBlock(initial_filters*4, 5, 2),
            ConvBlock(initial_filters*8, 5, 2),
            ConvBlock(initial_filters*16, 5, 2),
            ConvBlock(initial_filters*32, 5, 2),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        self.fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        if self.resize is None:
          if self.scaling_factor != 1:
            size_x = int(x.shape[1].value * self.scaling_factor)
            size_y = int(x.shape[2].value * self.scaling_factor)
            tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
            self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
          else:
            tf.logging.info("Multiscale discriminator operating on regular resolution")
            self.resize = lambda x: x

        x = self.resize(x)
        for block in self.blocks:
          x = block(x, training=training)
          x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def __init__(self, config):
      super(Deep480pNoiseMultiscaleDiscShallow.Discriminator, self).__init__()
      self.discriminators = [Deep480pNoiseMultiscaleDiscShallow.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3)) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(Deep480pNoiseMultiscaleDiscShallow.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class Deep480pNoiseResizeMultiscaleDiscShallow(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoiseResizeMultiscaleDiscShallow.Generator, self).__init__()

      initial_filters = int(512/32)

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          # ResizeBlock((30, 40), initial_filters*32, 5),
          # ResizeBlock((60, 80), initial_filters*16, 5),
          # ResizeBlock((120, 160), initial_filters*8, 5),
          # ResizeBlock((240, 320), initial_filters*4, 5),
          # ResizeBlock((480, 640), initial_filters*2, 5),

          DeconvBlock(initial_filters*32, 5, 2),
          DeconvBlock(initial_filters*16, 5, 2),
          DeconvBlock(initial_filters*8, 5, 2),
          DeconvBlock(initial_filters*4, 5, 2),
          DeconvBlock(initial_filters*2, 5, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout):
        super(Deep480pNoiseResizeMultiscaleDiscShallow.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        self.scaling_factor = scaling_factor
        self.resize = None

        initial_filters = 32//2

        self.blocks = [
            # resize is on smaller resolution so that it fits in memory...
            ResizeBlock((240, 320), initial_filters*2, 5),
            ResizeBlock((120, 160), initial_filters*4, 5),
            ResizeBlock((60, 80), initial_filters*8, 5),
            ResizeBlock((30, 40), initial_filters*16, 5),
            ResizeBlock((15, 20), initial_filters*32, 5),

            # ConvBlock(initial_filters*2, 5, 2),
            # ConvBlock(initial_filters*4, 5, 2),
            # ConvBlock(initial_filters*8, 5, 2),
            # ConvBlock(initial_filters*16, 5, 2),
            # ConvBlock(initial_filters*32, 5, 2),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        self.fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        if self.resize is None:
          if self.scaling_factor != 1:
            size_x = int(x.shape[1].value * self.scaling_factor)
            size_y = int(x.shape[2].value * self.scaling_factor)
            tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
            self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
          else:
            tf.logging.info("Multiscale discriminator operating on regular resolution")
            self.resize = lambda x: x

        x = self.resize(x)
        for block in self.blocks:
          x = block(x, training=training)
          x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def __init__(self, config):
      super(Deep480pNoiseResizeMultiscaleDiscShallow.Discriminator, self).__init__()
      self.discriminators = [Deep480pNoiseResizeMultiscaleDiscShallow.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3)) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(Deep480pNoiseResizeMultiscaleDiscShallow.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class Deep60pNoise(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep60pNoise.Generator, self).__init__()

      initial_filters = int(512/32) * 4

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          DeconvBlock(initial_filters*32, 5, 2),
          ConvBlock(initial_filters*16, 5, 1),

          DeconvBlock(initial_filters*16, 5, 2),
          ConvBlock(initial_filters*8, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(Deep60pNoise.Discriminator, self).__init__()

      initial_filters = 32 * 4

      self.blocks = [
          ConvBlock(initial_filters*2, 5, 2),
          ConvBlock(initial_filters*1, 5, 1),

          ConvBlock(initial_filters*4, 5, 2),
          ConvBlock(initial_filters*2, 5, 1),

          ConvBlock(initial_filters*8, 5, 2),
          ConvBlock(initial_filters*4, 5, 1),
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


class Deep60pNoiseDeeper(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep60pNoiseDeeper.Generator, self).__init__()

      initial_filters = int(512/32) * 2

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          ConvBlock(initial_filters*32, 5, 1),
          ConvBlock(initial_filters*16, 5, 1),
          ConvBlock(initial_filters*16, 5, 1),

          DeconvBlock(initial_filters*32, 5, 2),
          ConvBlock(initial_filters*16, 5, 1),
          ConvBlock(initial_filters*16, 5, 1),

          DeconvBlock(initial_filters*16, 5, 2),
          ConvBlock(initial_filters*8, 5, 1),
          ConvBlock(initial_filters*8, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(Deep60pNoiseDeeper.Discriminator, self).__init__()

      initial_filters = 32 * 2

      self.blocks = [
          ConvBlock(initial_filters*2, 5, 2),
          ConvBlock(initial_filters*1, 5, 1),
          ConvBlock(initial_filters*1, 5, 1),

          ConvBlock(initial_filters*4, 5, 2),
          ConvBlock(initial_filters*2, 5, 1),
          ConvBlock(initial_filters*2, 5, 1),

          ConvBlock(initial_filters*8, 5, 2),
          ConvBlock(initial_filters*4, 5, 1),
          ConvBlock(initial_filters*4, 5, 1),
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


class Deep120pNoise(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep120pNoise.Generator, self).__init__()

      initial_filters = int(512/32) * 4

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          DeconvBlock(initial_filters*32, 5, 2),
          # ConvBlock(initial_filters*16, 5, 1),

          DeconvBlock(initial_filters*16, 5, 2),
          ConvBlock(initial_filters*8, 5, 1),

          DeconvBlock(initial_filters*8, 5, 2),
          ConvBlock(initial_filters*4, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(Deep120pNoise.Discriminator, self).__init__()

      initial_filters = 32 * 4

      self.blocks = [
          ConvBlock(initial_filters*2, 5, 2),
          ConvBlock(initial_filters*1, 5, 1),

          ConvBlock(initial_filters*4, 5, 2),
          ConvBlock(initial_filters*2, 5, 1),

          ConvBlock(initial_filters*8, 5, 2),
          ConvBlock(initial_filters*4, 5, 1),
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


class Deep120pNoiseMultiscaleDisc(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep120pNoiseMultiscaleDisc.Generator, self).__init__()

      initial_filters = int(512/32) * 4

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          DeconvBlock(initial_filters*32, 5, 2),
          # ConvBlock(initial_filters*16, 5, 1),

          DeconvBlock(initial_filters*16, 5, 2),
          ConvBlock(initial_filters*8, 5, 1),

          DeconvBlock(initial_filters*8, 5, 2),
          ConvBlock(initial_filters*4, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout):
        super(Deep120pNoiseMultiscaleDisc.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(160 * scaling_factor)
          size_y = int(120 * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32//1 * 4

        self.blocks = [
          ConvBlock(initial_filters*2, 5, 2),
          ConvBlock(initial_filters*1, 5, 1),

          ConvBlock(initial_filters*4, 5, 2),
          ConvBlock(initial_filters*2, 5, 1),

          ConvBlock(initial_filters*8, 5, 2),
          ConvBlock(initial_filters*4, 5, 1),
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
      super(Deep120pNoiseMultiscaleDisc.Discriminator, self).__init__()
      self.discriminators = [Deep120pNoiseMultiscaleDisc.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3)) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(Deep120pNoiseMultiscaleDisc.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class Deep120pNoiseDeeper(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep120pNoiseDeeper.Generator, self).__init__()

      initial_filters = int(512/32) * 2

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          DeconvBlock(initial_filters*32, 5, 2),
          ConvBlock(initial_filters*16, 5, 1),
          ConvBlock(initial_filters*16, 5, 1),

          DeconvBlock(initial_filters*16, 5, 2),
          ConvBlock(initial_filters*8, 5, 1),
          ConvBlock(initial_filters*8, 5, 1),

          DeconvBlock(initial_filters*8, 5, 2),
          ConvBlock(initial_filters*4, 5, 1),
          ConvBlock(initial_filters*4, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(Deep120pNoiseDeeper.Discriminator, self).__init__()

      initial_filters = 32 * 2

      self.blocks = [
          ConvBlock(initial_filters*2, 5, 2),
          ConvBlock(initial_filters*1, 5, 1),
          ConvBlock(initial_filters*1, 5, 1),

          ConvBlock(initial_filters*4, 5, 2),
          ConvBlock(initial_filters*2, 5, 1),
          ConvBlock(initial_filters*2, 5, 1),

          ConvBlock(initial_filters*8, 5, 2),
          ConvBlock(initial_filters*4, 5, 1),
          ConvBlock(initial_filters*4, 5, 1),

          ConvBlock(initial_filters*16, 5, 2),
          ConvBlock(initial_filters*8, 5, 1),
          ConvBlock(initial_filters*8, 5, 1),
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

class Deep120pNoiseShallowGenMultiscaleDisc(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep120pNoiseShallowGenMultiscaleDisc.Generator, self).__init__()

      initial_filters = 1024

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          DeconvBlock(initial_filters, 5, 2),
          DeconvBlock(initial_filters, 5, 2),
          DeconvBlock(initial_filters, 5, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout):
        super(Deep120pNoiseShallowGenMultiscaleDisc.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(160 * scaling_factor)
          size_y = int(120 * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32//1 * 4

        self.blocks = [
          ConvBlock(initial_filters*2, 5, 2),
          ConvBlock(initial_filters*1, 5, 1),

          ConvBlock(initial_filters*4, 5, 2),
          ConvBlock(initial_filters*2, 5, 1),

          ConvBlock(initial_filters*8, 5, 2),
          ConvBlock(initial_filters*4, 5, 1),
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
      super(Deep120pNoiseShallowGenMultiscaleDisc.Discriminator, self).__init__()
      self.discriminators = [Deep120pNoiseShallowGenMultiscaleDisc.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3)) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(Deep120pNoiseShallowGenMultiscaleDisc.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class Deep240pNoise(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep240pNoise.Generator, self).__init__()

      initial_filters = int(512/32) * 2

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
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
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(Deep240pNoise.Discriminator, self).__init__()

      initial_filters = 32 * 2

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


class Deep240pNoiseMultiscaleDisc(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep240pNoiseMultiscaleDisc.Generator, self).__init__()

      initial_filters = int(512/32) * 2

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          DeconvBlock(initial_filters*32, 5, 2),
          ConvBlock(initial_filters*16, 5, 1),
          ConvBlock(initial_filters*16, 5, 1),

          DeconvBlock(initial_filters*16, 5, 2),
          ConvBlock(initial_filters*8, 5, 1),
          ConvBlock(initial_filters*8, 5, 1),

          DeconvBlock(initial_filters*8, 5, 2),
          ConvBlock(initial_filters*4, 5, 1),
          ConvBlock(initial_filters*4, 5, 1),

          DeconvBlock(initial_filters*4, 5, 2),
          ConvBlock(initial_filters*2, 5, 1),
          ConvBlock(initial_filters*2, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout):
        super(Deep240pNoiseMultiscaleDisc.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(320 * scaling_factor)
          size_y = int(240 * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32//2 * 2

        self.blocks = [
            ConvBlock(initial_filters*2, 5, 2),
            ConvBlock(initial_filters*2, 5, 1),
            # ConvBlock(initial_filters*2, 5, 1),

            ConvBlock(initial_filters*4, 5, 2),
            ConvBlock(initial_filters*4, 5, 1),
            # ConvBlock(initial_filters*4, 5, 1),

            ConvBlock(initial_filters*8, 5, 2),
            ConvBlock(initial_filters*8, 5, 1),
            # ConvBlock(initial_filters*8, 5, 1),

            # NOTE: keep track of image resizing+conv!
            ConvBlock(initial_filters*16, 5, 2),
            ConvBlock(initial_filters*16, 5, 1),
            # ConvBlock(initial_filters*16, 5, 1),
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
      super(Deep240pNoiseMultiscaleDisc.Discriminator, self).__init__()
      self.discriminators = [Deep240pNoiseMultiscaleDisc.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3)) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(Deep240pNoiseMultiscaleDisc.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class Deep480pNoiseMsDiscS2S1(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoiseMsDiscS2S1.Generator, self).__init__()

      initial_filters = int(512/32)*1

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
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
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)

      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout):
        super(Deep480pNoiseMsDiscS2S1.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(640 * scaling_factor)
          size_y = int(480 * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32//2*1

        self.blocks = [
            ConvBlock(initial_filters*2, 4, 2),
            ConvBlock(initial_filters*1, 4, 1),

            ConvBlock(initial_filters*4, 4, 2),
            ConvBlock(initial_filters*2, 4, 1),

            ConvBlock(initial_filters*8, 4, 2),
            ConvBlock(initial_filters*4, 4, 1),

            ConvBlock(initial_filters*16, 4, 2),
            ConvBlock(initial_filters*8, 4, 1),

            ConvBlock(initial_filters*32, 4, 2),
            ConvBlock(initial_filters*16, 4, 1),
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
      super(Deep480pNoiseMsDiscS2S1.Discriminator, self).__init__()
      self.discriminators = [Deep480pNoiseMsDiscS2S1.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3)) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(Deep480pNoiseMsDiscS2S1.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class Deep480pNoiseMsDiscS2(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoiseMsDiscS2.Generator, self).__init__()

      initial_filters = int(512/32)*1

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
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
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout):
        super(Deep480pNoiseMsDiscS2.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(640 * scaling_factor)
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
      super(Deep480pNoiseMsDiscS2.Discriminator, self).__init__()
      self.discriminators = [Deep480pNoiseMsDiscS2.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3)) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(Deep480pNoiseMsDiscS2.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class Deep480pNoiseMsDiscS2S1Shared(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoiseMsDiscS2S1Shared.Generator, self).__init__()

      initial_filters = int(512/32)

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
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
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout):
        super(Deep480pNoiseMsDiscS2S1Shared.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(640 * scaling_factor)
          size_y = int(480 * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32//2//2

        self.s2_blocks = [
            ConvBlock(initial_filters*2, 5, 2),

            ConvBlock(initial_filters*4, 5, 2),

            ConvBlock(initial_filters*8, 5, 2),

            ConvBlock(initial_filters*16, 5, 2),

            # ConvBlock(initial_filters*32, 5, 2),
            ]

        self.s1_blocks = [
            ConvBlock(initial_filters*2, 5, 1),

            ConvBlock(initial_filters*4, 5, 1),

            ConvBlock(initial_filters*8, 5, 1),

            ConvBlock(initial_filters*16, 5, 1),

            # ConvBlock(initial_filters*32, 5, 1),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        self.s2_fc = Dense(config.discriminator_classes, use_bias=False)
        self.s2s1_fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        x = self.resize(x)
        s2 = x
        for block in self.s2_blocks:
          s2 = block(s2, training=training)
          s2 = self.dropout(s2, training=training)
        s2 = self.flatten(s2)
        s2 = self.s2_fc(s2)
        s2s1 = x
        for i in range(len(self.s1_blocks)):
          s2s1 = self.s2_blocks[i](s2s1, training=training)
          s2s1 = self.dropout(s2s1, training=training)
          s2s1 = self.s1_blocks[i](s2s1, training=training)
          s2s1 = self.dropout(s2s1, training=training)
        s2s1 = self.flatten(s2s1)
        s2s1 = self.s2s1_fc(s2s1)
        return tf.reduce_mean(tf.concat([s2, s2s1], axis=-1), axis=-1, keepdims=True)

    def __init__(self, config):
      super(Deep480pNoiseMsDiscS2S1Shared.Discriminator, self).__init__()
      self.discriminators = [Deep480pNoiseMsDiscS2S1Shared.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3)) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(Deep480pNoiseMsDiscS2S1Shared.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class Deep480pNoiseS2S1Shared(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoiseS2S1Shared.Generator, self).__init__()

      initial_filters = int(512/32)

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
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
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoiseS2S1Shared.Discriminator, self).__init__()

      initial_filters = 32//2

      self.s2_blocks = [
          ConvBlock(initial_filters*2, 4, 2),

          ConvBlock(initial_filters*4, 4, 2),

          ConvBlock(initial_filters*8, 4, 2),

          ConvBlock(initial_filters*16, 4, 2),

          # ConvBlock(initial_filters*32, 4, 2),
          ]

      self.s1_blocks = [
          ConvBlock(initial_filters*2, 4, 1),

          ConvBlock(initial_filters*4, 4, 1),

          ConvBlock(initial_filters*8, 4, 1),

          ConvBlock(initial_filters*16, 4, 1),

          # ConvBlock(initial_filters*32, 4, 1),
          ]

      self.dropout = Dropout(0.3)
      self.flatten = Flatten()
      self.s2_fc = Dense(config.discriminator_classes, use_bias=False)
      self.s2s1_fc = Dense(config.discriminator_classes, use_bias=False)

    def call(self, x, training):
      s2 = x
      for block in self.s2_blocks:
        s2 = block(s2, training=training)
        s2 = self.dropout(s2, training=training)
      s2 = self.flatten(s2)
      s2 = self.s2_fc(s2)
      s2s1 = x
      for i in range(len(self.s1_blocks)):
        s2s1 = self.s2_blocks[i](s2s1, training=training)
        s2s1 = self.dropout(s2s1, training=training)
        s2s1 = self.s1_blocks[i](s2s1, training=training)
        s2s1 = self.dropout(s2s1, training=training)
      s2s1 = self.flatten(s2s1)
      s2s1 = self.s2s1_fc(s2s1)
      return tf.reduce_mean(tf.concat([s2, s2s1], axis=-1), axis=-1, keepdims=True)


class Deep480pNoiseMsDiscS2S1Modified(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoiseMsDiscS2S1Modified.Generator, self).__init__()

      initial_filters = int(512/32)

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
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
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout):
        super(Deep480pNoiseMsDiscS2S1Modified.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(640 * scaling_factor)
          size_y = int(480 * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32//2

        self.blocks = [
            ConvBlock(initial_filters*2, 7, 2),
            ConvBlock(initial_filters*2, 7, 1),

            ConvBlock(initial_filters*4, 7, 2),
            ConvBlock(initial_filters*4, 7, 1),

            ConvBlock(initial_filters*8, 7, 2),
            ConvBlock(initial_filters*8, 7, 1),

            ConvBlock(initial_filters*16, 7, 2),
            ConvBlock(initial_filters*16, 7, 1),

            # ConvBlock(initial_filters*32, 7, 2),
            # ConvBlock(initial_filters*16, 7, 1),
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
      super(Deep480pNoiseMsDiscS2S1Modified.Discriminator, self).__init__()
      self.discriminators = [Deep480pNoiseMsDiscS2S1Modified.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3)) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(Deep480pNoiseMsDiscS2S1Modified.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class Deep480pNoisePatch(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoisePatch.Generator, self).__init__()

      initial_filters = int(512/32)

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
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
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoisePatch.Discriminator, self).__init__()
      del config

      initial_filters = 32

      self.blocks = [
          ConvBlock(initial_filters*2, 4, 2),
          # ConvBlock(initial_filters*1, 4, 1),

          ConvBlock(initial_filters*4, 4, 2),
          # ConvBlock(initial_filters*2, 4, 1),

          ConvBlock(initial_filters*8, 4, 2),
          # ConvBlock(initial_filters*4, 4, 1),

          ConvBlock(initial_filters*16, 4, 2),
          # ConvBlock(initial_filters*8, 4, 1),
          ]

      self.dropout = Dropout(0.3)
      self.final_conv = Conv(1, 4, 1)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
        x = self.dropout(x, training=training)
      x = self.final_conv(x)
      x = tf.reduce_mean(x, axis=[1, 2])
      return x


class Deep480pNoiseMsDiscS2EvenG(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep480pNoiseMsDiscS2EvenG.Generator, self).__init__()

      initial_filters = 64

      self.fc = tf.keras.layers.Dense(15*20*64, use_bias=False)
      self.initial_norm = tf.keras.layers.BatchNormalization()

      self.blocks = [
          # default
          DeconvBlock(initial_filters*1, 5, 2),
          # ConvBlock(initial_filters*1, 5, 1),

          DeconvBlock(initial_filters*1, 5, 2),
          ConvBlock(initial_filters*1, 5, 1),

          DeconvBlock(initial_filters*1, 5, 2),
          ConvBlock(initial_filters*1, 5, 1),

          DeconvBlock(initial_filters*1, 5, 2),
          ConvBlock(initial_filters*1, 5, 1),

          DeconvBlock(initial_filters*1, 5, 2),
          # ConvBlock(initial_filters*1, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.fc(x)
      x = self.initial_norm(x, training=training)
      x = tf.nn.relu(x)
      x = tf.reshape(x, shape=(-1, 15, 20, 64))

      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout):
        super(Deep480pNoiseMsDiscS2EvenG.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(640 * scaling_factor)
          size_y = int(480 * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32//2

        self.blocks = [
            ConvBlock(initial_filters*2, 5, 2),

            ConvBlock(initial_filters*4, 5, 2),

            ConvBlock(initial_filters*8, 5, 2),

            ConvBlock(initial_filters*16, 5, 2),

            ConvBlock(initial_filters*32, 5, 2),
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
      super(Deep480pNoiseMsDiscS2EvenG.Discriminator, self).__init__()
      self.discriminators = [Deep480pNoiseMsDiscS2EvenG.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3)) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(Deep480pNoiseMsDiscS2EvenG.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)
