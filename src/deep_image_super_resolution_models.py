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


class Deep120pTo480p(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep120pTo480p.Generator, self).__init__()

      initial_filters = 1024//4

      self.blocks = [
          ConvBlock(initial_filters*1, 5, 1),
          ConvBlock(initial_filters*2, 5, 1),

          DeconvBlock(initial_filters*4, 5, 2),

          ConvBlock(initial_filters*4, 5, 1),
          ConvBlock(initial_filters*4, 5, 1),

          DeconvBlock(initial_filters*4, 5, 2),

          ConvBlock(initial_filters*2, 5, 1),
          ConvBlock(initial_filters*1, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout):
        super(Deep120pTo480p.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(640 * scaling_factor)
          size_y = int(480 * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32//1

        self.blocks = [
            ConvBlock(initial_filters*2, 5, 2),
            ConvBlock(initial_filters*2, 5, 1),

            ConvBlock(initial_filters*4, 5, 2),
            ConvBlock(initial_filters*4, 5, 1),

            ConvBlock(initial_filters*8, 5, 2),
            ConvBlock(initial_filters*8, 5, 1),

            ConvBlock(initial_filters*16, 5, 2),
            ConvBlock(initial_filters*16, 5, 1),
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
      super(Deep120pTo480p.Discriminator, self).__init__()
      self.discriminators = [Deep120pTo480p.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3)) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(Deep120pTo480p.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class Deep240pTo480p(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(Deep240pTo480p.Generator, self).__init__()

      initial_filters = 1024//4

      self.blocks = [
          ConvBlock(initial_filters*1, 5, 1),
          ConvBlock(initial_filters*2, 5, 1),

          DeconvBlock(initial_filters*4, 5, 2),

          ConvBlock(initial_filters*2, 5, 1),
          ConvBlock(initial_filters*1, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)
      self._low_res_generator = None

    def set_low_res_generator(self, low_res_generator):
      self._low_res_generator = low_res_generator
      for layer in self._low_res_generator.layers:
        layer.trainable = False

    def call(self, x, training=True, return_lowres=False):
      if len(x.shape) == 2:
        x = self._low_res_generator(x, training=False)
      lowres = x
      for block in self.blocks:
        x = block(x, training=training)
      hires = tanh(self.final_conv(x))
      return (hires, lowres) if return_lowres else hires

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout):
        super(Deep240pTo480p.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(640 * scaling_factor)
          size_y = int(480 * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32//1

        self.blocks = [
            ConvBlock(initial_filters*2, 5, 2),
            ConvBlock(initial_filters*2, 5, 1),

            ConvBlock(initial_filters*4, 5, 2),
            ConvBlock(initial_filters*4, 5, 1),

            ConvBlock(initial_filters*8, 5, 2),
            ConvBlock(initial_filters*8, 5, 1),

            ConvBlock(initial_filters*16, 5, 2),
            ConvBlock(initial_filters*16, 5, 1),
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
      super(Deep240pTo480p.Discriminator, self).__init__()
      self.discriminators = [Deep240pTo480p.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3)) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(Deep240pTo480p.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)
