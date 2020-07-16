#!/usr/bin/env python

# pylint: disable=wrong-import-position

import os
from abc import ABC, abstractmethod

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils import logistic


class Evaluation(ABC):
  def __init__(self, model, config):
    self._model = model
    self._config = config

  @property
  @abstractmethod
  def data_set(self):
    pass

  @property
  @abstractmethod
  def extra_discriminator_data_set(self):
    pass

  @property
  @abstractmethod
  def test_data_set(self):
    pass

  @property
  @abstractmethod
  def epoch_sample_input(self):
    pass

  @abstractmethod
  def set_up_model(self):
    """
    Sets up the models (G/D) including the respective optimizers as well as other aspects of the evaluation.
    """
    pass

  @abstractmethod
  def load_data(self):
    """
    Loads the specified data sets.

    :returns: The number of samples of the data set.
    """
    pass

  @abstractmethod
  def train(self, epochs, metrics_writer):
    """
    Trains the models for the specified number of epochs.
    """
    pass

  @abstractmethod
  def _plot_epoch_samples(self, generator, discriminator):
    """
    Generates and plots samples based on the epoch-samples-input. The samples are judged by the discriminator.

    :returns: Tuple consisting of the genearted samples and the associated discriminator predictions.
    """
    pass

  @abstractmethod
  def _plot_hq_epoch_samples(self, generated_samples, discriminator_probabilities):
    """
    Plots some of the generated samples with higher resolution.
    """
    pass

  @staticmethod
  def plot_image(image, title=None, denormalize=True, title_size=None):
    """
    Plots the provided image. By default, the image is expected to be normalized to [-1,1] and,
    consequently, is denormalized before plotting.
    """
    assert len(image.shape) == 3, "expected image to have 3 dimensions"
    assert image.shape[2] in [1, 3], "expected to have either 1 or 3 color channels"
    plt.axis("off")
    if title is not None:
      if title_size:
        plt.title(str(title), fontsize=title_size)
      else:
        plt.title(str(title))
    image = np.array((image+1) * 127.5, dtype=np.uint8) if denormalize else np.array(image, dtype=np.uint8)
    if image.shape[2] == 1:
      plt.imshow(image[:, :, 0], cmap="gray", vmin=0, vmax=255)
    else:
      plt.imshow(image)

  def save_epoch_samples(self, generator, discriminator, epoch, save_hq_samples):
    """
    Generates and stores the epoch samples. Optionally, some may also be stored with higher resolution.
    """
    plt.figure(figsize=(16, 16))
    plt.suptitle("{}: Epoch {}".format(self._config.eid, epoch), fontsize=16)
    generated_samples, discriminator_probabilities = self._plot_epoch_samples(generator, discriminator)
    figure_file = os.path.join(self._config.figures_dir, "image_at_epoch_{:04d}.png").format(epoch)
    plt.savefig(figure_file)
    plt.close()

    if save_hq_samples:
      plt.figure(figsize=(16, 16))
      plt.suptitle("{}: Epoch {}".format(self._config.eid, epoch), fontsize=16)
      self._plot_hq_epoch_samples(generated_samples, discriminator_probabilities)
      figure_file = os.path.join(self._config.hq_figures_dir, "image_at_epoch_{:04d}.png").format(epoch)
      plt.savefig(figure_file)
      plt.close()

    return generated_samples

  def _discriminate_data_set(self, discriminator, data_set):
    discriminations = []
    for batch in data_set:
      inputs, targets = batch
      if self._config.online_augmentation:
        targets = tf.concat([
          targets,
          tf.image.flip_left_right(targets),
          tf.image.flip_up_down(targets),
          tf.image.flip_left_right(tf.image.flip_up_down(targets))
          ], axis=0)
        inputs = tf.random_normal([targets.shape[0], self._config.noise_dimensions])
      disc_input = tf.concat([inputs, targets], axis=-1) if self._config.conditioned_discriminator else targets
      disc_predictions = discriminator(disc_input, training=False)
      discriminations.extend(logistic(disc_predictions))
    return discriminations
