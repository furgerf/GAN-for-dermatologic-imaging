#!/usr/bin/env python

# pylint: disable=wrong-import-position

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from evaluation import Evaluation
from one_way_evaluation import OneWayEvaluation
from utils import (load_checkpoint, load_image_names, load_images, load_model,
                   logistic)


class NoiseToImageEvaluation(OneWayEvaluation):
  def __init__(self, model, config):
    self._epoch_images_shape = (5, 4)
    self._data_set = None
    self._test_data_set = None
    self._epoch_sample_input = tf.random_normal([np.prod(self._epoch_images_shape), config.noise_dimensions])
    assert not config.conditioned_discriminator, "Doesn't make sense for D to process noise input"
    super(NoiseToImageEvaluation, self).__init__(model, config)

  def load_data(self):
    image_names = load_image_names(self._config.data_dir, self._config.match_pattern)
    target_images = load_images(image_names, self._config.data_dir, self._config.target_type,
        flip_lr=self._config.augmentation_flip_lr, flip_ud=self._config.augmentation_flip_ud)

    # load/prepare test data
    test_target_images = None
    if self._config.test_data_dir:
      tf.logging.info("Loading test data")
      test_image_names = load_image_names(self._config.test_data_dir, self._config.match_pattern)
      test_target_images = load_images(test_image_names, self._config.test_data_dir, self._config.target_type,
          flip_lr=False, flip_ud=False)
    if self._config.test_data_percentage:
      tf.logging.warning("Using the first {}% of the training data for testing".format(100*self._config.test_data_percentage))
      split = int(len(target_images) * self._config.test_data_percentage)
      test_target_images = target_images[:split]
      target_images = target_images[split:]
    data_set_size = len(target_images)
    test_set_size = len([] if test_target_images is None else test_target_images)

    # build data sets
    self._data_set = tf.data.Dataset.from_tensor_slices(target_images)\
        .map(lambda x: (tf.random_normal([self._config.noise_dimensions]), x))\
        .shuffle(data_set_size).batch(self._config.batch_size)
    del target_images
    if test_target_images is not None:
      self._test_data_set = tf.data.Dataset.from_tensor_slices(test_target_images)\
          .map(lambda x: (tf.random_normal([self._config.noise_dimensions]), x))\
          .shuffle(test_set_size).batch(self._config.batch_size)
      del test_target_images
    else:
      tf.logging.warning("Running evaluation without test data!")

    return data_set_size, test_set_size

  @property
  def data_set(self):
    return self._data_set

  @property
  def extra_discriminator_data_set(self):
    return None

  @property
  def test_data_set(self):
    return self._test_data_set

  @property
  def epoch_sample_input(self):
    return self._epoch_sample_input

  def _plot_epoch_samples(self, generator, discriminator):
    samples = []
    predictions = []
    for batch in tf.data.Dataset.from_tensor_slices(self.epoch_sample_input).batch(self._config.batch_size):
      new_samples = generator(batch, training=True)
      new_predictions = logistic(discriminator(new_samples, training=True))
      samples.append(new_samples)
      predictions.append(new_predictions)

    samples = tf.concat(samples, axis=0)
    predictions = tf.concat(predictions, axis=0)
    for i in range(samples.shape[0]):
      plt.subplot(*self._epoch_images_shape, i+1)
      Evaluation.plot_image(samples[i], np.round(predictions[i].numpy(), 5))
    return samples, predictions

  def _plot_hq_epoch_samples(self, generated_samples, discriminator_probabilities):
    for i in range((self._epoch_images_shape[0]+1)//2 * self._epoch_images_shape[1]//2):
      plt.subplot((self._epoch_images_shape[0]+1)//2, self._epoch_images_shape[1]//2, i+1)
      Evaluation.plot_image(generated_samples[i], np.round(discriminator_probabilities[i].numpy(), 5))


class NoiseImageSuperResolutionEvaluation(OneWayEvaluation):
  def __init__(self, model, config):
    self._epoch_images_shape = (5, 2)
    self._data_set = None
    self._test_data_set = None
    self._low_res_generator = None
    self._epoch_sample_input = tf.random_normal([np.prod(self._epoch_images_shape), config.noise_dimensions])
    super(NoiseImageSuperResolutionEvaluation, self).__init__(model, config)

  def set_up_model(self):
    super(NoiseImageSuperResolutionEvaluation, self).set_up_model()

    assert self._config.alternate_model_name and self._config.alternate_eid, "Alternate model required for low-res generator"
    tf.logging.warning("Loading low-res generatior '{}' from '{}'".format(self._config.alternate_model_name, self._config.alternate_eid))
    main_model = self._config.model_name
    main_checkpoint_dir = self._config.checkpoint_dir
    self._config.model_name = self._config.alternate_model_name
    self._config.checkpoint_dir = os.path.join("output", self._config.alternate_eid, "checkpoints")
    self._low_res_generator = load_model(self._config).get_generator()
    load_checkpoint(self._config, generator=self._low_res_generator) # always load the latest checkpoint
    for layer in self._low_res_generator.layers:
      layer.trainable = False
    self._config.model_name = main_model
    self._config.checkpoint_dir = main_checkpoint_dir

    self._model.print_model_summary(self._generator, self._discriminator, self._low_res_generator(self.epoch_sample_input[:1]))

  def load_data(self):
    image_names = load_image_names(self._config.data_dir, self._config.match_pattern)
    target_images = load_images(image_names, self._config.data_dir, self._config.target_type,
        flip_lr=self._config.augmentation_flip_lr, flip_ud=self._config.augmentation_flip_ud)

    # load/prepare test data
    test_target_images = None
    if self._config.test_data_dir:
      tf.logging.info("Loading test data")
      test_image_names = load_image_names(self._config.test_data_dir, self._config.match_pattern)
      test_target_images = load_images(test_image_names, self._config.test_data_dir, self._config.target_type,
          flip_lr=False, flip_ud=False)
    if self._config.test_data_percentage:
      tf.logging.warning("Using the first {}% of the training data for testing".format(100*self._config.test_data_percentage))
      split = int(len(target_images) * self._config.test_data_percentage)
      test_target_images = target_images[:split]
      target_images = target_images[split:]
    data_set_size = len(target_images)
    test_set_size = len([] if test_target_images is None else test_target_images)

    # build data sets
    self._data_set = tf.data.Dataset.from_tensor_slices(target_images)\
        .map(lambda x: (tf.random_normal([self._config.noise_dimensions]), x))\
        .shuffle(data_set_size).batch(self._config.batch_size)
    del target_images
    if test_target_images is not None:
      self._test_data_set = tf.data.Dataset.from_tensor_slices(test_target_images)\
          .map(lambda x: (tf.random_normal([self._config.noise_dimensions]), x))\
          .shuffle(test_set_size).batch(self._config.batch_size)
      del test_target_images
    else:
      tf.logging.warning("Running evaluation without test data!")

    return data_set_size, test_set_size

  @property
  def data_set(self):
    return self._data_set

  @property
  def extra_discriminator_data_set(self):
    return None

  @property
  def test_data_set(self):
    return self._test_data_set

  @property
  def epoch_sample_input(self):
    return self._epoch_sample_input

  def _plot_epoch_samples(self, generator, discriminator):
    hires_samples = []
    lowres_samples = []
    predictions = []
    for batch in tf.data.Dataset.from_tensor_slices(self.epoch_sample_input).batch(self._config.batch_size):
      new_lowres_samples = self._low_res_generator(batch, training=True)
      new_samples = generator(new_lowres_samples, training=True)
      new_predictions = logistic(discriminator(new_samples, training=True))
      hires_samples.append(new_samples)
      lowres_samples.append(new_lowres_samples)
      predictions.append(new_predictions)

    hires_samples = tf.concat(hires_samples, axis=0)
    lowres_samples = tf.concat(lowres_samples, axis=0)
    predictions = tf.concat(predictions, axis=0)
    rows_and_cols = (self._epoch_images_shape[0], self._epoch_images_shape[1]*2)
    for i in range(hires_samples.shape[0]):
      plt.subplot(*rows_and_cols, 2*i+1)
      Evaluation.plot_image(lowres_samples[i], "{}x{}".format(lowres_samples.shape[1], lowres_samples.shape[2]))
      plt.subplot(*rows_and_cols, 2*i+2)
      Evaluation.plot_image(hires_samples[i], "{}x{}: {:.5f}".format(hires_samples.shape[1], hires_samples.shape[2], predictions[i].numpy()))
    return (hires_samples, lowres_samples), predictions

  def _plot_hq_epoch_samples(self, generated_samples, discriminator_probabilities):
    hires_samples, lowres_samples = generated_samples
    rows_and_cols = ((self._epoch_images_shape[0]+1)//2, self._epoch_images_shape[1]//2*2)
    for i in range(rows_and_cols[0] * rows_and_cols[1] // 2):
      plt.subplot(*rows_and_cols, 2*i+1)
      Evaluation.plot_image(lowres_samples[i], "{}x{}".format(lowres_samples.shape[1], lowres_samples.shape[2]))
      plt.subplot(*rows_and_cols, 2*i+2)
      Evaluation.plot_image(hires_samples[i], "{}x{}: {:.5f}".format(hires_samples.shape[1], hires_samples.shape[2], discriminator_probabilities[i]))


class ImageToImageEvaluation(OneWayEvaluation):
  def __init__(self, model, config):
    self._epoch_images_shape = (6, 2)
    self._data_set = None
    self._test_data_set = None
    self._epoch_sample_input = None
    self._epoch_sample_targets = None
    super(ImageToImageEvaluation, self).__init__(model, config)

  def load_data(self):
    # load and prepare names
    input_image_names = load_image_names(self._config.data_dir, self._config.match_pattern)
    target_image_names = input_image_names if not self._config.target_data_dir else \
        load_image_names(self._config.target_data_dir, self._config.match_pattern)
    name_size = min(len(input_image_names), len(target_image_names))
    if len(input_image_names) != len(target_image_names):
      tf.logging.warning("Reducing data set to {} (input: {}, target: {})".format(name_size,
        len(input_image_names), len(target_image_names)))
      input_image_names = input_image_names[:name_size]
      assert self._config.data_dir == self._config.target_data_dir, "should shuffle target images before reducing"
      target_image_names = target_image_names[:name_size]
    assert not self._config.train_disc_on_extra_targets, "not implemented"

    # load images
    input_images = load_images(input_image_names, self._config.data_dir, self._config.input_type,
        flip_lr=self._config.augmentation_flip_lr, flip_ud=self._config.augmentation_flip_ud)
    target_images = load_images(target_image_names, self._config.target_data_dir or self._config.data_dir, self._config.target_type,
        flip_lr=self._config.augmentation_flip_lr, flip_ud=self._config.augmentation_flip_ud)
    assert len(input_images) == len(target_images)

    # load/prepare test data
    test_input_images = None
    test_target_images = None
    if self._config.test_data_dir:
      assert not self._config.target_data_dir, "alternative test target data currently isn't supported"
      tf.logging.info("Loading test data")
      test_input_image_names = load_image_names(self._config.test_data_dir, self._config.match_pattern)
      test_input_images = load_images(test_input_image_names, self._config.test_data_dir, self._config.input_type,
          flip_lr=False, flip_ud=False)
      test_target_images = load_images(test_input_image_names, self._config.test_data_dir, self._config.target_type,
          flip_lr=False, flip_ud=False)
      assert len(test_input_images) == len(test_target_images)
    if self._config.test_data_percentage:
      tf.logging.warning("Using the first {}% of the training data for testing".format(100*self._config.test_data_percentage))
      split = int(len(target_images) * self._config.test_data_percentage)
      test_input_images = input_images[:split]
      input_images = input_images[split:]
      test_target_images = target_images[:split]
      target_images = target_images[split:]
    data_set_size = len(target_images)
    test_set_size = len([] if test_target_images is None else test_target_images)

    # set up epoch samples
    sample_indexes = np.random.choice(data_set_size, np.prod(self._epoch_images_shape), replace=False)
    self._epoch_sample_input = tf.convert_to_tensor(input_images[sample_indexes])
    self._epoch_sample_targets = target_images[sample_indexes]

    # build data sets
    self._data_set = tf.data.Dataset.from_tensor_slices((input_images, target_images))\
        .shuffle(data_set_size).batch(self._config.batch_size)
    del input_images
    del target_images
    if test_target_images is not None:
      self._test_data_set = tf.data.Dataset.from_tensor_slices((test_input_images, test_target_images))\
          .shuffle(test_set_size).batch(self._config.batch_size)
      del test_input_images
      del test_target_images
    else:
      tf.logging.warning("Running evaluation without test data!")

    return data_set_size, test_set_size

  @property
  def data_set(self):
    return self._data_set

  @property
  def extra_discriminator_data_set(self):
    return None

  @property
  def test_data_set(self):
    return self._test_data_set

  @property
  def epoch_sample_input(self):
    return self._epoch_sample_input

  def _plot_epoch_samples(self, generator, discriminator):
    samples = []
    predictions = []
    for batch in tf.data.Dataset.from_tensor_slices(self.epoch_sample_input).batch(self._config.batch_size):
      new_samples = generator(batch, training=True)
      disc_input = tf.concat([batch, new_samples], axis=-1) if self._config.conditioned_discriminator else new_samples
      new_predictions = logistic(discriminator(disc_input, training=True))
      samples.append(new_samples)
      predictions.append(new_predictions)

    images_per_sample = 3
    rows_and_cols = (self._epoch_images_shape[0], self._epoch_images_shape[1]*images_per_sample)
    samples = tf.concat(samples, axis=0)
    predictions = tf.concat(predictions, axis=0)
    for i in range(samples.shape[0]):
      plt.subplot(*rows_and_cols, (i*images_per_sample)+1)
      Evaluation.plot_image(self.epoch_sample_input[i])
      plt.subplot(*rows_and_cols, (i*images_per_sample)+2)
      Evaluation.plot_image(samples[i], np.round(predictions[i].numpy(), 5))
      plt.subplot(*rows_and_cols, (i*images_per_sample)+3)
      Evaluation.plot_image(self._epoch_sample_targets[i])
    return samples, predictions

  def _plot_hq_epoch_samples(self, generated_samples, discriminator_probabilities):
    images_per_sample = 3
    rows_and_cols = (self._epoch_images_shape[0]//2, self._epoch_images_shape[1]//2*images_per_sample)
    for i in range(generated_samples.shape[0]//4):
      plt.subplot(*rows_and_cols, (i*images_per_sample)+1)
      Evaluation.plot_image(self.epoch_sample_input[i])
      plt.subplot(*rows_and_cols, (i*images_per_sample)+2)
      Evaluation.plot_image(generated_samples[i], np.round(discriminator_probabilities[i].numpy(), 5))
      plt.subplot(*rows_and_cols, (i*images_per_sample)+3)
      Evaluation.plot_image(self._epoch_sample_targets[i])


class ImageInpaintingEvaluation(OneWayEvaluation):
  def __init__(self, model, config):
    self._epoch_images_shape = (6, 3)
    self._data_set = None
    self._extra_discriminator_data_set = None
    self._test_data_set = None
    self._epoch_sample_input = None
    assert config.has_colored_input and not config.has_colored_second_input and config.has_colored_target and not config.has_noise_input
    assert not config.conditioned_discriminator, "Doesn't make sense for D to process G input"
    super(ImageInpaintingEvaluation, self).__init__(model, config)

  def load_data(self):
    # load and prepare names
    input_image_names = load_image_names(self._config.data_dir, self._config.match_pattern)
    second_input_image_names = input_image_names if not self._config.second_data_dir else \
        load_image_names(self._config.second_data_dir, self._config.match_pattern)
    if self._config.use_extra_first_inputs:
      assert len(second_input_image_names) < len(input_image_names)
      second_input_times = 2
      tf.logging.warning("Using each second input {} times".format(second_input_times))
      second_input_image_names = second_input_image_names * second_input_times
    if len(second_input_image_names) < len(input_image_names):
      tf.logging.warning("There are fewer second input images; shuffling and reducing input images")
      np.random.shuffle(input_image_names)
      input_image_names = input_image_names[:len(second_input_image_names)]
    target_image_names = input_image_names if not self._config.target_data_dir else \
        second_input_image_names if self._config.target_data_dir == self._config.second_data_dir else \
        load_image_names(self._config.target_data_dir, self._config.match_pattern)
    assert len(target_image_names) >= len(input_image_names)
    name_size = min(min(len(input_image_names), len(second_input_image_names)), len(target_image_names))
    if len(input_image_names) != len(target_image_names) or len(second_input_image_names) != len(target_image_names):
      tf.logging.warning("Reducing data set to {} (input: {}, second input: {}, target: {})".format(name_size,
        len(input_image_names), len(second_input_image_names), len(target_image_names)))
      input_image_names = input_image_names[:name_size]
      second_input_image_names = second_input_image_names[:name_size]
      tf.logging.info("Input and target data are different; shuffling targets before reducing")
      np.random.shuffle(target_image_names)
      if self._config.train_disc_on_extra_targets:
        extra_target_image_names = target_image_names[name_size:]
        extra_target_image_names = extra_target_image_names[:len(input_image_names)] # select the first X extra images
      else:
        extra_target_image_names = None
      target_image_names = target_image_names[:name_size]
    else:
      assert not self._config.train_disc_on_extra_targets, "there are no extra targets to train on"
      extra_target_image_names = None

    # load images
    input_images = load_images(input_image_names, self._config.data_dir, self._config.input_type,
        flip_lr=self._config.augmentation_flip_lr, flip_ud=self._config.augmentation_flip_ud)
    second_input_images = load_images(second_input_image_names, self._config.second_data_dir or self._config.data_dir, self._config.second_input_type,
        flip_lr=self._config.augmentation_flip_lr, flip_ud=self._config.augmentation_flip_ud)
    combined_input_images = input_images * (second_input_images == -1)
    del input_images
    del second_input_images
    target_images = load_images(target_image_names, self._config.target_data_dir or self._config.data_dir, self._config.target_type,
        flip_lr=self._config.augmentation_flip_lr, flip_ud=self._config.augmentation_flip_ud)
    if extra_target_image_names:
      extra_target_images = load_images(extra_target_image_names,
          self._config.target_data_dir or self._config.data_dir, self._config.target_type,
          flip_lr=self._config.augmentation_flip_lr, flip_ud=self._config.augmentation_flip_ud)
      tf.logging.warning("Adding {} extra targets for the discriminator!".format(len(extra_target_images)))
      self._extra_discriminator_data_set = tf.data.Dataset.from_tensor_slices(extra_target_images)\
          .shuffle(len(extra_target_images)).batch(self._config.batch_size)
      del extra_target_images
    assert len(combined_input_images) == len(target_images)

    # load/prepare test data
    test_input_images = None
    test_target_images = None
    if self._config.test_data_dir:
      tf.logging.info("Loading test data")
      test_input_image_names = load_image_names(self._config.test_data_dir, self._config.match_pattern)
      test_second_input_image_names = test_input_image_names if not self._config.test_second_data_dir else \
          load_image_names(self._config.test_second_data_dir, self._config.match_pattern)
      if len(test_second_input_image_names) < len(test_input_image_names):
        tf.logging.warning("TEST: There are fewer second input images; shuffling and reducing input images")
        np.random.shuffle(test_input_image_names)
        test_input_image_names = test_input_image_names[:len(test_second_input_image_names)]
      test_target_image_names = test_input_image_names if not self._config.test_target_data_dir else \
          test_second_input_image_names if self._config.test_target_data_dir == self._config.test_second_data_dir else \
          load_image_names(self._config.test_target_data_dir, self._config.match_pattern)
      assert len(test_target_image_names) >= len(test_input_image_names)
      name_size = min(min(len(test_input_image_names), len(test_second_input_image_names)), len(test_target_image_names))
      if len(test_input_image_names) != len(test_target_image_names) or len(test_second_input_image_names) != len(test_target_image_names):
        tf.logging.warning("Reducing data set to {} (input: {}, second input: {}, target: {})".format(name_size,
          len(test_input_image_names), len(test_second_input_image_names), len(test_target_image_names)))
        test_input_image_names = test_input_image_names[:name_size]
        test_second_input_image_names = test_second_input_image_names[:name_size]
        tf.logging.info("Input and target data are different; shuffling targets before reducing")
        np.random.shuffle(test_target_image_names)
        test_target_image_names = test_target_image_names[:name_size]

      test_input_images = load_images(test_input_image_names, self._config.test_data_dir, self._config.input_type,
          flip_lr=False, flip_ud=False)
      test_second_input_images = load_images(test_second_input_image_names,
          self._config.test_second_data_dir or self._config.test_data_dir, self._config.second_input_type, flip_lr=False, flip_ud=False)
      test_combined_input_images = np.concatenate([test_input_images, test_second_input_images], axis=-1)
      del test_input_images
      del test_second_input_images
      test_target_images = load_images(test_target_image_names,
          self._config.test_target_data_dir or self._config.test_data_dir, self._config.target_type, flip_lr=False, flip_ud=False)
      assert len(test_combined_input_images) == len(test_target_images)

    if self._config.test_data_percentage:
      tf.logging.warning("Using the first {}% of the training data for testing".format(100*self._config.test_data_percentage))
      split = int(len(target_images) * self._config.test_data_percentage)
      test_combined_input_images = combined_input_images[:split]
      combined_input_images = combined_input_images[split:]
      test_target_images = target_images[:split]
      target_images = target_images[split:]
      assert False, "not implemented"
    data_set_size = len(target_images)
    test_set_size = len([] if test_target_images is None else test_target_images)

    # set up epoch samples
    sample_indexes = np.random.choice(data_set_size, np.prod(self._epoch_images_shape), replace=False)
    self._epoch_sample_input = tf.convert_to_tensor(combined_input_images[sample_indexes])

    # build data sets
    self._data_set = tf.data.Dataset.from_tensor_slices((combined_input_images, target_images))\
        .shuffle(data_set_size).batch(self._config.batch_size)
    del combined_input_images
    del target_images
    if test_target_images is not None:
      self._test_data_set = tf.data.Dataset.from_tensor_slices((test_combined_input_images, test_target_images))\
          .shuffle(test_set_size).batch(self._config.batch_size)
      del test_combined_input_images
      del test_target_images
    else:
      tf.logging.warning("Running evaluation without test data!")

    return data_set_size, test_set_size

  @property
  def data_set(self):
    return self._data_set

  @property
  def extra_discriminator_data_set(self):
    return self._extra_discriminator_data_set

  @property
  def test_data_set(self):
    return self._test_data_set

  @property
  def epoch_sample_input(self):
    return self._epoch_sample_input

  def _plot_epoch_samples(self, generator, discriminator):
    samples = []
    predictions = []
    for batch in tf.data.Dataset.from_tensor_slices(self.epoch_sample_input).batch(self._config.batch_size):
      new_samples = generator(batch, training=True)
      disc_input = tf.concat([batch, new_samples], axis=-1) if self._config.conditioned_discriminator else new_samples
      new_predictions = logistic(discriminator(disc_input, training=True))
      samples.append(new_samples)
      predictions.append(new_predictions)

    images_per_sample = 2
    rows_and_cols = (self._epoch_images_shape[0], self._epoch_images_shape[1]*images_per_sample)
    samples = tf.concat(samples, axis=0)
    predictions = tf.concat(predictions, axis=0)
    for i in range(samples.shape[0]):
      plt.subplot(*rows_and_cols, (i*images_per_sample)+1)
      Evaluation.plot_image(self.epoch_sample_input[i])
      plt.subplot(*rows_and_cols, (i*images_per_sample)+2)
      Evaluation.plot_image(samples[i], np.round(predictions[i].numpy(), 5))
    return samples, predictions

  def _plot_hq_epoch_samples(self, generated_samples, discriminator_probabilities):
    images_per_sample = 2
    rows_and_cols = (4, 2*images_per_sample)
    for i in range(np.prod(rows_and_cols)//images_per_sample):
      plt.subplot(*rows_and_cols, (i*images_per_sample)+1)
      Evaluation.plot_image(self.epoch_sample_input[i])
      plt.subplot(*rows_and_cols, (i*images_per_sample)+2)
      Evaluation.plot_image(generated_samples[i], np.round(discriminator_probabilities[i].numpy(), 5))

class TwoImagesToOneImageEvaluation(OneWayEvaluation):
  def __init__(self, model, config):
    self._epoch_images_shape = (6, 2)
    self._data_set = None
    self._extra_discriminator_data_set = None
    self._test_data_set = None
    self._epoch_sample_input = None
    assert config.has_colored_input and not config.has_colored_second_input and config.has_colored_target and not config.has_noise_input
    super(TwoImagesToOneImageEvaluation, self).__init__(model, config)

  def load_data(self):
    # load and prepare names
    input_image_names = load_image_names(self._config.data_dir, self._config.match_pattern)
    second_input_image_names = input_image_names if not self._config.second_data_dir else \
        load_image_names(self._config.second_data_dir, self._config.match_pattern)
    if self._config.use_extra_first_inputs:
      assert len(second_input_image_names) < len(input_image_names)
      second_input_times = 2
      tf.logging.warning("Using each second input {} times".format(second_input_times))
      second_input_image_names = second_input_image_names * second_input_times
    if len(second_input_image_names) < len(input_image_names):
      tf.logging.warning("There are fewer second input images; shuffling and reducing input images")
      np.random.shuffle(input_image_names)
      input_image_names = input_image_names[:len(second_input_image_names)]
    target_image_names = input_image_names if not self._config.target_data_dir else \
        second_input_image_names if self._config.target_data_dir == self._config.second_data_dir else \
        load_image_names(self._config.target_data_dir, self._config.match_pattern)
    assert len(target_image_names) >= len(input_image_names)
    name_size = min(min(len(input_image_names), len(second_input_image_names)), len(target_image_names))
    if len(input_image_names) != len(target_image_names) or len(second_input_image_names) != len(target_image_names):
      tf.logging.warning("Reducing data set to {} (input: {}, second input: {}, target: {})".format(name_size,
        len(input_image_names), len(second_input_image_names), len(target_image_names)))
      input_image_names = input_image_names[:name_size]
      second_input_image_names = second_input_image_names[:name_size]
      tf.logging.info("Input and target data are different; shuffling targets before reducing")
      np.random.shuffle(target_image_names)
      if self._config.train_disc_on_extra_targets:
        extra_target_image_names = target_image_names[name_size:]
        extra_target_image_names = extra_target_image_names[:2*len(input_image_names)] # select the first X extra images
      else:
        extra_target_image_names = None
      target_image_names = target_image_names[:name_size]
    else:
      assert not self._config.train_disc_on_extra_targets, "there are no extra targets to train on"
      extra_target_image_names = None

    # load images
    input_images = load_images(input_image_names, self._config.data_dir, self._config.input_type,
        flip_lr=self._config.augmentation_flip_lr, flip_ud=self._config.augmentation_flip_ud)
    second_input_images = load_images(second_input_image_names, self._config.second_data_dir or self._config.data_dir, self._config.second_input_type,
        flip_lr=self._config.augmentation_flip_lr, flip_ud=self._config.augmentation_flip_ud)
    combined_input_images = np.concatenate([input_images, second_input_images], axis=-1)
    del input_images
    del second_input_images
    target_images = load_images(target_image_names, self._config.target_data_dir or self._config.data_dir, self._config.target_type,
        flip_lr=self._config.augmentation_flip_lr, flip_ud=self._config.augmentation_flip_ud)
    if extra_target_image_names:
      extra_target_images = load_images(extra_target_image_names,
          self._config.target_data_dir or self._config.data_dir, self._config.target_type,
          flip_lr=self._config.augmentation_flip_lr, flip_ud=self._config.augmentation_flip_ud)
      tf.logging.warning("Adding {} extra targets for the discriminator!".format(len(extra_target_images)))
      self._extra_discriminator_data_set = tf.data.Dataset.from_tensor_slices(extra_target_images)\
          .shuffle(len(extra_target_images)).batch(self._config.batch_size)
      del extra_target_images
    assert len(combined_input_images) == len(target_images)

    # load/prepare test data
    test_input_images = None
    test_target_images = None
    if self._config.test_data_dir:
      tf.logging.info("Loading test data")
      test_input_image_names = load_image_names(self._config.test_data_dir, self._config.match_pattern)
      test_second_input_image_names = test_input_image_names if not self._config.test_second_data_dir else \
          load_image_names(self._config.test_second_data_dir, self._config.match_pattern)
      if len(test_second_input_image_names) < len(test_input_image_names):
        tf.logging.warning("TEST: There are fewer second input images; shuffling and reducing input images")
        np.random.shuffle(test_input_image_names)
        test_input_image_names = test_input_image_names[:len(test_second_input_image_names)]
      test_target_image_names = test_input_image_names if not self._config.test_target_data_dir else \
          test_second_input_image_names if self._config.test_target_data_dir == self._config.test_second_data_dir else \
          load_image_names(self._config.test_target_data_dir, self._config.match_pattern)
      assert len(test_target_image_names) >= len(test_input_image_names)
      name_size = min(min(len(test_input_image_names), len(test_second_input_image_names)), len(test_target_image_names))
      if len(test_input_image_names) != len(test_target_image_names) or len(test_second_input_image_names) != len(test_target_image_names):
        tf.logging.warning("Reducing data set to {} (input: {}, second input: {}, target: {})".format(name_size,
          len(test_input_image_names), len(test_second_input_image_names), len(test_target_image_names)))
        test_input_image_names = test_input_image_names[:name_size]
        test_second_input_image_names = test_second_input_image_names[:name_size]
        tf.logging.info("Input and target data are different; shuffling targets before reducing")
        np.random.shuffle(test_target_image_names)
        test_target_image_names = test_target_image_names[:name_size]

      test_input_images = load_images(test_input_image_names, self._config.test_data_dir, self._config.input_type,
          flip_lr=False, flip_ud=False)
      test_second_input_images = load_images(test_second_input_image_names,
          self._config.test_second_data_dir or self._config.test_data_dir, self._config.second_input_type, flip_lr=False, flip_ud=False)
      test_combined_input_images = np.concatenate([test_input_images, test_second_input_images], axis=-1)
      del test_input_images
      del test_second_input_images
      test_target_images = load_images(test_target_image_names,
          self._config.test_target_data_dir or self._config.test_data_dir, self._config.target_type, flip_lr=False, flip_ud=False)
      assert len(test_combined_input_images) == len(test_target_images)

    if self._config.test_data_percentage:
      tf.logging.warning("Using the first {}% of the training data for testing".format(100*self._config.test_data_percentage))
      split = int(len(target_images) * self._config.test_data_percentage)
      test_combined_input_images = combined_input_images[:split]
      combined_input_images = combined_input_images[split:]
      test_target_images = target_images[:split]
      target_images = target_images[split:]
      assert False, "not implemented"
    data_set_size = len(target_images)
    test_set_size = len([] if test_target_images is None else test_target_images)

    # set up epoch samples
    sample_indexes = np.random.choice(data_set_size, np.prod(self._epoch_images_shape), replace=False)
    self._epoch_sample_input = tf.convert_to_tensor(combined_input_images[sample_indexes])

    # build data sets
    self._data_set = tf.data.Dataset.from_tensor_slices((combined_input_images, target_images))\
        .shuffle(data_set_size).batch(self._config.batch_size)
    del combined_input_images
    del target_images
    if test_set_size:
      self._test_data_set = tf.data.Dataset.from_tensor_slices((test_combined_input_images, test_target_images))\
          .shuffle(test_set_size).batch(self._config.batch_size)
      del test_combined_input_images
      del test_target_images
    else:
      tf.logging.warning("Running evaluation without test data!")

    return data_set_size, test_set_size

  @property
  def data_set(self):
    return self._data_set

  @property
  def extra_discriminator_data_set(self):
    return self._extra_discriminator_data_set

  @property
  def test_data_set(self):
    return self._test_data_set

  @property
  def epoch_sample_input(self):
    return self._epoch_sample_input

  def _plot_epoch_samples(self, generator, discriminator):
    samples = []
    predictions = []
    for batch in tf.data.Dataset.from_tensor_slices(self.epoch_sample_input).batch(self._config.batch_size):
      new_samples = generator(batch, training=True)
      disc_input = tf.concat([batch, new_samples], axis=-1) if self._config.conditioned_discriminator else new_samples
      new_predictions = logistic(discriminator(disc_input, training=True))
      samples.append(new_samples)
      predictions.append(new_predictions)

    images_per_sample = 3
    samples = tf.concat(samples, axis=0)
    predictions = tf.concat(predictions, axis=0)
    rows_and_cols = (self._epoch_images_shape[0], self._epoch_images_shape[1]*images_per_sample)
    for i in range(self.epoch_sample_input.shape[0]):
      # NOTE: this assumes colored+B/W -> colored
      plt.subplot(*rows_and_cols, (i*images_per_sample)+1)
      Evaluation.plot_image(self.epoch_sample_input[i, :, :, :-1])
      plt.subplot(*rows_and_cols, (i*images_per_sample)+2)
      Evaluation.plot_image(self.epoch_sample_input[i, :, :, -1:])
      plt.subplot(*rows_and_cols, (i*images_per_sample)+3)
      Evaluation.plot_image(samples[i], np.round(predictions[i].numpy(), 5))
    return samples, predictions

  def _plot_hq_epoch_samples(self, generated_samples, discriminator_probabilities):
    images_per_sample = 3
    rows_and_cols = (self._epoch_images_shape[0]//2, self._epoch_images_shape[1]//2*images_per_sample)
    for i in range(generated_samples.shape[0]//4):
      plt.subplot(*rows_and_cols, (i*images_per_sample)+1)
      Evaluation.plot_image(self.epoch_sample_input[i, :, :, :-1])
      plt.subplot(*rows_and_cols, (i*images_per_sample)+2)
      Evaluation.plot_image(self.epoch_sample_input[i, :, :, -1:])
      plt.subplot(*rows_and_cols, (i*images_per_sample)+3)
      Evaluation.plot_image(generated_samples[i], np.round(discriminator_probabilities[i].numpy(), 5))
