#!/usr/bin/env python

# pylint: disable=wrong-import-position

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from evaluation import Evaluation
from two_way_evaluation import TwoWayEvaluation
from utils import load_image_names, load_images, logistic


class TwoWayImageToImageEvaluation(TwoWayEvaluation):
  def __init__(self, model, config):
    self._epoch_images_shape = (7, 1)
    self._data_set = None
    self._test_data_set = None
    self._extra_discriminator_data_set = None
    self._epoch_sample_input = None
    self._epoch_sample_targets = None
    assert not config.has_noise_input
    super(TwoWayImageToImageEvaluation, self).__init__(model, config)

  def load_data(self):
    # load and prepare names
    input_image_names = load_image_names(self._config.data_dir, self._config.match_pattern)
    assert not self._config.use_extra_first_inputs, "not implemented"
    target_image_names = input_image_names if not self._config.target_data_dir else \
        load_image_names(self._config.target_data_dir, self._config.match_pattern)
    assert len(target_image_names) >= len(input_image_names)
    name_size = min(len(input_image_names), len(target_image_names))
    if len(input_image_names) != len(target_image_names):
      tf.logging.warning("Reducing data set to {} (input: {}, target: {})".format(name_size,
        len(input_image_names), len(target_image_names)))
      input_image_names = input_image_names[:name_size]
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
    assert len(input_images) == len(target_images)
    data_set_size = len(target_images)

    # load/prepare test data
    test_input_images = None
    test_target_images = None
    if self._config.test_data_dir:
      tf.logging.info("Loading test data")
      test_input_image_names = load_image_names(self._config.test_data_dir, self._config.match_pattern)
      test_target_image_names = test_input_image_names if not self._config.test_target_data_dir else \
          load_image_names(self._config.test_target_data_dir, self._config.match_pattern)
      assert len(test_target_image_names) >= len(test_input_image_names)
      name_size = min(len(test_input_image_names), len(test_target_image_names))
      if len(test_input_image_names) != len(test_target_image_names):
        tf.logging.warning("Reducing data set to {} (input: {}, target: {})".format(name_size,
          len(test_input_image_names), len(test_target_image_names)))
        test_input_image_names = test_input_image_names[:name_size]
        tf.logging.info("Input and target data are different; shuffling targets before reducing")
        np.random.shuffle(test_target_image_names)
        test_target_image_names = test_target_image_names[:name_size]

      test_input_images = load_images(test_input_image_names, self._config.test_data_dir, self._config.input_type,
          flip_lr=False, flip_ud=False)
      test_target_images = load_images(test_target_image_names,
          self._config.test_target_data_dir or self._config.test_data_dir, self._config.target_type, flip_lr=False, flip_ud=False)
      assert len(test_input_images) == len(test_target_images)

    assert not self._config.test_data_percentage, "not implemented"
    test_set_size = len([] if test_target_images is None else test_target_images)

    # set up epoch samples
    sample_indexes = np.random.choice(data_set_size, np.prod(self._epoch_images_shape), replace=False)
    self._epoch_sample_input = tf.convert_to_tensor(input_images[sample_indexes])
    self._epoch_sample_targets = tf.convert_to_tensor(target_images[sample_indexes])

    # build data sets
    self._data_set = tf.data.Dataset.from_tensor_slices((input_images, tf.random_shuffle(target_images)))\
        .shuffle(data_set_size).batch(self._config.batch_size)
    del input_images
    del target_images
    if test_set_size:
      self._test_data_set = tf.data.Dataset.from_tensor_slices((test_input_images, tf.random_shuffle(test_target_images)))\
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
    return self._extra_discriminator_data_set

  @property
  def test_data_set(self):
    return self._test_data_set

  @property
  def epoch_sample_input(self):
    return self._epoch_sample_input

  def _plot_epoch_samples(self, generator, discriminator):
    first_generator, second_generator = generator
    first_discriminator, second_discriminator = discriminator

    first_samples = []
    first_reconstructions = []
    first_predictions = []
    first_reconstruction_predictions = []
    for batch in tf.data.Dataset.from_tensor_slices(self.epoch_sample_input).batch(self._config.batch_size):
      new_samples = first_generator(batch, training=True)
      first_samples.append(new_samples)
      disc_input = tf.concat([batch, new_samples], axis=-1) if self._config.conditioned_discriminator else new_samples
      first_predictions.append(logistic(second_discriminator(disc_input, training=True)))
      new_reconstructions = second_generator(new_samples, training=True)
      first_reconstructions.append(new_reconstructions)
      disc_input = tf.concat([new_reconstructions, new_samples], axis=-1) if self._config.conditioned_discriminator else new_reconstructions
      first_reconstruction_predictions.append(logistic(first_discriminator(disc_input, training=True)))

    second_samples = []
    second_reconstructions = []
    second_predictions = []
    second_reconstruction_predictions = []
    for batch in tf.data.Dataset.from_tensor_slices(self._epoch_sample_targets).batch(self._config.batch_size):
      new_samples = second_generator(batch, training=True)
      second_samples.append(new_samples)
      disc_input = tf.concat([batch, new_samples], axis=-1) if self._config.conditioned_discriminator else new_samples
      second_predictions.append(logistic(first_discriminator(disc_input, training=True)))
      new_reconstructions = first_generator(new_samples, training=True)
      second_reconstructions.append(new_reconstructions)
      disc_input = tf.concat([new_reconstructions, new_samples], axis=-1) if self._config.conditioned_discriminator else new_reconstructions
      second_reconstruction_predictions.append(logistic(second_discriminator(disc_input, training=True)))

    first_samples = tf.concat(first_samples, axis=0)
    first_reconstructions = tf.concat(first_reconstructions, axis=0)
    first_predictions = tf.concat(first_predictions, axis=0)
    first_reconstruction_predictions = tf.concat(first_reconstruction_predictions, axis=0)
    second_samples = tf.concat(second_samples, axis=0)
    second_reconstructions = tf.concat(second_reconstructions, axis=0)
    second_predictions = tf.concat(second_predictions, axis=0)
    second_reconstruction_predictions = tf.concat(second_reconstruction_predictions, axis=0)

    images_per_row = 6
    rows_and_cols = (self._epoch_images_shape[0], self._epoch_images_shape[1]*images_per_row)
    for i in range(self._epoch_sample_input.shape[0]):
      # first column: first gen input
      plt.subplot(*rows_and_cols, (i*images_per_row)+1)
      Evaluation.plot_image(self._epoch_sample_input[i])
      # first gen output
      plt.subplot(*rows_and_cols, (i*images_per_row)+2)
      Evaluation.plot_image(first_samples[i], np.round(first_predictions[i].numpy(), 5))
      # second gen reconstruction
      plt.subplot(*rows_and_cols, (i*images_per_row)+3)
      Evaluation.plot_image(first_reconstructions[i], np.round(first_reconstruction_predictions[i].numpy(), 5))
      # second gen input
      plt.subplot(*rows_and_cols, (i*images_per_row)+4)
      Evaluation.plot_image(self._epoch_sample_targets[i])
      # second gen output
      plt.subplot(*rows_and_cols, (i*images_per_row)+5)
      Evaluation.plot_image(second_samples[i], np.round(second_predictions[i].numpy(), 5))
      # first gen reconstruction
      plt.subplot(*rows_and_cols, (i*images_per_row)+6)
      Evaluation.plot_image(second_reconstructions[i], np.round(second_reconstruction_predictions[i].numpy(), 5))
    return (((first_samples, first_reconstructions), (second_samples, second_reconstructions)),
        ((first_predictions, first_reconstruction_predictions), (second_predictions, second_reconstruction_predictions)))

  def _plot_hq_epoch_samples(self, generated_samples, discriminator_probabilities):
    (first_samples, first_reconstructions), (second_samples, second_reconstructions) = generated_samples
    (first_predictions, first_reconstruction_predictions), (second_predictions, second_reconstruction_predictions) = discriminator_probabilities
    images_per_row = 3
    rows_and_cols = (4, self._epoch_images_shape[1]*images_per_row)
    for i in range(rows_and_cols[0] // 2): # first two rows: FWD
      # first column: first gen input
      plt.subplot(*rows_and_cols, (i*images_per_row)+1)
      Evaluation.plot_image(self._epoch_sample_input[i])
      # first gen output
      plt.subplot(*rows_and_cols, (i*images_per_row)+2)
      Evaluation.plot_image(first_samples[i], np.round(first_predictions[i].numpy(), 5))
      # second gen reconstruction
      plt.subplot(*rows_and_cols, (i*images_per_row)+3)
      Evaluation.plot_image(first_reconstructions[i], np.round(first_reconstruction_predictions[i].numpy(), 5))
    for i in range(rows_and_cols[0] // 2, rows_and_cols[0]): # last two rows: BWD
      # first column: second gen input
      plt.subplot(*rows_and_cols, (i*images_per_row)+1)
      Evaluation.plot_image(self._epoch_sample_targets[i-rows_and_cols[0]//2])
      # second gen output
      plt.subplot(*rows_and_cols, (i*images_per_row)+2)
      Evaluation.plot_image(second_samples[i-rows_and_cols[0]//2], np.round(second_predictions[i-rows_and_cols[0]//2].numpy(), 5))
      # first gen reconstruction
      plt.subplot(*rows_and_cols, (i*images_per_row)+3)
      Evaluation.plot_image(second_reconstructions[i-rows_and_cols[0]//2],
          np.round(second_reconstruction_predictions[i-rows_and_cols[0]//2].numpy(), 5))


class TwoWayTwoImagesToImageEvaluation(TwoWayEvaluation):
  def __init__(self, model, config):
    self._epoch_images_shape = (7, 1)
    self._data_set = None
    self._test_data_set = None
    self._extra_discriminator_data_set = None
    self._epoch_sample_input = None
    self._epoch_sample_targets = None
    assert config.has_colored_input and not config.has_colored_second_input and config.has_colored_target and not config.has_noise_input
    super(TwoWayTwoImagesToImageEvaluation, self).__init__(model, config)

  def load_data(self):
    # load and prepare names
    input_image_names = load_image_names(self._config.data_dir, self._config.match_pattern)
    second_input_image_names = input_image_names if not self._config.second_data_dir else \
        load_image_names(self._config.second_data_dir, self._config.match_pattern)
    assert not self._config.use_extra_first_inputs, "not implemented"
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
    combined_input_images = np.concatenate([input_images, second_input_images], axis=-1)
    del input_images
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
    combined_target_images = np.concatenate([target_images, second_input_images], axis=-1)
    data_set_size = len(target_images)
    del second_input_images
    del target_images

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
      test_target_images = load_images(test_target_image_names,
          self._config.test_target_data_dir or self._config.test_data_dir, self._config.target_type, flip_lr=False, flip_ud=False)
      test_combined_target_images = np.concatenate([test_target_images, test_second_input_images], axis=-1)
      assert len(test_combined_input_images) == len(test_target_images)
      del test_second_input_images

    assert not self._config.test_data_percentage, "not implemented"
    test_set_size = len([] if test_target_images is None else test_target_images)
    del test_target_images

    # set up epoch samples
    sample_indexes = np.random.choice(data_set_size, np.prod(self._epoch_images_shape), replace=False)
    self._epoch_sample_input = tf.convert_to_tensor(combined_input_images[sample_indexes])
    self._epoch_sample_targets = tf.convert_to_tensor(combined_target_images[sample_indexes])

    # build data sets
    self._data_set = tf.data.Dataset.from_tensor_slices((combined_input_images, tf.random_shuffle(combined_target_images)))\
        .shuffle(data_set_size).batch(self._config.batch_size)
    del combined_input_images
    del combined_target_images
    if test_set_size:
      self._test_data_set = tf.data.Dataset.from_tensor_slices((test_combined_input_images, tf.random_shuffle(test_combined_target_images)))\
          .shuffle(test_set_size).batch(self._config.batch_size)
      del test_combined_input_images
      del test_combined_target_images
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
    first_generator, second_generator = generator
    first_discriminator, second_discriminator = discriminator

    first_samples = []
    first_reconstructions = []
    first_predictions = []
    first_reconstruction_predictions = []
    for batch in tf.data.Dataset.from_tensor_slices(self.epoch_sample_input).batch(self._config.batch_size):
      segmentations = batch[:, :, :, -1:]
      new_samples = first_generator(batch, training=True)
      first_samples.append(new_samples)
      disc_input = tf.concat([batch, new_samples], axis=-1) if self._config.conditioned_discriminator else new_samples
      first_predictions.append(logistic(second_discriminator(disc_input, training=True)))
      new_reconstructions = second_generator(tf.concat([new_samples, segmentations], axis=-1), training=True)
      first_reconstructions.append(new_reconstructions)
      disc_input = tf.concat([batch, new_reconstructions], axis=-1) if self._config.conditioned_discriminator else new_reconstructions
      first_reconstruction_predictions.append(logistic(first_discriminator(disc_input, training=True)))

    second_samples = []
    second_reconstructions = []
    second_predictions = []
    second_reconstruction_predictions = []
    for batch in tf.data.Dataset.from_tensor_slices(self._epoch_sample_targets).batch(self._config.batch_size):
      segmentations = batch[:, :, :, -1:]
      new_samples = second_generator(batch, training=True)
      second_samples.append(new_samples)
      disc_input = tf.concat([batch, new_samples], axis=-1) if self._config.conditioned_discriminator else new_samples
      second_predictions.append(logistic(first_discriminator(disc_input, training=True)))
      new_reconstructions = first_generator(tf.concat([new_samples, segmentations], axis=-1), training=True)
      second_reconstructions.append(new_reconstructions)
      disc_input = tf.concat([batch, new_reconstructions], axis=-1) if self._config.conditioned_discriminator else new_reconstructions
      second_reconstruction_predictions.append(logistic(second_discriminator(disc_input, training=True)))

    first_samples = tf.concat(first_samples, axis=0)
    first_reconstructions = tf.concat(first_reconstructions, axis=0)
    first_predictions = tf.concat(first_predictions, axis=0)
    first_reconstruction_predictions = tf.concat(first_reconstruction_predictions, axis=0)
    second_samples = tf.concat(second_samples, axis=0)
    second_reconstructions = tf.concat(second_reconstructions, axis=0)
    second_predictions = tf.concat(second_predictions, axis=0)
    second_reconstruction_predictions = tf.concat(second_reconstruction_predictions, axis=0)

    images_per_row = 7
    rows_and_cols = (self._epoch_images_shape[0], self._epoch_images_shape[1]*images_per_row)
    for i in range(self._epoch_sample_input.shape[0]):
      # NOTE: this assumes colored+B/W -> colored
      # first column: segmentation
      plt.subplot(*rows_and_cols, (i*images_per_row)+1)
      Evaluation.plot_image(self._epoch_sample_input[i, :, :, -1:])
      # first gen input
      plt.subplot(*rows_and_cols, (i*images_per_row)+2)
      Evaluation.plot_image(self._epoch_sample_input[i, :, :, :-1])
      # first gen output
      plt.subplot(*rows_and_cols, (i*images_per_row)+3)
      Evaluation.plot_image(first_samples[i], np.round(first_predictions[i].numpy(), 5))
      # second gen reconstruction
      plt.subplot(*rows_and_cols, (i*images_per_row)+4)
      Evaluation.plot_image(first_reconstructions[i], np.round(first_reconstruction_predictions[i].numpy(), 5))
      # second gen input
      plt.subplot(*rows_and_cols, (i*images_per_row)+5)
      Evaluation.plot_image(self._epoch_sample_targets[i, :, :, :-1])
      # second gen output
      plt.subplot(*rows_and_cols, (i*images_per_row)+6)
      Evaluation.plot_image(second_samples[i], np.round(second_predictions[i].numpy(), 5))
      # first gen reconstruction
      plt.subplot(*rows_and_cols, (i*images_per_row)+7)
      Evaluation.plot_image(second_reconstructions[i], np.round(second_reconstruction_predictions[i].numpy(), 5))
    return (((first_samples, first_reconstructions), (second_samples, second_reconstructions)),
        ((first_predictions, first_reconstruction_predictions), (second_predictions, second_reconstruction_predictions)))

  def _plot_hq_epoch_samples(self, generated_samples, discriminator_probabilities):
    (first_samples, first_reconstructions), (second_samples, second_reconstructions) = generated_samples
    (first_predictions, first_reconstruction_predictions), (second_predictions, second_reconstruction_predictions) = discriminator_probabilities
    images_per_row = 4
    rows_and_cols = (4, self._epoch_images_shape[1]*images_per_row)
    for i in range(rows_and_cols[0] // 2): # first two rows: FWD
      # first column: segmentation
      plt.subplot(*rows_and_cols, (i*images_per_row)+1)
      Evaluation.plot_image(self._epoch_sample_input[i, :, :, -1:])
      # first gen input
      plt.subplot(*rows_and_cols, (i*images_per_row)+2)
      Evaluation.plot_image(self._epoch_sample_input[i, :, :, :-1])
      # first gen output
      plt.subplot(*rows_and_cols, (i*images_per_row)+3)
      Evaluation.plot_image(first_samples[i], np.round(first_predictions[i].numpy(), 5))
      # second gen reconstruction
      plt.subplot(*rows_and_cols, (i*images_per_row)+4)
      Evaluation.plot_image(first_reconstructions[i], np.round(first_reconstruction_predictions[i].numpy(), 5))
    for i in range(rows_and_cols[0] // 2, rows_and_cols[0]): # last two rows: BWD
      # first column: segmentation
      plt.subplot(*rows_and_cols, (i*images_per_row)+1)
      Evaluation.plot_image(self._epoch_sample_input[i-rows_and_cols[0]//2, :, :, -1:])
      # second gen input
      plt.subplot(*rows_and_cols, (i*images_per_row)+2)
      Evaluation.plot_image(self._epoch_sample_targets[i-rows_and_cols[0]//2, :, :, :-1])
      # second gen output
      plt.subplot(*rows_and_cols, (i*images_per_row)+3)
      Evaluation.plot_image(second_samples[i-rows_and_cols[0]//2], np.round(second_predictions[i-rows_and_cols[0]//2].numpy(), 5))
      # first gen reconstruction
      plt.subplot(*rows_and_cols, (i*images_per_row)+4)
      Evaluation.plot_image(second_reconstructions[i-rows_and_cols[0]//2],
          np.round(second_reconstruction_predictions[i-rows_and_cols[0]//2].numpy(), 5))
