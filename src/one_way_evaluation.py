#!/usr/bin/env python

import os
import pickle
import time
from abc import abstractmethod
from datetime import datetime, timedelta

import numpy as np
import tensorflow as tf

from evaluation import Evaluation
from generator_loss_args import GeneratorLossArgs
from one_way_metrics import OneWayMetrics
from perceptual_scores import PerceptualScores
from utils import get_memory_usage_string, logistic

# pylint: disable=too-many-statements

class OneWayEvaluation(Evaluation):
  def __init__(self, model, config):
    self._generator = None
    self._generator_optimizer = None
    self._discriminator = None
    self._discriminator_optimizer = None
    self._checkpoint = None
    self._final_checkpoint = None
    self._perceptual_scores = PerceptualScores(config) if config.target_type == "image" else None
    super(OneWayEvaluation, self).__init__(model, config)

  def set_up_model(self):
    tf.logging.info("Setting up models with learing rate {} for G, {} for D".format(
      self._model.gen_learning, self._model.disc_learning))
    self._generator = self._model.get_generator()
    self._discriminator = self._model.get_discriminator()
    # defun gives 10 secs/epoch performance boost
    self._generator.call = tf.contrib.eager.defun(self._generator.call)
    self._discriminator.call = tf.contrib.eager.defun(self._discriminator.call)

    self._generator_optimizer = tf.train.AdamOptimizer(self._model.gen_learning)
    self._discriminator_optimizer = tf.train.AdamOptimizer(self._model.disc_learning)
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=self._generator_optimizer,
        discriminator_optimizer=self._discriminator_optimizer,
        generator=self._generator,
        discriminator=self._discriminator)
    self._checkpoint = tf.contrib.checkpoint.CheckpointManager(checkpoint, self._config.checkpoint_dir,
        max_to_keep=None if self._config.keep_all_checkpoints else 5)
    if self._config.keep_final_checkpoints:
      final_checkpoint = tf.train.Checkpoint(
          generator_optimizer=self._generator_optimizer,
          discriminator_optimizer=self._discriminator_optimizer,
          generator=self._generator,
          discriminator=self._discriminator)
      self._final_checkpoint = tf.contrib.checkpoint.CheckpointManager(final_checkpoint,
          self._config.final_checkpoint_dir, max_to_keep=None)

    try:
      self._model.print_model_summary(self._generator, self._discriminator, self.epoch_sample_input)
    except Exception as ex:
      tf.logging.warning("Unable to print model summary ({}: {})".format(ex.__class__.__name__, ex))

    if self._perceptual_scores:
      self._perceptual_scores.initialize()

  @property
  @abstractmethod
  def data_set(self):
    """
    The data set to train on. Each batch should consist of a tuple of generator inputs and real
    target samples.
    """
    pass

  @property
  @abstractmethod
  def extra_discriminator_data_set(self):
    """
    The data set of additional real samples for the discriminator to train on.
    Only makes sense for non-conditioned discriminators.
    """
    pass

  @property
  @abstractmethod
  def test_data_set(self):
    """
    The data set to test on. Each batch should consist of a tuple of generator inputs and real
    target samples - same as the main data set.
    """
    pass

  @property
  @abstractmethod
  def epoch_sample_input(self):
    """
    Generator input for the generation of epoch samples.
    """
    pass

  # pylint: disable=too-many-locals,too-many-branches
  def train(self, epochs, metrics_writer):
    try:
      tf.logging.info("Memory usage before training: {}".format(get_memory_usage_string()))
    except: # pylint: disable=bare-except
      tf.logging.warning("Unable to get memory usage, no GPU available?")

    if self._config.online_augmentation:
      tf.logging.fatal("Performing online augmentation!")

    if self._config.train_disc_on_previous_images:
      # logging this here since for these it's implemented (unlike the two-way evaluation)
      tf.logging.fatal("Also training discriminator on previously-generated images")

    previous_generated_images = None
    checkpoint_interval = 25 # always have same interval for easier epoch number -> checkpoint-number conversion
    gradients_interval = epochs // 5 // 25 * 25 # aim for at least 5 gradients in total but have it a multiple of 25
    gradients_interval = 25 if gradients_interval == 0 else min(gradients_interval, 150)
    if self._config.scores_every_epoch:
      scores_interval = 1
    else:
      scores_interval = epochs // 10 // 10 * 10 # aim for at least 10 percentual scores in total but have it a multiple of 10
      scores_interval = 10 if scores_interval == 0 else min(scores_interval, 25)
    tf.logging.info("Intervals: checkpoint {}, scores {}, gradients {}".format(checkpoint_interval, scores_interval, gradients_interval))
    for epoch in range(epochs):
      start = time.time()

      metrics = OneWayMetrics(epoch+1, 4)

      if self._config.extra_disc_step_real or self._config.extra_disc_step_both:
        for batch_number, batch in enumerate(self.data_set):
          inputs, targets = batch
          if self._config.online_augmentation:
            targets = tf.concat([
              targets,
              tf.image.flip_left_right(targets),
              tf.image.flip_up_down(targets),
              tf.image.flip_left_right(tf.image.flip_up_down(targets))
              ], axis=0)
            inputs = tf.random_normal([targets.shape[0], self._config.noise_dimensions])
          with tf.GradientTape() as disc_tape:
            disc_input_real = tf.concat([inputs, targets], axis=-1) if self._config.conditioned_discriminator else targets
            disc_on_real = self._discriminator(disc_input_real, training=True)

            disc_on_generated = None
            if self._config.extra_disc_step_both:
              generated_images = self._generator(inputs, training=False)
              disc_input_generated = tf.concat([inputs, generated_images], axis=-1) if self._config.conditioned_discriminator else generated_images
              disc_on_generated = self._discriminator(disc_input_generated, training=True)

            disc_loss = self._model.disc_loss(disc_on_real, disc_on_generated)
          gradients_of_discriminator = disc_tape.gradient(disc_loss, self._discriminator.variables)
          self._discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self._discriminator.variables))

      for batch_number, batch in enumerate(self.data_set):
        inputs, targets = batch
        if self._config.online_augmentation:
          targets = tf.concat([
            targets,
            tf.image.flip_left_right(targets),
            tf.image.flip_up_down(targets),
            tf.image.flip_left_right(tf.image.flip_up_down(targets))
            ], axis=0)
          inputs = tf.random_normal([targets.shape[0], self._config.noise_dimensions])

        if self._config.train_disc_on_previous_images and previous_generated_images is not None:
          with tf.GradientTape() as disc_tape:
            disc_on_generated = self._discriminator(previous_generated_images, training=True)
            disc_loss = self._model.disc_loss(None, disc_on_generated)
          gradients_of_discriminator = disc_tape.gradient(disc_loss, self._discriminator.variables)
          self._discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self._discriminator.variables))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
          generated_images = self._generator(inputs, training=True)
          previous_generated_images = generated_images

          if self._config.real_image_noise_stdev:
            # stddev = tf.random_uniform((1,), maxval=self._config.real_image_noise_stdev)
            stddev = self._config.real_image_noise_stdev
            targets = tf.minimum(tf.maximum(tf.add(targets, tf.random_normal(targets.shape, mean=0.0, stddev=stddev)), -1), 1)

          disc_input_real = tf.concat([inputs, targets], axis=-1) if self._config.conditioned_discriminator else targets
          disc_input_generated = tf.concat([inputs, generated_images], axis=-1) if self._config.conditioned_discriminator else generated_images

          disc_on_real = self._discriminator(disc_input_real, training=True)
          disc_on_generated = self._discriminator(disc_input_generated, training=True)

          # set up additional args for the generator loss calculation
          gen_loss_args = GeneratorLossArgs(generated_images, inputs, targets=targets)

          gen_losses = self._model.gen_loss(disc_on_generated, gen_loss_args)
          gen_loss = sum(gen_losses.values())
          disc_loss = self._model.disc_loss(disc_on_real, disc_on_generated)

          metrics.add_losses(gen_losses, disc_loss)
          metrics.add_discriminations(logistic(disc_on_real), logistic(disc_on_generated[:disc_on_real.shape[0]]))

        gradients_of_generator = gen_tape.gradient(gen_loss, self._generator.variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self._discriminator.variables)

        self._generator_optimizer.apply_gradients(zip(gradients_of_generator, self._generator.variables))
        self._discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self._discriminator.variables))

        if batch_number == 0:
          # work with the gradients of the first (rather than last) batch since here, the batch is full for sure
          for i, variable in enumerate(self._generator.variables):
            if "batch_normalization" in variable.name or gradients_of_generator[i] is None:
              continue
            tf.contrib.summary.histogram(variable.name.replace(":", "_"), gradients_of_generator[i], "gradients/gen", epoch)
          for i, variable in enumerate(self._discriminator.variables):
            if "batch_normalization" in variable.name or gradients_of_discriminator[i] is None:
              continue
            tf.contrib.summary.histogram(variable.name.replace(":", "_"), gradients_of_discriminator[i], "gradients/disc", epoch)

          if (epoch+1) % gradients_interval == 0 or epoch == epochs - 1:
            generator_gradients = [(variable.name, gradients_of_generator[i].numpy()) \
                for i, variable in enumerate(self._generator.variables) if "batch_normalization" not in variable.name]
            discriminator_gradients = [(variable.name, gradients_of_discriminator[i].numpy()) \
                for i, variable in enumerate(self._discriminator.variables) if "batch_normalization" not in variable.name]
            with open(os.path.join(self._config.gradients_dir, "gradients_at_epoch_{:04d}.pkl".format(epoch+1)), "wb") as fh:
              pickle.dump((generator_gradients, gen_loss.numpy(), discriminator_gradients, disc_loss.numpy()), fh)

      if self.extra_discriminator_data_set:
        assert not self._config.conditioned_discriminator, "doesn't make sense - extra targets means 'too few inputs'"
        assert not self._config.train_disc_on_previous_images and \
            not self._config.real_image_noise_stdev, "not implemented"

        for disc_input in self.extra_discriminator_data_set:
          with tf.GradientTape() as disc_tape:
            disc_on_real = self._discriminator(disc_input, training=True)
            disc_loss = self._model.disc_loss(disc_on_real, []) # no generated samples

          gradients_of_discriminator = disc_tape.gradient(disc_loss, self._discriminator.variables)
          self._discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self._discriminator.variables))

      _ = self.save_epoch_samples(self._generator, self._discriminator, epoch+1, (epoch+1) % 5 == 0)

      tf.contrib.summary.histogram("gen", metrics.gen_loss, "loss", epoch)
      tf.contrib.summary.histogram("disc", metrics.disc_loss, "loss", epoch)

      tf.contrib.summary.histogram("on_real", metrics.disc_on_real, "predictions", epoch)
      tf.contrib.summary.histogram("on_gen", metrics.disc_on_generated, "predictions", epoch)

      if (epoch+1) % checkpoint_interval == 0 or epoch == epochs - 1:
        self._checkpoint.save()
      elif epoch > epochs - 6 and self._final_checkpoint:
        self._final_checkpoint.save()

      if epoch == 0 or epoch == 4 or (epoch+1) % 10 == 0:
        memory_usage = ""
        if epoch == 0 or epoch == 4 or (epoch+1) % 50 == 0:
          try:
            memory_usage = " - memory: " + get_memory_usage_string()
          except: # pylint: disable=bare-except
            memory_usage = " - Unable to get memory usage, no GPU available?"
        time_remaining = (time.time()-self._config.start_time)/(epoch+1)*(epochs-epoch-1)/60
        tf.logging.info("{}/{}: Round time: {:.1f}m - ETA {:%H:%M} ({:.1f}h){}".format(epoch + 1, epochs,
          (time.time()-start)/60, datetime.now() + timedelta(minutes=time_remaining), time_remaining/60,
          memory_usage))

      # pylint: disable=line-too-long
      tf.logging.info("{}/{}: Loss G {:.2f}+-{:.3f}, D {:.2f}+-{:.3f}; D on real {:.3f}+-{:.3f}, on fake {:.3f}+-{:.3f}".format(epoch + 1, epochs,
        np.mean(metrics.gen_loss), np.std(metrics.gen_loss), np.mean(metrics.disc_loss), np.std(metrics.disc_loss),
        np.mean(metrics.disc_on_real), np.std(metrics.disc_on_real), np.mean(metrics.disc_on_generated), np.std(metrics.disc_on_generated)))

      is_near_interval = \
          (epoch+0) % scores_interval == 0 or \
          (epoch+1) % scores_interval == 0 or \
          (epoch+2) % scores_interval == 0
      if self._perceptual_scores and (epoch > 1 or self._config.scores_every_epoch) and (epoch > epochs - 6 or is_near_interval):
        fid, mmd, clustering_high, clustering_low, low_level_fids, combined_fid = self._perceptual_scores.compute_scores_from_generator(
            self._generator, self.data_set.map(lambda x, y: x))
        tf.logging.warning("{}/{}: Computed perceptual scores: FID={:.1f}, MMD={:.3f}, clustering-high={:.3f}, clustering-low={:.3f}".format(
          epoch + 1, epochs, fid, mmd, clustering_high, clustering_low))
        metrics.add_perceptual_scores(fid, mmd, clustering_high, clustering_low, low_level_fids, combined_fid)
        tf.contrib.summary.scalar("fid", fid, "perceptual", epoch)
        tf.contrib.summary.scalar("mmd", mmd, "perceptual", epoch)
        tf.contrib.summary.scalar("clustering_high", clustering_high, "perceptual", epoch)
        tf.contrib.summary.scalar("clustering_low", clustering_low, "perceptual", epoch)
        # not adding low-level FIDs to TB since I'm not using it anyway
        tf.contrib.summary.scalar("combined_fid", combined_fid, "perceptual", epoch)

      if self.test_data_set and (epoch > 1 or self._config.scores_every_epoch) and (epoch > epochs - 6 or is_near_interval):
        disc_on_training = self._discriminate_data_set(self._discriminator, self.data_set)
        disc_on_training_mean = np.mean(disc_on_training)
        disc_on_training_std = np.std(disc_on_training)
        disc_on_test = self._discriminate_data_set(self._discriminator, self.test_data_set)
        disc_on_test_mean = np.mean(disc_on_test)
        disc_on_test_std = np.std(disc_on_test)
        tf.logging.warning("{}/{}: Disc on training: {:.3f}+-{:.3f}, on test: {:.3f}+-{:.3f}, diff: {:.3f}".format(
          epoch + 1, epochs, disc_on_training_mean, disc_on_training_std, disc_on_test_mean, disc_on_test_std,
          disc_on_training_mean - disc_on_test_mean))
        metrics.add_disc_on_training_test(disc_on_training_mean, disc_on_training_std, disc_on_test_mean, disc_on_test_std)
        tf.contrib.summary.scalar("disc_on_training_mean", disc_on_training_mean, "disc_overfitting", epoch)
        tf.contrib.summary.scalar("disc_on_training_std", disc_on_training_std, "disc_overfitting", epoch)
        tf.contrib.summary.scalar("disc_on_test_mean", disc_on_test_mean, "disc_overfitting", epoch)
        tf.contrib.summary.scalar("disc_on_test_std", disc_on_test_std, "disc_overfitting", epoch)

      metrics_writer.writerow(metrics.get_row_data())

  @abstractmethod
  def _plot_epoch_samples(self, generator, discriminator):
    pass

  @abstractmethod
  def _plot_hq_epoch_samples(self, generated_samples, discriminator_probabilities):
    pass
