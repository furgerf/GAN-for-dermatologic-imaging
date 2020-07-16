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
from perceptual_scores import PerceptualScores
from two_way_metrics import TwoWayMetrics
from utils import get_memory_usage_string, logistic


class TwoWayEvaluation(Evaluation):
  def __init__(self, model, config):
    assert not config.has_noise_input, "we don't want to translate back to noise"
    self._first_generator = None
    self._first_generator_optimizer = None
    self._first_discriminator = None
    self._first_discriminator_optimizer = None
    self._second_generator = None
    self._second_generator_optimizer = None
    self._second_discriminator = None
    self._second_discriminator_optimizer = None
    self._checkpoint = None
    self._final_checkpoint = None
    self._perceptual_scores = PerceptualScores(config) if config.target_type == "image" else None
    self._reverse_perceptual_scores = PerceptualScores(config) if config.input_type == "image" else None
    super(TwoWayEvaluation, self).__init__(model, config)

  def set_up_model(self):
    tf.logging.info("Setting up models with learing rate {} for G, {} for D".format(
      self._model.gen_learning, self._model.disc_learning))
    self._first_generator = self._model.get_generator()
    self._first_discriminator = self._model.get_discriminator()
    has_colored_target = self._config.has_colored_target
    self._config.has_colored_target = self._config.has_colored_input
    self._second_generator = self._model.get_generator()
    self._config.has_colored_target = has_colored_target
    self._second_discriminator = self._model.get_discriminator()
    # defun gives 10 secs/epoch performance boost
    self._first_generator.call = tf.contrib.eager.defun(self._first_generator.call)
    self._first_discriminator.call = tf.contrib.eager.defun(self._first_discriminator.call)
    self._second_generator.call = tf.contrib.eager.defun(self._second_generator.call)
    self._second_discriminator.call = tf.contrib.eager.defun(self._second_discriminator.call)

    self._first_generator_optimizer = tf.train.AdamOptimizer(self._model.gen_learning)
    self._first_discriminator_optimizer = tf.train.AdamOptimizer(self._model.disc_learning)
    self._second_generator_optimizer = tf.train.AdamOptimizer(self._model.gen_learning)
    self._second_discriminator_optimizer = tf.train.AdamOptimizer(self._model.disc_learning)
    checkpoint = tf.train.Checkpoint(
        first_generator_optimizer=self._first_generator_optimizer,
        first_discriminator_optimizer=self._first_discriminator_optimizer,
        first_generator=self._first_generator,
        first_discriminator=self._first_discriminator,
        second_generator_optimizer=self._second_generator_optimizer,
        second_discriminator_optimizer=self._second_discriminator_optimizer,
        second_generator=self._second_generator,
        second_discriminator=self._second_discriminator)
    self._checkpoint = tf.contrib.checkpoint.CheckpointManager(checkpoint, self._config.checkpoint_dir,
        max_to_keep=None if self._config.keep_all_checkpoints else 5)
    if self._config.keep_final_checkpoints:
      final_checkpoint = tf.train.Checkpoint(
          first_generator_optimizer=self._first_generator_optimizer,
          first_discriminator_optimizer=self._first_discriminator_optimizer,
          first_generator=self._first_generator,
          first_discriminator=self._first_discriminator,
          second_generator_optimizer=self._second_generator_optimizer,
          second_discriminator_optimizer=self._second_discriminator_optimizer,
          second_generator=self._second_generator,
          second_discriminator=self._second_discriminator)
      self._final_checkpoint = tf.contrib.checkpoint.CheckpointManager(final_checkpoint, self._config.final_checkpoint_dir,
          max_to_keep=None if self._config.keep_all_checkpoints else 5)

    try:
      self._model.print_model_summary(self._first_generator, self._second_discriminator, self.epoch_sample_input)
    except Exception as ex:
      tf.logging.warning("Unable to print model summary ({}: {})".format(ex.__class__.__name__, ex))

    if self._perceptual_scores:
      self._perceptual_scores.initialize()
    if self._reverse_perceptual_scores:
      self._reverse_perceptual_scores.initialize(self._config.data_dir)

  @property
  @abstractmethod
  def data_set(self):
    """
    The data set to train on. Each batch should consist of a tuple of generator inputs in the first
    and in the second domain.
    """
    pass

  @property
  @abstractmethod
  def extra_discriminator_data_set(self):
    """
    The data set of additional real samples for the SECOND discriminator to train on.
    Only makes sense for non-conditioned discriminators.
    """
    pass

  @property
  @abstractmethod
  def test_data_set(self):
    """
    The data set to train on. Each batch should consist of a tuple of generator inputs in the first
    and in the second domain - same as the main data set.
    """
    pass

  @property
  @abstractmethod
  def epoch_sample_input(self):
    """
    Generator input for the generation of epoch samples.
    """
    pass

  class TrainingResult:
    def __init__(self, gen_loss, gen_losses, disc_loss, disc_on_real, disc_on_generated,
        gen_gradients, disc_gradients):
      # pylint: disable=too-many-arguments
      self.gen_loss = gen_loss
      self.gen_losses = gen_losses
      self.disc_loss = disc_loss

      self.disc_on_real = disc_on_real
      self.disc_on_generated = disc_on_generated

      self.gen_gradients = gen_gradients
      self.disc_gradients = disc_gradients

  def train_generator_discriminator_pair(self, generator, discriminator, reverse_generator, batch_input, batch_target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      # FORWARD DIRECTION
      generated_images = generator(batch_input, training=True)

      if batch_input.shape[-1] == 4:
        assert batch_target.shape[-1] == 4
        disc_input_real = tf.concat([batch_input, batch_target[:, :, :, :3]], axis=-1) \
            if self._config.conditioned_discriminator else batch_target[:, :, :, :3]
        disc_input_generated = tf.concat([batch_input, generated_images], axis=-1) \
            if self._config.conditioned_discriminator else generated_images
      else:
        assert batch_target.shape[-1] in [1, 3] and batch_input.shape[-1] in [1, 3]
        disc_input_real = tf.concat([batch_input, batch_target], axis=-1) \
            if self._config.conditioned_discriminator else batch_target
        disc_input_generated = tf.concat([batch_input, generated_images], axis=-1) \
            if self._config.conditioned_discriminator else generated_images

      disc_on_real = discriminator(disc_input_real, training=True)
      disc_on_generated = discriminator(disc_input_generated, training=True)

      # BACKWARD DIRECTION - not training or discriminating reconstructed image
      # the input for the reconstruction may need to be augmented with the segmentation
      reconstruction_input = tf.concat([generated_images, batch_input[:, :, :, 3:]], axis=-1) if batch_input.shape[-1] > 3 else generated_images
      reconstructed_images = reverse_generator(reconstruction_input, training=False)

      if self._config.loss_identity:
        targets = batch_target
        identity_images = generator(batch_target, training=True)
      else:
        targets = None
        identity_images = None

      gen_losses = self._model.gen_loss(disc_on_generated, GeneratorLossArgs(generated_images, batch_input,
        targets=targets, reconstructed_images=reconstructed_images, identity_images=identity_images))
      gen_loss = sum(gen_losses.values())
      disc_loss = self._model.disc_loss(disc_on_real, disc_on_generated)

    # compute gradients
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)

    return TwoWayEvaluation.TrainingResult(gen_loss, gen_losses, disc_loss, disc_on_real, disc_on_generated,
        gradients_of_generator, gradients_of_discriminator)

  def train_discriminator(self, generator, discriminator, batch_input, batch_target):
    with tf.GradientTape() as disc_tape:
      if generator:
        generated_images = generator(batch_input, training=True)

      if batch_input.shape[-1] == 4:
        assert batch_target.shape[-1] == 4
        disc_input_real = tf.concat([batch_input, batch_target[:, :, :, :3]], axis=-1) \
            if self._config.conditioned_discriminator else batch_target[:, :, :, :3]
        if generator:
          disc_input_generated = tf.concat([batch_input, generated_images], axis=-1) \
              if self._config.conditioned_discriminator else generated_images
      else:
        assert batch_target.shape[-1] == 3 and batch_input.shape[-1] == 3
        disc_input_real = tf.concat([batch_input, batch_target], axis=-1) \
            if self._config.conditioned_discriminator else batch_target
        if generator:
          disc_input_generated = tf.concat([batch_input, generated_images], axis=-1) \
              if self._config.conditioned_discriminator else generated_images

      disc_on_real = discriminator(disc_input_real, training=True)
      if generator:
        disc_on_generated = discriminator(disc_input_generated, training=True)
      else:
        disc_on_generated = None

      disc_loss = self._model.disc_loss(disc_on_real, disc_on_generated)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)

    return TwoWayEvaluation.TrainingResult(None, None, disc_loss, disc_on_real, disc_on_generated,
        None, gradients_of_discriminator)

  def train(self, epochs, metrics_writer):
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    try:
      tf.logging.info("Memory usage before training: {}".format(get_memory_usage_string()))
    except: # pylint: disable=bare-except
      tf.logging.warning("Unable to get memory usage, no GPU available?")

    assert not self._config.train_disc_on_previous_images \
        and not self._config.real_image_noise_stdev, "not implemented"

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

      metrics = TwoWayMetrics(epoch+1, 4)

      # NOTE, KEEP IN MIND: "first"/"second" refers to the input domain
      # ie "first generator" is the generator that receives input from the first domain (and generates for the second)

      if self._config.extra_disc_step_real or self._config.extra_disc_step_both:
        for batch_number, batch in enumerate(self.data_set):
          batch_first_domain, batch_second_domain = batch

          forward_result = self.train_discriminator(self._first_generator if self._config.extra_disc_step_both else None,
              self._second_discriminator, batch_first_domain, batch_second_domain)
          backward_result = self.train_discriminator(self._second_generator if self._config.extra_disc_step_both else None,
              self._first_discriminator, batch_second_domain, batch_first_domain)

          self._second_discriminator_optimizer.apply_gradients(zip(forward_result.disc_gradients, self._second_discriminator.variables))
          self._first_discriminator_optimizer.apply_gradients(zip(backward_result.disc_gradients, self._first_discriminator.variables))

      for batch_number, batch in enumerate(self.data_set):
        batch_first_domain, batch_second_domain = batch

        # evaluate models
        forward_result = self.train_generator_discriminator_pair(self._first_generator, self._second_discriminator,
            self._second_generator, batch_first_domain, batch_second_domain)
        backward_result = self.train_generator_discriminator_pair(self._second_generator, self._first_discriminator,
            self._first_generator, batch_second_domain, batch_first_domain)

        # store results
        metrics.add_losses((forward_result.gen_losses, backward_result.gen_losses), (backward_result.disc_loss, forward_result.disc_loss))
        metrics.add_discriminations((logistic(backward_result.disc_on_real), logistic(forward_result.disc_on_real)),
            (logistic(backward_result.disc_on_generated), logistic(forward_result.disc_on_generated)))

        # train
        self._first_generator_optimizer.apply_gradients(zip(forward_result.gen_gradients, self._first_generator.variables))
        self._second_discriminator_optimizer.apply_gradients(zip(forward_result.disc_gradients, self._second_discriminator.variables))
        self._second_generator_optimizer.apply_gradients(zip(backward_result.gen_gradients, self._second_generator.variables))
        self._first_discriminator_optimizer.apply_gradients(zip(backward_result.disc_gradients, self._first_discriminator.variables))

        if batch_number == 0:
          # work with the gradients of the first (rather than last) batch since here, the batch is full for sure
          for i, variable in enumerate(self._first_generator.variables):
            if "batch_normalization" in variable.name or forward_result.gen_gradients[i] is None:
              continue
            tf.contrib.summary.histogram(variable.name.replace(":", "_"), forward_result.gen_gradients[i], "gradients/first_gen", epoch)
          for i, variable in enumerate(self._first_discriminator.variables):
            if "batch_normalization" in variable.name or backward_result.disc_gradients[i] is None:
              continue
            tf.contrib.summary.histogram(variable.name.replace(":", "_"), backward_result.disc_gradients[i], "gradients/first_disc", epoch)
          for i, variable in enumerate(self._second_generator.variables):
            if "batch_normalization" in variable.name or backward_result.gen_gradients[i] is None:
              continue
            tf.contrib.summary.histogram(variable.name.replace(":", "_"), backward_result.gen_gradients[i], "gradients/second_gen", epoch)
          for i, variable in enumerate(self._second_discriminator.variables):
            if "batch_normalization" in variable.name or forward_result.disc_gradients[i] is None:
              continue
            tf.contrib.summary.histogram(variable.name.replace(":", "_"), forward_result.disc_gradients[i], "gradients/second_disc", epoch)

          if (epoch+1) % gradients_interval == 0 or epoch == epochs - 1:
            first_generator_gradients = [(variable.name, forward_result.gen_gradients[i].numpy()) \
                for i, variable in enumerate(self._first_generator.variables) if "batch_normalization" not in variable.name]
            first_discriminator_gradients = [(variable.name, backward_result.disc_gradients[i].numpy()) \
                for i, variable in enumerate(self._first_discriminator.variables) if "batch_normalization" not in variable.name]
            second_generator_gradients = [(variable.name, backward_result.gen_gradients[i].numpy()) \
                for i, variable in enumerate(self._second_generator.variables) if "batch_normalization" not in variable.name]
            second_discriminator_gradients = [(variable.name, forward_result.disc_gradients[i].numpy()) \
                for i, variable in enumerate(self._second_discriminator.variables) if "batch_normalization" not in variable.name]
            with open(os.path.join(self._config.gradients_dir, "gradients_at_epoch_{:04d}.pkl".format(epoch+1)), "wb") as fh:
              pickle.dump((first_generator_gradients, forward_result.gen_loss, second_generator_gradients, backward_result.gen_loss,
                first_discriminator_gradients, backward_result.disc_loss, second_discriminator_gradients, forward_result.disc_loss), fh)

      if self.extra_discriminator_data_set:
        assert not self._config.conditioned_discriminator and \
            not self._config.train_disc_on_previous_images and \
            not self._config.real_image_noise_stdev, "not implemented"

        for disc_input in self.extra_discriminator_data_set:
          with tf.GradientTape() as disc_tape:
            disc_on_real = self._second_discriminator(disc_input, training=True)
            disc_loss = self._model.disc_loss(disc_on_real, []) # no generated samples

          gradients_of_discriminator = disc_tape.gradient(disc_loss, self._second_discriminator.variables)
          self._second_discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self._second_discriminator.variables))

      _ = self.save_epoch_samples((self._first_generator, self._second_generator),
          (self._first_discriminator, self._second_discriminator), epoch+1, (epoch+1) % 5 == 0)

      tf.contrib.summary.histogram("first_gen", metrics.first_gen_loss, "loss", epoch)
      tf.contrib.summary.histogram("first_disc", metrics.first_disc_loss, "loss", epoch)
      tf.contrib.summary.histogram("second_gen", metrics.second_gen_loss, "loss", epoch)
      tf.contrib.summary.histogram("second_disc", metrics.second_disc_loss, "loss", epoch)

      tf.contrib.summary.histogram("first_on_real", metrics.first_disc_on_real, "predictions", epoch)
      tf.contrib.summary.histogram("first_on_gen", metrics.first_disc_on_generated, "predictions", epoch)
      tf.contrib.summary.histogram("second_on_real", metrics.second_disc_on_real, "predictions", epoch)
      tf.contrib.summary.histogram("second_on_gen", metrics.second_disc_on_generated, "predictions", epoch)

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

      tf.logging.info("{}/{} FWD: Loss G {:.2f}+-{:.3f}, D {:.2f}+-{:.3f}; D on real {:.3f}+-{:.3f}, on fake {:.3f}+-{:.3f}".format(
        epoch + 1, epochs,
        np.mean(metrics.first_gen_loss), np.std(metrics.first_gen_loss),
        np.mean(metrics.second_disc_loss), np.std(metrics.second_disc_loss),
        np.mean(metrics.second_disc_on_real), np.std(metrics.second_disc_on_real),
        np.mean(metrics.second_disc_on_generated), np.std(metrics.second_disc_on_generated)))
      tf.logging.info("{}/{} BWD: Loss G {:.2f}+-{:.3f}, D {:.2f}+-{:.3f}; D on real {:.3f}+-{:.3f}, on fake {:.3f}+-{:.3f}".format(
        epoch + 1, epochs,
        np.mean(metrics.second_gen_loss), np.std(metrics.second_gen_loss),
        np.mean(metrics.first_disc_loss), np.std(metrics.first_disc_loss),
        np.mean(metrics.first_disc_on_real), np.std(metrics.first_disc_on_real),
        np.mean(metrics.first_disc_on_generated), np.std(metrics.first_disc_on_generated)))

      is_near_interval = \
          (epoch+0) % scores_interval == 0 or \
          (epoch+1) % scores_interval == 0 or \
          (epoch+2) % scores_interval == 0
      if (epoch > 1 or self._config.scores_every_epoch) and (epoch > epochs - 6 or is_near_interval):
        first_fid = first_mmd = first_clustering_high = first_clustering_low = first_combined_fid = tf.convert_to_tensor(np.nan)
        first_low_level_fids = [tf.convert_to_tensor(np.nan)] * metrics.n_low_level_fids
        second_fid = second_mmd = second_clustering_high = second_clustering_low = second_combined_fid = tf.convert_to_tensor(np.nan)
        second_low_level_fids = [tf.convert_to_tensor(np.nan)] * metrics.n_low_level_fids
        if self._perceptual_scores:
          first_fid, first_mmd, first_clustering_high, first_clustering_low, first_low_level_fids, first_combined_fid = \
              self._perceptual_scores.compute_scores_from_generator(self._first_generator, self.data_set.map(lambda x, y: x))
          tf.logging.warning("{}/{}: FWD: Computed perceptual scores: FID={:.1f}, MMD={:.3f}, clustering-high={:.3f}, clustering-low={:.3f}".format(
            epoch + 1, epochs, first_fid, first_mmd, first_clustering_high, first_clustering_low))
        if self._reverse_perceptual_scores:
          second_fid, second_mmd, second_clustering_high, second_clustering_low, second_low_level_fids, second_combined_fid = \
              self._reverse_perceptual_scores.compute_scores_from_generator(self._second_generator, self.data_set.map(lambda x, y: y))
          tf.logging.warning("{}/{}: BWD: Computed perceptual scores: FID={:.1f}, MMD={:.3f}, clustering-high={:.3f}, clustering-low={:.3f}".format(
            epoch + 1, epochs, second_fid, second_mmd, second_clustering_high, second_clustering_low))
        metrics.add_perceptual_scores((first_fid, second_fid), (first_mmd, second_mmd), (first_clustering_high, second_clustering_high),
            (first_clustering_low, second_clustering_low), (first_low_level_fids, second_low_level_fids), (first_combined_fid, second_combined_fid))
        tf.contrib.summary.scalar("first_fid", first_fid, "perceptual", epoch)
        tf.contrib.summary.scalar("first_mmd", first_mmd, "perceptual", epoch)
        tf.contrib.summary.scalar("first_clustering_high", first_clustering_high, "perceptual", epoch)
        tf.contrib.summary.scalar("first_clustering_low", first_clustering_low, "perceptual", epoch)
        # not adding low-level FIDs to TB since I'm not using it anyway
        tf.contrib.summary.scalar("first_combined_fid", first_combined_fid, "perceptual", epoch)
        tf.contrib.summary.scalar("second_fid", second_fid, "perceptual", epoch)
        tf.contrib.summary.scalar("second_mmd", second_mmd, "perceptual", epoch)
        tf.contrib.summary.scalar("second_clustering_high", second_clustering_high, "perceptual", epoch)
        tf.contrib.summary.scalar("second_clustering_low", second_clustering_low, "perceptual", epoch)
        # not adding low-level FIDs to TB since I'm not using it anyway
        tf.contrib.summary.scalar("second_combined_fid", second_combined_fid, "perceptual", epoch)

      if self.test_data_set and (epoch > 1 or self._config.scores_every_epoch) and (epoch > epochs - 6 or is_near_interval):
        first_disc_on_training = self._discriminate_data_set(self._first_discriminator, self.data_set.map(lambda x, y: (y, x[:, :, :, :3])))
        first_disc_on_training_mean = np.mean(first_disc_on_training)
        first_disc_on_training_std = np.std(first_disc_on_training)
        first_disc_on_test = self._discriminate_data_set(self._first_discriminator, self.test_data_set.map(lambda x, y: (y, x[:, :, :, :3])))
        first_disc_on_test_mean = np.mean(first_disc_on_test)
        first_disc_on_test_std = np.std(first_disc_on_test)
        second_disc_on_training = self._discriminate_data_set(self._second_discriminator, self.data_set.map(lambda x, y: (x, y[:, :, :, :3])))
        second_disc_on_training_mean = np.mean(second_disc_on_training)
        second_disc_on_training_std = np.std(second_disc_on_training)
        second_disc_on_test = self._discriminate_data_set(self._second_discriminator, self.test_data_set.map(lambda x, y: (x, y[:, :, :, :3])))
        second_disc_on_test_mean = np.mean(second_disc_on_test)
        second_disc_on_test_std = np.std(second_disc_on_test)
        tf.logging.warning("{}/{}: First disc on training: {:.3f}+-{:.3f}, on test: {:.3f}+-{:.3f}, diff: {:.3f}".format(
          epoch + 1, epochs, first_disc_on_training_mean, first_disc_on_training_std, first_disc_on_test_mean, first_disc_on_test_std,
          first_disc_on_training_mean - first_disc_on_test_mean))
        tf.logging.warning("{}/{}: Second disc on training: {:.3f}+-{:.3f}, on test: {:.3f}+-{:.3f}, diff: {:.3f}".format(
          epoch + 1, epochs, second_disc_on_training_mean, second_disc_on_training_std, second_disc_on_test_mean, second_disc_on_test_std,
          second_disc_on_training_mean - second_disc_on_test_mean))
        metrics.add_disc_on_training_test((first_disc_on_training_mean, second_disc_on_training_mean), (first_disc_on_training_std,
          second_disc_on_training_std), (first_disc_on_test_mean, second_disc_on_test_mean), (first_disc_on_test_std, second_disc_on_test_std))
        tf.contrib.summary.scalar("first_disc_on_training_mean", first_disc_on_training_mean, "disc_overfitting", epoch)
        tf.contrib.summary.scalar("first_disc_on_training_std", first_disc_on_training_std, "disc_overfitting", epoch)
        tf.contrib.summary.scalar("first_disc_on_test_mean", first_disc_on_test_mean, "disc_overfitting", epoch)
        tf.contrib.summary.scalar("first_disc_on_test_std", first_disc_on_test_std, "disc_overfitting", epoch)
        tf.contrib.summary.scalar("second_disc_on_training_mean", second_disc_on_training_mean, "disc_overfitting", epoch)
        tf.contrib.summary.scalar("second_disc_on_training_std", second_disc_on_training_std, "disc_overfitting", epoch)
        tf.contrib.summary.scalar("second_disc_on_test_mean", second_disc_on_test_mean, "disc_overfitting", epoch)
        tf.contrib.summary.scalar("second_disc_on_test_std", second_disc_on_test_std, "disc_overfitting", epoch)

      metrics_writer.writerow(metrics.get_row_data())

  @abstractmethod
  def _plot_epoch_samples(self, generator, discriminator):
    pass

  @abstractmethod
  def _plot_hq_epoch_samples(self, generated_samples, discriminator_probabilities):
    pass
