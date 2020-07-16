#!/usr/bin/env python

import tensorflow as tf
from tqdm import tqdm, trange


class ReconstructionScores:

  def __init__(self, generator, targets, noise_dimensions):
    self._generator = generator
    self._targets = targets
    self._noise_dimensions = noise_dimensions
    tf.logging.info("Computing reconstructions based on MSE")

  def compute_scores(self, extended_results=False):
    input_z = tf.Variable(tf.random_normal([self._targets.shape[0], self._noise_dimensions]))
    adam = tf.train.AdamOptimizer(learning_rate=5e-2)

    if extended_results:
      images = [[] for _ in range(input_z.shape[0])]
      titles = [[] for _ in range(input_z.shape[0])]
      all_similarity_losses = []

    epochs = 100+1
    min_reduction = 1e-4
    epochs_without_reduction = 999
    current_epochs_without_reduction = 0
    epoch_iterator = trange(epochs) if extended_results else range(epochs)
    epoch = 0
    for epoch in epoch_iterator:
      with tf.GradientTape() as tape:
        samples = self._generator(input_z, training=False)

        similarity_losses = tf.convert_to_tensor([tf.losses.mean_squared_error(self._targets[i], samples[i]) for i in range(samples.shape[0])])
        # similarity_losses = 1-tf.image.ssim(self._targets, samples, 2.)
        # similarity_losses = 1-tf.image.ssim_multiscale(self._targets, samples, 2.)
        if extended_results:
          all_similarity_losses.append(similarity_losses)

        gradient = tape.gradient(similarity_losses, input_z)
        adam.apply_gradients([(gradient, input_z)])

      if extended_results:
        for i in range(samples.shape[0]):
          images[i].append(samples[i])
          titles[i].append("Epoch {}: similarity loss {:.5f}".format(epoch, similarity_losses[i]))

      if extended_results and epoch % 5 == 4:
        tqdm.write("{:02d}/{}: Losses: {}".format(epoch+1, epochs, " - ".join(
          ["{:.5f} ({:.5f})".format(similarity_losses[i], all_similarity_losses[-2][i] - similarity_losses[i])
            for i in range(similarity_losses.shape[0])])))

      current_mean_loss = tf.reduce_mean(tf.nn.relu(similarity_losses))
      if epoch > 1:
        if previous_mean_loss - current_mean_loss < min_reduction:
          current_epochs_without_reduction += 1
          if current_epochs_without_reduction >= epochs_without_reduction:
            if extended_results:
              tqdm.write("{:02d}/{}: No significant reduction in losses for {} consecutive epochs, stopping".format(
                epoch+1, epochs, epochs_without_reduction))
            break
          elif extended_results:
            tqdm.write("{:02d}/{}: No significant reduction in losses: {:.6f}".format(epoch+1, epochs, previous_mean_loss-current_mean_loss))
        else:
          current_epochs_without_reduction = 0
      previous_mean_loss = current_mean_loss

    last_losses = similarity_losses.numpy()
    if extended_results:
      tf.logging.warning("Finished after {} epochs with loss {:.5f}+-{:.5f}".format(
        epoch+1, last_losses.mean(), last_losses.std()))
    return (images, titles, all_similarity_losses) if extended_results else (last_losses.mean(), last_losses.std())
