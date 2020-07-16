#!/usr/bin/env python

import csv
import os
import time
import traceback
from argparse import ArgumentParser
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.decomposition import PCA

from model import Model
from utils import configure_logging, logistic

# pylint: disable=invalid-name,arguments-differ,too-many-arguments

# TODO: instead of discriminating a single point, discriminate a batch of X points

np.random.seed(42)
configure_logging()

successful_learning = 0
bad_generator = 1
bad_discriminator = 2
bad_both = 3
overfitting_generator = 4
overfitting_discriminator = 5
overfitting_both = 6

experiment = successful_learning

tf.logging.info("Running experiment {}".format(experiment))

dims = 10
gaussians = 5

generator_layer_count = [8, 4, 8, 4, 32, 8, 32][experiment]
generator_neurons = [256, 64, 256, 64, 1024, 256, 1024][experiment]
generator_use_bias = False
generator_use_batchnorm = True
generator_activation = tf.nn.relu

discriminator_layer_count = [4, 4, 2, 2, 4, 16, 16][experiment]
discriminator_neurons = [64, 64, 8, 8, 64, 256, 256][experiment]
discriminator_use_bias = False
discriminator_use_batchnorm = False
discriminator_use_dropout = False
discriminator_activation = tf.nn.leaky_relu

means = [tuple([np.random.uniform(-20, 20) for _ in range(dims)]) for _ in range(gaussians)]
stds = [tuple([np.random.uniform(0, 5) for _ in range(dims)]) for _ in range(gaussians)]
assert len(means) == len(stds)

def draw_real_samples(sample_count):
  counts = [sample_count // gaussians] * gaussians
  counts[-1] -= sum(counts) - sample_count
  assert sum(counts) == sample_count

  samples = []
  for i in range(gaussians):
    samples.append(np.array([np.random.normal(means[i][j], stds[i][j], counts[i]) for j in range(dims)], dtype=np.float32))
  return np.random.permutation(np.concatenate(samples, axis=-1).T)

pca = PCA(n_components=2)
transformed = pca.fit_transform(draw_real_samples(int(1e6)))
x_lim = (min(transformed[:, 0]), max(transformed[:, 0]))
x_range = x_lim[1]-x_lim[0]
x_lim = (x_lim[0]-x_range/4, x_lim[1]+x_range/4)
y_lim = (min(transformed[:, 1]), max(transformed[:, 1]))
y_range = y_lim[1]-y_lim[0]
y_lim = (y_lim[0]-y_range/4, y_lim[1]+y_range/4)

tf.logging.info("Amount of explained variance by 2 components for {} dims: {:.1f}% ({})".format(dims,
  100*np.sum(pca.explained_variance_ratio_), pca.explained_variance_ratio_))

class NoiseToGaussianModel(Model):
  class Generator(tf.keras.Model):
    def __init__(self, args):
      super(NoiseToGaussianModel.Generator, self).__init__()

      tf.logging.error("G: {} dense ({} neurons)".format(generator_layer_count, generator_neurons))

      self.dense_layers = []
      for _ in range(generator_layer_count):
        self.dense_layers.append(tf.keras.layers.Dense(generator_neurons, use_bias=generator_use_bias))
        if generator_use_batchnorm:
          self.dense_layers.append(tf.keras.layers.BatchNormalization())
      self.final_layer = tf.keras.layers.Dense(dims, use_bias=generator_use_bias)

    def call(self, x, training=True):
      for i in range(0, len(self.dense_layers), 2 if generator_use_batchnorm else 1):
        x = self.dense_layers[i](x) # dense
        if generator_use_batchnorm:
          x = self.dense_layers[i+1](x, training=training) # batchnorm
        x = generator_activation(x)
      return self.final_layer(x)

  class Discriminator(tf.keras.Model):
    def __init__(self):
      super(NoiseToGaussianModel.Discriminator, self).__init__()

      tf.logging.error("D: {} dense ({} neurons)".format(discriminator_layer_count, discriminator_neurons))

      self.dense_layers = []
      for _ in range(discriminator_layer_count):
        self.dense_layers.append(tf.keras.layers.Dense(discriminator_neurons, use_bias=discriminator_use_bias))
        if discriminator_use_batchnorm:
          self.dense_layers.append(tf.keras.layers.BatchNormalization())
      self.final_layer = tf.keras.layers.Dense(1, use_bias=discriminator_use_bias)
      if discriminator_use_dropout:
        self.dropout = tf.keras.layers.Dropout(0.1)

    def call(self, x, training=True):
      for i in range(0, len(self.dense_layers), 2 if discriminator_use_batchnorm else 1):
        x = self.dense_layers[i](x) # dense
        if discriminator_use_batchnorm:
          x = self.dense_layers[i+1](x, training=training) # batchnorm
        x = discriminator_activation(x)
        if discriminator_use_dropout:
          x = self.dropout(x, training=training)
      return self.final_layer(x)

def plot_samples(samples, axis, real=True, old_generated=False):
  data = samples.numpy()
  data = pca.transform(data)
  axis.scatter(data[:, 0], data[:, 1], s=1, c="k" if real else "orange" if old_generated else "r",
      alpha=0.1 if old_generated else 0.2)

def plot_probability_map(discriminator, axis, tile_size=1.):
  # pcolormesh seems to skip the last row/col so add a dummy column; also add tolerance before casting to int
  x_lim_extended = (int(x_lim[0]-1), int(x_lim[1]+2))
  y_lim_extended = (int(y_lim[0]-1), int(y_lim[1]+2))
  my_x_range = int((x_lim_extended[1]-x_lim_extended[0]+1) / tile_size)
  my_y_range = int((y_lim_extended[1]-y_lim_extended[0]+1) / tile_size)
  xx = np.tile(np.arange(x_lim_extended[0], x_lim_extended[1]+1, tile_size), my_y_range).reshape(my_y_range, -1)
  yy = np.tile(np.arange(y_lim_extended[1], y_lim_extended[0]-1, -tile_size), my_x_range).reshape(my_x_range, -1).T
  inputs = np.zeros(shape=(*xx.shape, dims), dtype=np.float32)
  for index, _ in np.ndenumerate(xx):
    high_dimensional_point = pca.inverse_transform((xx[index], yy[index]-tile_size))
    inputs[index[0], index[1], :] = high_dimensional_point

  probability_map = tf.reshape(logistic(discriminator(tf.convert_to_tensor(inputs.reshape(-1, dims)), training=False)), shape=xx.shape)

  _ = axis.pcolormesh(xx-tile_size/2.0, yy-tile_size/2.0, probability_map, cmap="coolwarm", alpha=0.5, vmin=0, vmax=1)

def plot_marginal_distribution(real, generated, dim, axis):
  assert real.shape == generated.shape, "{} != {}".format(real.shape, generated.shape)
  real_values = real[:, dim]
  sns.kdeplot(real_values, vertical=dim, color="k", ax=axis)
  generated_values = generated[:, dim]
  sns.kdeplot(generated_values, vertical=dim, color="r", ax=axis)


def store_epoch_results(real, generated, generator, discriminator, args, epoch, prediction_difference, generator_input_offset=0):
  max_samples_of_constant_distribution = min(int(1e4), real.shape[0])
  assert max_samples_of_constant_distribution >= real.shape[0]
  real_to_plot = real[:max_samples_of_constant_distribution]
  newly_generated = generator(draw_inputs(max_samples_of_constant_distribution, args.noise_dimensions, generator_input_offset), training=False)

  _, axes = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(16, 16),
      gridspec_kw={"height_ratios": [1, 3], "width_ratios": [3, 1]})
  plt.suptitle("{}: Epoch {} (D prediction diff: {:.5f})".format(args.eid, epoch, prediction_difference), fontsize=16)
  ((ax_x_dist, _), (ax_main, ax_y_dist)) = axes
  _.set_visible(False)
  ax_main.set_xlim(*x_lim)
  ax_main.set_ylim(*y_lim)
  ax_x_dist.set_yticks([])
  ax_y_dist.set_xticks([])
  plt.subplots_adjust(hspace=0, wspace=0)
  plt.setp(axes.flat, adjustable="box")

  plot_probability_map(discriminator, ax_main)
  plot_samples(real_to_plot, ax_main)
  plot_samples(generated, ax_main, False, True)
  plot_samples(newly_generated, ax_main, False)

  real_for_marginal = pca.transform(real[:newly_generated.shape[0]])
  generated_for_marginal = pca.transform(newly_generated)
  plot_marginal_distribution(real_for_marginal, generated_for_marginal, 0, ax_x_dist)
  plot_marginal_distribution(real_for_marginal, generated_for_marginal, 1, ax_y_dist)

  figure_file = os.path.join(args.figures_dir, "image_at_epoch_{:04d}.png").format(epoch) if generator_input_offset == 0 else \
      os.path.join(args.figures_dir, "image_at_epoch_{:04d}-offset-{}.png").format(epoch, generator_input_offset)
  plt.savefig(figure_file)
  plt.close()

def draw_inputs(input_count, dimensions, offset=0):
  return tf.random_normal([input_count, dimensions]) + offset


def parse_arguments():
  parser = ArgumentParser()

  parser.add_argument("--eid", type=str, required=True,
      help="ID of the evaluation")
  parser.add_argument("--epochs", type=int, default=5000,
      help="The number of epochs to train for")
  parser.add_argument("--batch-size", type=int, default=32,
      help="The number of samples in one batch")
  parser.add_argument("--batches", type=int, default=128,
      help="The number of batches per epoch")
  parser.add_argument("--noise-dimensions", type=int, default=1,
      help="The dimensionality of the noise input")

  return parser.parse_args()

def set_up_model(model, args):
  tf.logging.info("Setting up models with learing rate {} for G, {} for D".format(
    model.gen_learning, model.disc_learning))
  generator = model.get_generator()
  discriminator = model.get_discriminator()
  # defun gives 10 secs/epoch performance boost
  generator.call = tf.contrib.eager.defun(generator.call)
  discriminator.call = tf.contrib.eager.defun(discriminator.call)

  generator_optimizer = tf.train.AdamOptimizer(model.gen_learning)
  discriminator_optimizer = tf.train.AdamOptimizer(model.disc_learning)
  checkpoint = tf.train.Checkpoint(
      generator_optimizer=generator_optimizer,
      discriminator_optimizer=discriminator_optimizer,
      generator=generator,
      discriminator=discriminator)
  checkpoint = tf.contrib.checkpoint.CheckpointManager(checkpoint, args.checkpoint_dir, max_to_keep=5)
  model.print_model_summary(generator, discriminator, draw_inputs(1, args.noise_dimensions))
  return generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint

def train(model, generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint, metrics_writer, args):
  for epoch in range(args.epochs):
    start = time.time()

    losses = []
    discriminations = [list(), list()]
    all_targets = []
    all_generated = []

    for _ in range(args.batches):
      inputs = draw_inputs(args.batch_size, args.noise_dimensions)
      # if (epoch+1) % 10 == 0:
      #   inputs += 2
      targets = tf.convert_to_tensor(draw_real_samples(args.batch_size))
      # additional_targets = tf.convert_to_tensor(draw_real_samples(args.batch_size*2))
      all_targets.extend(targets)

      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated = generator(inputs, training=True)
        all_generated.extend(generated)

        disc_on_real = discriminator(targets, training=True)
        # disc_on_additional_real = discriminator(additional_targets, training=True)
        disc_on_generated = discriminator(generated, training=True)

        gen_losses = model.gen_loss(disc_on_generated, None)
        gen_loss = sum(gen_losses.values())
        # disc_on_combined_real = tf.concat([disc_on_real, disc_on_additional_real], axis=0)
        # disc_loss = model.disc_loss(disc_on_combined_real, disc_on_generated)
        disc_loss = model.disc_loss(disc_on_real, disc_on_generated)
        losses.append([gen_loss, disc_loss])
        discriminations[0].extend(logistic(disc_on_real))
        discriminations[1].extend(logistic(disc_on_generated))

      gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
      gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)

      generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
      discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.variables))

    if (epoch+1) % 10 == 0:
      checkpoint.save()

    time_remaining = (time.time()-args.start_time)/(epoch+1)*(args.epochs-epoch-1)/60

    # get both types of metrics into numpy arrays with the metrics in the first dimension
    losses = np.array(losses).T
    discriminations = np.array(discriminations)

    if (epoch+1) % 1 == 0:
      store_epoch_results(tf.convert_to_tensor(all_targets), tf.convert_to_tensor(all_generated), generator, discriminator, args, epoch+1,
          discriminations[0].mean() - discriminations[1].mean())
      store_epoch_results(tf.convert_to_tensor(all_targets), tf.convert_to_tensor(all_generated), generator, discriminator, args, epoch+1,
          discriminations[0].mean() - discriminations[1].mean(), 2)

    tf.logging.info("{}/{}: Loss G {:.2f}, D {:.2f}; D on real {:.3f}, on fake {:.3f} ({:.3f}); {:.1f}s; ETA {:%H:%M} ({:.1f}h)".format(
      epoch + 1, args.epochs,
      losses[0].mean(), losses[1].mean(),
      discriminations[0].mean(), discriminations[1].mean(),
      discriminations[0].mean() - discriminations[1].mean(),
      time.time()-start, datetime.now() + timedelta(minutes=time_remaining), time_remaining/60))

    process_values = lambda values: (np.mean(values), np.std(values))

    aggregated_metrics = [item for sublist in [process_values(values) for values in losses] +
        [process_values(values) for values in discriminations] for item in sublist]
    metrics_writer.writerow([epoch+1, time.time()-start, *aggregated_metrics])

def main():
  tf.enable_eager_execution()
  args = parse_arguments()
  args.start_time = start_time
  args.loss_adversarial = 1.0
  args.loss_adversarial_power = 1.0
  args.loss_binary = args.loss_ground_truth = args.loss_relevancy = \
      args.loss_reconstruction = args.loss_patho_reconstruction = args.loss_variation = 0
  args.total_loss_weight = 1.0

  tf.logging.fatal("Starting GAN demo '{}', PID: {}".format(args.eid, os.getpid()))
  tf.logging.warning("Args: {}".format(args))

  # set up files and directories
  args.output_dir = os.path.join("output", args.eid)
  args.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
  args.figures_dir = os.path.join(args.output_dir, "figures")
  if not os.path.exists(args.figures_dir):
    os.makedirs(args.figures_dir)
  metrics_file = os.path.join(args.output_dir, "metrics.csv")
  with open(metrics_file, "a", buffering=1) as fh:
    metrics_writer = csv.writer(fh)
    if os.stat(metrics_file).st_size == 0:
      metrics_writer.writerow(["epoch", "epoch_time", "gen_loss_mean", "gen_loss_std", "disc_loss_mean", "disc_loss_std",
        "disc_on_real_mean", "disc_on_real_std", "disc_on_generated_mean", "disc_on_generated_std"])

    tf.logging.warning("Training for {} epochs of {} batches of size {}".format(args.epochs, args.batches, args.batch_size))
    model = NoiseToGaussianModel(args)
    train(model, *set_up_model(model, args), metrics_writer, args)

if __name__ == "__main__":
  start_time = time.time()
  # np.random.seed(42)
  # configure_logging()
  try:
    main()
  except Exception as ex:
    tf.logging.fatal("Exception occurred: {}".format(traceback.format_exc()))
  finally:
    tf.logging.info("Finished eval after {:.1f}m".format((time.time() - start_time) / 60))
