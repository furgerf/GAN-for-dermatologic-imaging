#!/usr/bin/env python

# pylint: disable=wrong-import-position,too-many-statements

import os
import time
import traceback
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import cm
from scipy.misc import imsave
from tqdm import tqdm, trange

from evaluation import Evaluation
from perceptual_scores import PerceptualScores
from reconstruction_scores import ReconstructionScores
from utils import (configure_logging, image_subdir, load_checkpoint,
                   load_image_names, load_images, load_model, logistic,
                   truncate_input)


def parse_arguments():
  parser = ArgumentParser()

  parser.add_argument("--eval-dir", type=str, required=True,
      help="Directory of the evaluation to test (output)")
  parser.add_argument("--data-dir", type=str,
      help="Directory containing the data set")
  parser.add_argument("--model-name", type=str,
      help="Name of the model to instantiate")
  parser.add_argument("--epoch", type=int,
      help="The epoch of the model to load")
  parser.add_argument("--description", type=str, default=None,
      help="An optional description of the images")

  parser.add_argument("--batch-size", type=int, default=4,
      help="The number of samples in one batch")
  parser.add_argument("--sample-count", type=int, default=100,
      help="The number of samples to generate. Only considered for explicit sample generation")
  parser.add_argument("--noise-dimensions", type=int, default=100,
      help="The number of dimensions of the noise vector")
  parser.add_argument("--extractor-name", type=str, default="VGG19", choices=PerceptualScores.EXTRACTOR_NAMES,
      help="The name of the feature extractor to use")
  parser.add_argument("--truncation-threshold", type=float, default=None,
      help="The threshold above which noise components are resampled")

  parser.add_argument("--samples", action="store_true",
      help="Specify if samples should be generated")
  parser.add_argument("--perceptual", action="store_true",
      help="Specify if perceptual scores should be computed")
  parser.add_argument("--reconstruction", action="store_true",
      help="Specify if reconstruction scores should be computed")
  parser.add_argument("--features", action="store_true",
      help="Specify if features should be maximized")

  return parser.parse_args()


def generate_samples(generator, discriminator, args, sample_count):
  if os.path.exists(args.samples_dir):
    tf.logging.info("Skipping sample generation, directory '{}' exists already".format(args.samples_dir))
    return

  tf.logging.info("Generating {} samples in '{}'".format(sample_count, args.samples_dir))
  os.makedirs(args.samples_dir)

  if args.truncation_threshold:
    tf.logging.warning("Truncating samples to {}".format(args.truncation_threshold))

  discriminations = []
  image_number = 0
  data_set = tf.data.Dataset.from_tensor_slices(tf.random_normal([sample_count, args.noise_dimensions])).batch(args.batch_size)
  for batch in tqdm(data_set, total=sample_count // args.batch_size + 1):
    if args.truncation_threshold:
      batch = truncate_input(batch, args.truncation_threshold)
    images = generator(batch)
    discriminations.extend(logistic(discriminator(images)))
    for image in images:
      imsave(os.path.join(args.samples_dir, "sample{}_{:05d}.png".format(
        "_{}".format(args.description) if args.description else "", image_number)), image)
      image_number += 1
  assert image_number == sample_count

  tf.logging.warning("Discriminations: {:.5f}+-{:.5f}".format(np.mean(discriminations), np.std(discriminations)))

def compute_perceptual_scores(generator, discriminator, args):
  tf.logging.fatal("Computing perceptual scores")
  assert args.data_dir
  generate_samples(generator, discriminator, args, len(load_image_names(args.data_dir)))

  extractor = PerceptualScores(args)
  extractor.initialize()

  fid, mmd, high_dimensional, low_dimensional, _, _ = extractor.compute_scores_from_samples()
  tf.logging.warning("Clustering scores, high-dimensional: {:.3f}; low-dimensional: {:.3f}".format(high_dimensional, low_dimensional))
  tf.logging.warning("FID: {:.3f}, k-MMD: {:.3f}".format(fid, mmd))
  tf.logging.info("Perceptual scores (FID,MMD,high,low): {:.5f}, {:.5f}, {:.5f}, {:.5f}".format(
    fid, mmd, high_dimensional, low_dimensional))

def compute_reconstruction_scores(generator, args):
  tf.logging.fatal("Computing reconstruction scores")
  image_names = load_image_names(args.data_dir)
  np.random.shuffle(image_names)
  images = load_images(image_names[:args.batch_size], args.data_dir, image_subdir)
  targets = tf.convert_to_tensor(images)

  rows = 4
  columns = 3
  images, titles, all_similarity_losses = ReconstructionScores(generator, targets, args.noise_dimensions).compute_scores(True)

  # assert len(images) == len(titles) and len(images[0]) == len(titles[0])
  # for j, _ in enumerate(images):
  #   plt.figure(figsize=(32, 32))
  #   subplot = 1
  #   for i in range(0, len(images[0]), 10):
  #     if i == len(images[0])-1:
  #       break # the last image is printed afterwards
  #     plt.subplot(rows, columns, subplot)
  #     Evaluation.plot_image(images[j][i], titles[j][i])
  #     subplot += 1
  #   plt.subplot(rows, columns, subplot)
  #   Evaluation.plot_image(images[j][-1], "Final " + titles[j][-1])
  #   plt.subplot(rows, columns, subplot+1)
  #   Evaluation.plot_image(targets[j], "Target")
  #   plt.savefig(os.path.join(args.output_dir, "{}reconstruction_{:03d}.png".format(
  #     "{}_".format(args.description) if args.description else "", j)))

  # plot for report
  plt.figure(figsize=(30, 4*args.batch_size))
  for j, _ in enumerate(images):
    epochs_to_plot = [0, 5, 10, 50, 100]
    for subplot, i in enumerate(epochs_to_plot):
      plt.subplot(args.batch_size, len(epochs_to_plot)+1, subplot+1+j*(len(epochs_to_plot)+1))
      Evaluation.plot_image(images[j][i], titles[j][i])
    plt.subplot(args.batch_size, len(epochs_to_plot)+1, subplot+2+j*(len(epochs_to_plot)+1))
    Evaluation.plot_image(targets[j], "Target")
  plt.tight_layout()
  plt.savefig(os.path.join(args.output_dir, "{}reconstruction_report.png".format(
    "{}_".format(args.description) if args.description else "")))

  plt.figure()
  last_losses = all_similarity_losses[-1].numpy()
  plt.title("Similarity losses, final ({} epochs): {:.5f}+-{:.5f}".format(len(all_similarity_losses), last_losses.mean(), last_losses.std()))
  plt.xlabel("Epochs")
  plt.ylabel("Similarity loss")
  for i in range(all_similarity_losses[0].shape[0]):
    plt.plot(range(1, len(all_similarity_losses)+1), [similarity[i] for similarity in all_similarity_losses], label="Reconstruction {}".format(i+1))
  plt.legend()
  plt.savefig(os.path.join(args.output_dir, "{}reconstruction_loss-curves.png".format(
    "{}_".format(args.description) if args.description else "")))
  plt.close()

def maximize_activations(generator, discriminator, args):
  tf.logging.fatal("Maximizing feature activations")

  block = 0
  feature = 0
  learning_rate = 1e-2
  adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
  samples = 1
  disc_input = tf.Variable(tf.random_normal([samples, 256, 256, 3]))

  _ = discriminator(disc_input)
  disc_summary = []
  discriminator.summary(print_fn=disc_summary.append)
  disc_summary = "\n".join(disc_summary)
  tf.logging.info("Discriminator model:\n{}".format(disc_summary))

  epochs = 1000
  epochs_per_image = epochs // 10
  inputs = []
  activations = []
  for i in trange(epochs+1):
    with tf.GradientTape() as tape:
      inputs.append(disc_input.numpy() if (i % epochs_per_image) == 0 else None)
      _ = discriminator(disc_input)
      activation = discriminator.discriminators[0].activations[block][:, :, :, feature]
      activations.append(activation if (i % epochs_per_image) == 0 else None)
      loss = tf.reduce_sum(activation)
      gradient = tape.gradient(-loss, disc_input)
      adam.apply_gradients([(gradient, disc_input)])
      if (i+1) % epochs_per_image == 0:
        tf.logging.info("{}/{}: Activations: {}".format(i+1, epochs, ", ".join(["{:.1f}".format(tf.reduce_sum(activation[j])) for j in range(samples)])))

  tf.logging.info("Plotting...")
  plt.figure(figsize=(32, 24))
  plt.tight_layout()
  columns = epochs // epochs_per_image + 1
  for i in range(columns):
    for j in range(samples):
      plt.subplot(2*samples, columns, 1+i+j*2*columns)
      Evaluation.plot_image(inputs[epochs_per_image*i][j], title="Epoch {}".format(i*epochs_per_image) if j == 0 else "")
      plt.subplot(2*samples, columns, columns+1+i+j*2*columns)
      Evaluation.plot_image(tf.expand_dims(-activations[epochs_per_image*i][j], -1), title=str(tf.reduce_sum(activations[epochs_per_image*i][j]).numpy()))
  plt.suptitle("Learning rate {}, block {}, feature {}".format(learning_rate, block, feature))
  plt.savefig("block-{}-feature-{}.png".format(block, feature))

def main(start_time):
  args = parse_arguments()
  args.start_time = start_time

  args.target_data_dir = None
  args.output_dir = os.path.join("output", args.eval_dir)
  if not os.path.exists(args.output_dir):
    args.output_dir = os.path.join("old-output", args.eval_dir)
  args.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
  args.samples_dir = os.path.join(args.output_dir, "samples")
  args.discriminator_classes = 1

  tf.enable_eager_execution()
  tf.logging.info("Args: {}".format(args))

  tf.logging.info("Loading model")
  args.has_colored_target = True
  model = load_model(args)
  generator = model.get_generator()
  discriminator = model.get_discriminator()
  load_checkpoint(args, checkpoint_number=(args.epoch+24)//25 if args.epoch else args.epoch, generator=generator, discriminator=discriminator)

  if args.samples:
    generate_samples(generator, args, args.sample_count)

  if args.perceptual:
    compute_perceptual_scores(generator, discriminator, args)

  if args.reconstruction:
    compute_reconstruction_scores(generator, args)

  if args.features:
    maximize_activations(generator, discriminator, args)

if __name__ == "__main__":
  START_TIME = time.time()
  np.random.seed(42)
  configure_logging()
  try:
    main(START_TIME)
  except Exception as ex:
    tf.logging.fatal("Exception occurred: {}".format(traceback.format_exc()))
  finally:
    tf.logging.info("Finished eval after {:.1f}m".format((time.time() - START_TIME) / 60))
