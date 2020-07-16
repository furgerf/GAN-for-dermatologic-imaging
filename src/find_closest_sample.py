#!/usr/bin/env python

# pylint: disable=wrong-import-position

import os
import time
import traceback
from argparse import ArgumentParser

import matplotlib
matplotlib.use("Agg")
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from evaluation import Evaluation
from one_way_evaluations import TwoImagesToOneImageEvaluation
from utils import configure_logging, load_checkpoint, load_model, load_images, image_subdir, load_image_names


def parse_arguments():
  parser = ArgumentParser()

  parser.add_argument("--eid", type=str, required=True,
      help="Directory of the evaluation to test (output)")
  parser.add_argument("--model-name", type=str, required=True,
      help="Name of the model to instantiate")
  parser.add_argument("--epoch", type=int, required=True,
      help="The epoch of the model to load")

  parser.add_argument("--data-dir", type=str, required=True,
      help="Directory of the input data")
  parser.add_argument("--second-data-dir", type=str,
      help="Directory containing the second input data set, if different from the maininput data dir")
  parser.add_argument("--target-data-dir", type=str,
      help="Directory of the target data")

  parser.add_argument("--noise", action="store_true",
      help="Specify if a noise-based model should be analyzed")

  parser.add_argument("--image-count", type=int, default=1,
      help="The number of images to generate")

  return parser.parse_args()


def find_nearest_sample(sample, dataset, batch_size):
  max_similarity = 0
  # min_error = 1e4
  nearest = None
  for batch_sample in dataset:
    similarity = tf.image.ssim(sample, batch_sample, 255).numpy()
    # error = tf.losses.mean_squared_error(sample, batch_sample).numpy()
    if similarity > max_similarity:
    # if error < min_error:
      nearest = batch_sample
      max_similarity = similarity
      # min_error = error
  tf.logging.info("Best similarity for image: {}".format(max_similarity))
  # tf.logging.info("Best error for image: {}".format(min_error))
  return nearest, max_similarity
  # return nearest, min_error

def legacy(args):
  args.has_colored_input = True
  args.has_colored_second_input = False
  args.has_colored_target = True
  args.has_noise_input = False
  args.noise_dimensions = None
  args.input_type = "image"
  args.second_input_type = "patho"
  args.target_type = "image"

  model = load_model(args)
  evaluation = TwoImagesToOneImageEvaluation(model, args)
  evaluation.load_data()
  def denormalize_samples(first, second):
    return ((first+1)*127.5, (second+1)*127.5)
  denormalized_data_set = evaluation.data_set.map(denormalize_samples)

  generator = model.get_generator()
  load_checkpoint(args, checkpoint_number=args.epoch//25, generator=generator)

  tf.logging.warning("Keep in mind that the closest sample potentially wasn't actually among the training data!")

  for image_number, batch in enumerate(evaluation.data_set):
    if image_number == args.image_count:
      break

    tf.logging.info("Generating images {}/{}".format(image_number+1, args.image_count))
    # sample_input = evaluation.prepare_batch(batch)
    sample_input, _ = batch

    plt.figure(figsize=(32, 32))
    plt.axis("off")
    # plt.suptitle("{}: {} closest samples ({}/{})".format(args.eid,
    #   args.batch_size, image_number+1, args.image_count), fontsize=16)

    generated_images = generator(sample_input, training=False)

    for i in range(args.batch_size):
      plt.subplot(args.batch_size, args.batch_size, args.batch_size*i+1)
      Evaluation.plot_image(sample_input[i, :, :, :-1], True)
      plt.subplot(args.batch_size, args.batch_size, args.batch_size*i+2)
      Evaluation.plot_image(sample_input[i, :, :, -1:], True)
      plt.subplot(args.batch_size, args.batch_size, args.batch_size*i+3)
      Evaluation.plot_image(generated_images[i], True)
      plt.subplot(args.batch_size, args.batch_size, args.batch_size*i+4)
      similar_sample, similarity = find_nearest_sample((generated_images[i:i+1]+1)*127.5, denormalized_data_set, args.batch_size)
      Evaluation.plot_image(similar_sample, False)
      plt.title("{}".format(similarity), fontsize=28)

    plt.tight_layout()
    figure_file = os.path.join("output", args.eid, "nearest_samples_ssim_{:03d}.png".format(image_number+1))
    plt.savefig(figure_file)
    plt.close()

def main(start_time):
  tf.enable_eager_execution()

  # handle arguments and config
  args = parse_arguments()
  args.start_time = start_time

  args.match_pattern = None
  args.batch_size = 4
  args.eval_dir = os.path.join("output", args.eid)
  if not os.path.exists(args.eval_dir):
    args.eval_dir = os.path.join("old-output", args.eid)
  args.checkpoint_dir = os.path.join(args.eval_dir, "checkpoints")

  tf.logging.info("Args: {}".format(args))

  if not args.noise:
    tf.logging.warning("Needs to be cleaned/fixed/re-implemented first")
    legacy(args)
  else:
    args.noise_dimensions = 100
    args.has_colored_target = True

    image_names = load_image_names(args.data_dir)
    images = load_images(image_names, args.data_dir, image_subdir)
    targets = (tf.convert_to_tensor(images) + 1) * 127.5

    model = load_model(args)
    generator = model.get_generator()
    # discriminator = model.get_discriminator()
    load_checkpoint(args, checkpoint_number=args.epoch//25, generator=generator)#, discriminator=discriminator)

    args.rows = 4
    args.columns = 2

    for i in trange(args.image_count):
      plt.figure(figsize=(32, 32))
      plt.axis("off")
      for j, noise in enumerate(tf.data.Dataset.from_tensor_slices(
          tf.random_normal([args.rows * args.columns, args.noise_dimensions])).batch(args.batch_size)):
        samples = generator(noise, training=False)
        for k, sample in enumerate(samples):
          plt.subplot(args.rows, 2*args.columns, 2*j*args.batch_size + 2*k + 1)
          Evaluation.plot_image(sample)
          plt.subplot(args.rows, 2*args.columns, 2*j*args.batch_size + 2*k + 2)
          similar_sample, similarity = find_nearest_sample((sample+1)*127.5, targets, args.batch_size)
          Evaluation.plot_image(similar_sample, similarity, denormalize=False)
      plt.tight_layout()
      figure_file = os.path.join(args.eval_dir, "nearest_samples_ssim_{:03d}.png".format(i))
      plt.savefig(figure_file)
      plt.close()

  tf.logging.info("Finished generating {} images".format(args.image_count))


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
