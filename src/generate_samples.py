#!/usr/bin/env python

# pylint: disable=wrong-import-position

import os
import time
import traceback
from argparse import ArgumentParser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import tensorflow as tf

from evaluation import Evaluation
from utils import configure_logging, load_checkpoint, load_model, logistic, truncate_input


def parse_arguments():
  parser = ArgumentParser()

  parser.add_argument("--eval-dir", type=str, required=True,
      help="Directory of the evaluation to test (output)")
  parser.add_argument("--model-name", type=str, required=True,
      help="Name of the model to instantiate")
  parser.add_argument("--epoch", type=int, required=True,
      help="The epoch of the model to load")

  parser.add_argument("--description", type=str, default=None,
      help="An optional description of the images")
  parser.add_argument("--image-count", type=int, default=1,
      help="The number of images to generate")
  parser.add_argument("--rows", type=int, default=8,
      help="The number of rows to generate")
  parser.add_argument("--columns", type=int, default=8,
      help="The number of columns to generate")
  parser.add_argument("--noise-dimensions", type=int, default=100,
      help="The number of dimensions of the noise vector")

  parser.add_argument("--colored", action="store_true",
      help="Specify if the model generates colored output")
  parser.add_argument("--titles", action="store_true",
      help="Specify if the discriminator output should be written as a title")
  parser.add_argument("--discriminator-classes", type=int, default=1,
      help="Specify the number of classes the discriminator is predicting")
  parser.add_argument("--truncation-threshold", type=float, default=None,
      help="The threshold above which noise components are resampled")

  return parser.parse_args()

def main(start_time):
  tf.enable_eager_execution()

  # handle arguments and config
  args = parse_arguments()
  args.start_time = start_time

  tf.logging.info("Args: {}".format(args))

  args.has_colored_target = args.colored
  args.output_dir = os.path.join("output", args.eval_dir)
  if not os.path.exists(args.output_dir):
    args.output_dir = os.path.join("old-output", args.eval_dir)
  args.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
  model = load_model(args)
  generator = model.get_generator()
  discriminator = model.get_discriminator()
  load_checkpoint(args, checkpoint_number=(args.epoch+24)//25, generator=generator, discriminator=discriminator)

  if args.truncation_threshold:
    tf.logging.warning("Truncating samples to {}".format(args.truncation_threshold))

  for image_number in trange(args.image_count):
    plt.figure(figsize=(32, 32))

    batch = tf.random_normal([args.rows*args.columns, args.noise_dimensions])
    if args.truncation_threshold:
      batch = truncate_input(batch, args.truncation_threshold)
    samples = generator(batch, training=True)
    if args.titles:
      predictions = logistic(discriminator(samples, training=True))

    for i in range(samples.shape[0]):
      plt.subplot(args.rows, args.columns, i+1)
      Evaluation.plot_image(samples[i], np.round(predictions[i].numpy(), 5) if args.titles else None)

    plt.tight_layout()
    figure_file = os.path.join(args.output_dir, "samples{}_{:03d}.png".format(
      "_{}".format(args.description) if args.description else "", image_number+1))
    plt.savefig(figure_file)
    plt.close()

    # also store store some of the images in HD
    plt.figure(figsize=(32, 32))
    for i in range(args.rows//2*args.columns//2):
      plt.subplot(args.rows//2, args.columns//2, i+1)
      Evaluation.plot_image(samples[i], np.round(predictions[i].numpy(), 5) if args.titles else None)
    plt.tight_layout()
    figure_file = os.path.join(args.output_dir, "samples{}_large_{:03d}.png".format(
      "_{}".format(args.description) if args.description else "", image_number+1))
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
