#!/usr/bin/env python

# pylint: disable=wrong-import-position,too-many-statements

import os
import time
import traceback
from argparse import ArgumentParser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from evaluation import Evaluation
from utils import load_checkpoint, load_model, logistic


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
  parser.add_argument("--search-samples", type=int, default=4,
      help="The number of samples to generate at each search step")
  parser.add_argument("--step-size", type=float,
      help="The distance to move in the various directions")
  parser.add_argument("--size-factor", type=float, default=0.9,
      help="The factor by which the step size is multiplied after each iteration")

  parser.add_argument("--colored", action="store_true",
      help="Specify if the model generates colored output")
  parser.add_argument("--discriminator-classes", type=int, default=1,
      help="Specify the number of classes the discriminator is predicting")

  return parser.parse_args()

def main(start_time):
  tf.enable_eager_execution()

  # handle arguments and config
  args = parse_arguments()
  args.start_time = start_time

  tf.logging.info("Args: {}".format(args))

  args.has_colored_target = args.colored
  args.checkpoint_dir = os.path.join("output", args.eval_dir, "checkpoints")
  model = load_model(args)
  generator = model.get_generator()
  discriminator = model.get_discriminator()
  load_checkpoint(args, checkpoint_number=args.epoch//25, generator=generator, discriminator=discriminator)

  gen_training = not False
  disc_training = False

  for image_number in range(args.image_count):
    tf.logging.info("Generating image {}/{}".format(image_number+1, args.image_count))
    plt.figure(figsize=(32, 32))

    inputs = tf.random_normal([args.search_samples, args.noise_dimensions])
    samples = generator(inputs, training=gen_training)
    predictions = logistic(discriminator(samples, training=disc_training))
    best_index = tf.argmax(predictions)
    best_index = best_index.numpy() if best_index.shape else best_index
    previous_prediction = predictions[best_index]
    plt.subplot(args.rows, args.columns, 1)
    Evaluation.plot_image(samples[best_index], np.round(predictions[best_index].numpy(), 5))
    previous_direction = None
    improvements = 0
    best_input = inputs[best_index]
    if args.step_size is not None:
      current_step_size = args.step_size

    for i in range(1, args.rows*args.columns):
      tf.logging.info("Looking for image {}/{}, previous prediction: {}{}".format(
        i+1, args.rows*args.columns, previous_prediction,
        "" if args.step_size is None else ", step: {:.3f}".format(current_step_size)))
      # get new possible directions to move
      directions = tf.random_normal([args.search_samples, args.noise_dimensions], stddev=0.1)
      if previous_direction is not None:
        directions = tf.concat([[previous_direction], directions[1:, :]], axis=0)

      # obtain new inputs by moving previous input into the various directions
      lengths = [tf.norm(direction).numpy() for direction in directions]
      tf.logging.debug("Direction lengths: {}".format(",".join([str(l) for l in lengths])))
      inputs = tf.reshape(tf.tile(best_input, [args.search_samples]), (-1, args.noise_dimensions))
      if args.step_size is None:
        inputs = inputs + directions
      else:
        directions = [direction * current_step_size / tf.norm(direction) for direction in directions]
        inputs = inputs + directions

      # get new sampels and predictions
      samples = generator(inputs, training=gen_training)
      predictions = logistic(discriminator(samples, training=disc_training))
      best_index = tf.argmax(predictions)
      best_index = best_index.numpy() if best_index.shape else best_index
      tf.logging.debug("Best previous input: {}, input at best position: {}, direction: {}".format(
        best_input[0], inputs[best_index, 0], directions[best_index][0]))

      if previous_direction is not None and best_index == 0:
        tf.logging.info("Going into the same direction again!")

      if predictions[best_index].numpy() > previous_prediction.numpy():
        previous_prediction = predictions[best_index]
        previous_direction = directions[best_index]
        best_input = inputs[best_index]
        plt.subplot(args.rows, args.columns, i+1)
        Evaluation.plot_image(samples[best_index], np.round(predictions[best_index].numpy(), 5))
        improvements += 1
      else:
        previous_direction = None
        tf.logging.info("No improvement found")

      if args.step_size is not None:
        current_step_size *= args.size_factor

    tf.logging.info("Improved the original image {} times ({:.1f}%)".format(
      improvements, 100. * improvements / (args.rows*args.columns-1)))

    plt.tight_layout()
    figure_file = os.path.join("output", args.eval_dir, "samples{}_{:03d}.png".format(
      "_{}".format(args.description) if args.description else "", image_number+1))
    plt.savefig(figure_file)
    plt.close()

  tf.logging.info("Finished generating {} images".format(args.image_count))


if __name__ == "__main__":
  START_TIME = time.time()
  # np.random.seed(42)
  tf.logging.set_verbosity(tf.logging.INFO)
  try:
    main(START_TIME)
  except Exception as ex:
    tf.logging.fatal("Exception occurred: {}".format(traceback.format_exc()))
  finally:
    tf.logging.info("Finished eval after {:.1f}m".format((time.time() - START_TIME) / 60))
