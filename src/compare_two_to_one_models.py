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
import tensorflow as tf

from evaluation import Evaluation
from one_way_evaluations import TwoImagesToOneImageEvaluation
from utils import load_checkpoint, load_model


def parse_arguments():
  parser = ArgumentParser()

  parser.add_argument("--data-dir", type=str, required=True,
      help="Directory containing the data set")
  parser.add_argument("--second-data-dir", type=str,
      help="Directory containing the second input data set, if different from the main input data dir")
  parser.add_argument("--eval-dir", type=str, required=True,
      help="Directory of the evaluation to test (output)")
  parser.add_argument("--second-eval-dir", type=str, required=True,
      help="Directory of the second evaluation to test (output)")
  parser.add_argument("--model-name", type=str, required=True,
      help="Name of the model to instantiate")
  parser.add_argument("--epoch", type=int, required=True,
      help="The epoch of the model to load")
  parser.add_argument("--second-model-name", type=str, required=True,
      help="Name of the second model to instantiate")
  parser.add_argument("--second-epoch", type=int, required=True,
      help="The epoch of the second model to load")
  parser.add_argument("--description", type=str, required=True,
      help="Description of the images")

  parser.add_argument("--image-count", type=int, default=1,
      help="The number of images to generate")
  parser.add_argument("--rows", type=int, default=8,
      help="The number of rows to generate")
  parser.add_argument("--columns", type=int, default=2,
      help="The number of columns to generate")

  return parser.parse_args()

def main(start_time):
  tf.enable_eager_execution()

  # handle arguments and config
  args = parse_arguments()
  args.start_time = start_time

  tf.logging.info("Args: {}".format(args))

  args.has_colored_input = True
  args.has_colored_second_input = False
  args.has_colored_target = True
  # args.has_colored_target = not True
  args.has_noise_input = False
  args.noise_dimensions = None
  args.match_pattern = None
  args.input_type = "image"
  args.second_input_type = "patho"
  args.target_type = "image"
  # args.target_type = "patho"
  args.target_data_dir = args.data_dir
  args.batch_size = args.rows * args.columns

  first_model = load_model(args)
  first_model_name = args.model_name
  args.model_name = args.second_model_name
  second_model = load_model(args)
  args.model_name = first_model_name
  first_generator = first_model.get_generator()
  second_generator = second_model.get_generator()
  args.checkpoint_dir = os.path.join("output", args.eval_dir, "checkpoints")
  load_checkpoint(args, checkpoint_number=args.epoch//25, generator=first_generator)
  args.checkpoint_dir = os.path.join("output", args.second_eval_dir, "checkpoints")
  load_checkpoint(args, checkpoint_number=args.second_epoch//25, generator=second_generator)

  evaluation = TwoImagesToOneImageEvaluation(first_model, args)
  # evaluation = ImageToImageEvaluation(first_model, args)
  evaluation.load_data()

  for image_number, batch in enumerate(evaluation.data_set):
    if image_number == args.image_count:
      break

    tf.logging.info("Generating image {}/{}".format(image_number+1, args.image_count))
    plt.figure(figsize=(32, 32))
    # plt.suptitle("{}: {} samples at epoch {} ({}/{})".format(args.eval_dir,
    #   args.rows*args.columns, args.epoch, image_number+1, args.image_count), fontsize=16)

    batch_input, _ = batch
    first_predictions = first_generator(batch_input, training=False)
    second_predictions = second_generator(batch_input, training=False)
    for i in range(first_predictions.shape[0]):
      plt.subplot(args.rows, args.columns*4, i*4+1)
      Evaluation.plot_image(batch_input[i][:, :, :-1])
      plt.subplot(args.rows, args.columns*4, i*4+2)
      Evaluation.plot_image(batch_input[i][:, :, -1:])
      plt.subplot(args.rows, args.columns*4, i*4+3)
      Evaluation.plot_image(first_predictions[i])
      plt.subplot(args.rows, args.columns*4, i*4+4)
      Evaluation.plot_image(second_predictions[i])

    plt.tight_layout()
    figure_file = "{}_{:03d}.png".format(args.description, image_number+1)
    plt.savefig(figure_file)
    plt.close()

  tf.logging.info("Finished generating {} images".format(args.image_count))


if __name__ == "__main__":
  START_TIME = time.time()
  np.random.seed(42)
  tf.logging.set_verbosity(tf.logging.INFO)
  try:
    main(START_TIME)
  except Exception as ex:
    tf.logging.fatal("Exception occurred: {}".format(traceback.format_exc()))
  finally:
    tf.logging.info("Finished eval after {:.1f}m".format((time.time() - START_TIME) / 60))
