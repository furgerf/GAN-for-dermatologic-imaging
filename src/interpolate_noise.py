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
from scipy.interpolate import interp1d
from tqdm import tqdm

from utils import load_checkpoint, load_model, logistic, slerp, configure_logging


def parse_arguments():
  parser = ArgumentParser()

  parser.add_argument("--eid", type=str, required=True,
      help="ID of the evaluation to test (output)")
  parser.add_argument("--model-name", type=str, required=True,
      help="Name of the model to instantiate")
  parser.add_argument("--epoch", type=int, required=True,
      help="The epoch of the model to load")

  parser.add_argument("--description", type=str, default=None,
      help="An optional description of the images")
  parser.add_argument("--image-count", type=int, default=1,
      help="The number of images to generate")
  parser.add_argument("--columns", type=int, default=8,
      help="The number of columns of the output image")
  parser.add_argument("--rows", type=int, default=8,
      help="The number of rows of the output image")
  parser.add_argument("--noise-dimensions", type=int, default=100,
      help="The number of dimensions of the noise vector")
  parser.add_argument("--width", type=int, default=3200,
      help="The width of the resulting image (multiple of 100)")
  parser.add_argument("--height", type=int, default=3200,
      help="The height of the resulting image (multiple of 100)")

  parser.add_argument("--spherical", action="store_true",
      help="Specify if spherical interpolation should be used")
  parser.add_argument("--single-interpolation", action="store_true",
      help="Specify if all samples of an image should be generated from interpolating a single pair")
  parser.add_argument("--grid-interpolation", action="store_true",
      help="Specify if all samples of an image should be generated from interpolating four samples")

  parser.add_argument("--colored", action="store_true",
      help="Specify if the model generates colored output")

  return parser.parse_args()

def main(start_time):
  tf.enable_eager_execution()

  # handle arguments and config
  args = parse_arguments()
  args.start_time = start_time

  tf.logging.info("Args: {}".format(args))

  assert not (args.single_interpolation and args.grid_interpolation)

  args.has_colored_target = args.colored
  args.discriminator_classes = 1
  args.eval_dir = os.path.join("output", args.eid)
  if not os.path.exists(args.eval_dir):
    args.eval_dir = os.path.join("old-output", args.eid)
  args.checkpoint_dir = os.path.join(args.eval_dir, "checkpoints")
  model = load_model(args)
  generator = model.get_generator()
  discriminator = model.get_discriminator()
  load_checkpoint(args, checkpoint_number=args.epoch//25, generator=generator, discriminator=discriminator)

  lerp = lambda val, low, high: interp1d([0, 1], np.vstack([low, high]), axis=0)(val)
  interpolate = lambda val, low, high: slerp(val, low, high) if args.spherical else lerp(val, low, high)
  disc_predictions = []

  for image_number in range(args.image_count):
    tf.logging.info("Generating image {}/{}".format(image_number+1, args.image_count))
    plt.figure(figsize=(int(args.width/100), int(args.height/100)))
    # plt.suptitle("{}: {} interpolations with {} steps ({}/{})".format(args.eval_dir,
    #   args.rows, args.columns, image_number+1, args.image_count), fontsize=16)

    if args.single_interpolation:
      start_end_samples = np.random.normal(size=(2, args.noise_dimensions))
    if args.grid_interpolation:
      corner_samples = np.random.normal(size=(4, args.noise_dimensions))

    with tqdm(total=args.rows*args.columns) as pbar:
      for i in range(args.rows):
        if args.grid_interpolation:
          row_samples = \
              (interpolate(i / (args.rows-1), corner_samples[0], corner_samples[2]),
              interpolate(i / (args.rows-1), corner_samples[1], corner_samples[3]))
        if not args.single_interpolation and not args.grid_interpolation:
          row_samples = np.random.normal(size=(2, args.noise_dimensions))
        for j in range(args.columns):
          pbar.update(1)
          if args.single_interpolation:
            sample = interpolate((i*args.columns+j) / args.rows / args.columns,
                start_end_samples[0], start_end_samples[1])
          else:
            # same for "normal" and for grid interpolation
            sample = interpolate(j / (args.columns-1), *row_samples)
          predictions = generator(tf.convert_to_tensor(sample.reshape(1, args.noise_dimensions), dtype=tf.float32),
              training=False)
          classification = logistic(discriminator(predictions, training=False).numpy())[0]
          disc_predictions.append(classification)
          # Evaluation.plot_image(predictions[i])
          plt.subplot(args.rows, args.columns, i*args.columns+j+1)
          plt.axis("off")
          # plt.title(classification, fontsize=20)
          if args.colored:
            plt.imshow(np.array((predictions[0]+1) * 127.5, dtype=np.uint8))
          else:
            plt.imshow(np.array((predictions[0, :, :, 0]+1) * 127.5, dtype=np.uint8), cmap="gray", vmin=0, vmax=255)
    tf.logging.info("Average disc prediction: {:.3f}+-{:.3f}".format(np.mean(disc_predictions), np.std(disc_predictions)))
    tf.logging.info("Saving image")
    plt.tight_layout()
    figure_file = os.path.join(args.eval_dir, "noise_interpolation{}_{:03d}.png".format(
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
