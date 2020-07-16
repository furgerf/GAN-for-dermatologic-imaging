#!/usr/bin/env python

# pylint: disable=wrong-import-position,too-many-statements

import os
import time
import traceback
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.misc import imsave
from tqdm import tqdm

from evaluation import Evaluation
from perceptual_scores import PerceptualScores
from reconstruction_scores import ReconstructionScores
from utils import (configure_logging, image_subdir, load_checkpoint, truncate_input,
                   load_image_names, load_images, load_model)


def parse_arguments():
  parser = ArgumentParser()

  parser.add_argument("--evals-dir", type=str, required=True,
      help="Directory of the evaluations containing the GAN evals")
  parser.add_argument("--model-name", type=str,
      help="Name of the model to instantiate")
  parser.add_argument("--batch-size", type=int, default=16,
      help="The number of samples in one batch")
  parser.add_argument("--samples", type=int, default=1000,
      help="The number of samples to generate for each class")
  parser.add_argument("--truncation-threshold", type=float, default=None,
      help="The threshold above which noise components are resampled")

  return parser.parse_args()


def main(start_time):
  args = parse_arguments()
  args.start_time = start_time

  args.has_colored_target = True
  args.gan_dir = os.path.join("isic-gans", args.evals_dir)
  args.noise_dimensions = 100
  args.output_dir = os.path.join(args.gan_dir, "train")

  tf.enable_eager_execution()
  tf.logging.info("Args: {}".format(args))

  tf.logging.info("Loading gens")
  model = load_model(args)

  classes = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
  gens = {}
  for cls in classes:
    matching_evals = [e for e in os.listdir(args.gan_dir) if cls in e]
    assert len(matching_evals) == 1 and "final-isic19" in matching_evals[0]
    args.checkpoint_dir = os.path.join(args.gan_dir, matching_evals[0], "checkpoints")
    tf.logging.info("Loading model for {} from {}".format(cls, args.checkpoint_dir))
    generator = model.get_generator()
    load_checkpoint(args, checkpoint_number=None, generator=generator)
    gens[cls] = generator

  tf.logging.info("Creating output directory {}".format(args.output_dir))
  os.makedirs(args.output_dir)

  for cls in classes:
    tf.logging.warning("Generating samples for class {}".format(cls))
    samples_dir = os.path.join(args.output_dir, cls)
    os.makedirs(samples_dir)
    generator = gens[cls]
    data_set = tf.data.Dataset.from_tensor_slices(tf.random_normal([args.samples, args.noise_dimensions])).batch(args.batch_size)
    image_number = 0
    for batch in tqdm(data_set, total=args.samples // args.batch_size + 1):
      if args.truncation_threshold:
        batch = truncate_input(batch, args.truncation_threshold)
      for image in generator(batch):
        imsave(os.path.join(samples_dir, "sample_{:05d}.png".format(image_number)), image)
        image_number += 1

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
