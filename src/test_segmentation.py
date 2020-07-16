#!/usr/bin/env python

import os
import time
import traceback
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
import tensorflow as tf

from utils import (configure_logging, load_checkpoint, load_image_names,
                   load_images, load_model)


def parse_arguments():
  parser = ArgumentParser()

  parser.add_argument("--eval-dir", type=str, required=True,
      help="Directory of the evaluation to test (output)")
  parser.add_argument("--model-name", type=str, required=True,
      help="Name of the model to instantiate")
  parser.add_argument("--epoch", type=int, required=True,
      help="The epoch of the model to load")
  parser.add_argument("--test-data", type=str, required=True,
      help="Directory of the data to test on")

  parser.add_argument("--match-pattern", type=str, default=None,
      help="Pattern for files to match")
  parser.add_argument("--batch-size", type=int, default=16,
      help="The number of samples in one batch")
  parser.add_argument("--score-name", type=str, default=None,
      help="Optional, name for the file to store scores")

  parser.add_argument("--cycle", action="store_true",
      help="Specify if the second generator of a CycleGAN should be used")

  return parser.parse_args()

def main(start_time):
  tf.enable_eager_execution()

  # handle arguments and config
  args = parse_arguments()
  args.start_time = start_time
  args.noise_dimensions = None
  args.has_colored_target = False
  args.output_dir = os.path.join("output", args.eval_dir)
  if not os.path.exists(args.output_dir):
    args.output_dir = os.path.join("old-output", args.eval_dir)
  args.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
  tf.logging.info("Args: {}".format(args))

  model = load_model(args)
  generator = model.get_generator()
  if args.cycle:
    tf.logging.warning("Loading second generator of CycleGAN")
    load_checkpoint(args, checkpoint_number=(args.epoch+24)//25, second_generator=generator)
  else:
    load_checkpoint(args, checkpoint_number=(args.epoch+24)//25, generator=generator)

  image_names = load_image_names(args.test_data, args.match_pattern)
  input_images = load_images(image_names, args.test_data, "image")
  target_images = load_images(image_names, args.test_data, "patho")

  target_data_set = tf.data.Dataset.from_tensor_slices((input_images, target_images)).batch(args.batch_size)

  tf.logging.info("Computing segmentation score over {} images in batches of {}".format(len(target_images), args.batch_size))
  scores = list()
  all_tp = 0
  all_fp = 0
  all_fn = 0
  with tqdm(total=len(target_images)) as pbar:
    for batch in target_data_set:
      inputs, targets = batch
      pbar.update(inputs.shape[0].value)
      generated_images = generator(inputs, training=False)

      predicted = tf.cast(generated_images >= 0, tf.uint8)
      actual = tf.cast(targets >= 0, tf.uint8)

      tp = tf.count_nonzero(predicted * actual)
      fp = tf.count_nonzero(predicted * (actual - 1))
      fn = tf.count_nonzero((predicted - 1) * actual)

      all_tp += tp
      all_fp += fp
      all_fn += fn

      precision = tp / (tp + fp)
      recall = tp / (tp + fn)
      f1 = (2 * precision * recall / (precision + recall)).numpy()

      if not np.isnan(f1):
        scores.append(f1)

  precision = all_tp / (all_tp + all_fp)
  recall = all_tp / (all_tp + all_fn)
  f1 = (2 * precision * recall / (precision + recall)).numpy()

  tf.logging.info("Segmentation score: {:.3f} ({:.3}+-{:.3})".format(f1, np.mean(scores), np.std(scores)))
  tf.logging.info("TP {}, FP {}, FN {}".format(all_tp, all_fp, all_fn))
  if args.score_name:
    with open(os.path.join(args.output_dir, args.score_name), "w") as fh:
      fh.write("{}\n".format(",".join([str(s) for s in scores])))

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
