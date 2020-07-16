#!/usr/bin/env python

import csv
import os
import time
import traceback
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils import (configure_logging, data_subdirs, flatten, load_checkpoint,
                   load_image_names, load_images, load_model, logistic)


def parse_arguments():
  parser = ArgumentParser()

  parser.add_argument("--eval-dir", type=str, required=True,
      help="Directory of the evaluation to test (output)")
  parser.add_argument("--model-name", type=str, required=True,
      help="Name of the model to instantiate")
  parser.add_argument("--epoch", type=int,
      help="The epoch of the model to load")
  parser.add_argument("--batch-size", type=int, default=16,
      help="The number of samples in one batch")
  parser.add_argument("--split-data", action="store_true",
      help="Specify if the data should be processed in splits")

  parser.add_argument("--data-dir", type=str, required=True,
      help="Directory containing the data set")
  parser.add_argument("--second-data-dir", type=str,
      help="Directory containing the second input data set, if different from the main input data dir")
  parser.add_argument("--target-data-dir", type=str,
      help="Directory containing the targer data set, if different from the input data dir")
  parser.add_argument("--input-type", type=str, default="noise", choices=list(data_subdirs.keys()) + ["noise"],
      help="The type of the input for the generation")
  parser.add_argument("--second-input-type", type=str, choices=list(data_subdirs.keys()) + ["noise"],
      help="The type of the secondary input for the generation")
  parser.add_argument("--target-type", type=str, default="image", choices=data_subdirs.keys(),
      help="The type of the target of the generation")

  parser.add_argument("--conditioned-discriminator", type=bool, default=False,
      help="Specify if the discriminator should discriminate the combination of generation input+output")

  parser.add_argument("--generator", action="store_true", dest="evaluate_generator",
      help="Specify if the generator should be evaluated")
  parser.add_argument("--no-discriminator", action="store_false", dest="evaluate_discriminator",
      help="Specify if the discriminator shouldn't be evaluated")


  parser.add_argument("--no-test", action="store_false", dest="compute_test_scores",
      help="Specify if no test data should be evaluated")
  parser.add_argument("--test-data-samples", type=int, default=0,
      help="Use the first X source images for testing (before augmentation)")

  parser.add_argument("--no-store-scores", action="store_false", dest="store_scores",
      help="Specify if the computed scores shouldn't be stored")

  return parser.parse_args()

def get_training_data(args, current_input_names, current_second_input_names, current_target_names,
    brightness=0, original=True, flip_lr=False, flip_ud=False):
  if brightness:
    assert original and not flip_lr and not flip_ud
    target_images = tf.minimum(tf.maximum(load_images(current_target_names, args.target_data_dir, args.target_type) + brightness, -1), 1)
  else:
    target_images = load_images(current_target_names, args.target_data_dir, args.target_type,
        original=original, flip_lr=flip_lr, flip_ud=flip_ud)
  input_images = tf.random_normal([len(current_input_names), 100]) if args.has_noise_input \
      else load_images(current_input_names, args.data_dir, args.input_type)
  if args.second_input_type:
    second_input_images = load_images(current_second_input_names, args.second_data_dir, args.second_input_type)
    combined_input_images = np.concatenate([input_images, second_input_images], axis=-1)
    data_set = tf.data.Dataset.from_tensor_slices((combined_input_images, target_images)).batch(args.batch_size)
    del second_input_images
    del combined_input_images
  else:
    data_set = tf.data.Dataset.from_tensor_slices((input_images, target_images)).batch(args.batch_size)
  del input_images
  del target_images
  return data_set

def get_original_data(args, current_input_names, current_second_input_names, current_target_names):
  return get_training_data(args, current_input_names, current_second_input_names, current_target_names)

def get_minus_10_data(args, current_input_names, current_second_input_names, current_target_names):
  return get_training_data(args, current_input_names, current_second_input_names, current_target_names, -0.2)

def get_plus_10_data(args, current_input_names, current_second_input_names, current_target_names):
  return get_training_data(args, current_input_names, current_second_input_names, current_target_names, 0.2)

def get_lrflip_data(args, current_input_names, current_second_input_names, current_target_names):
  return get_training_data(args, current_input_names, current_second_input_names, current_target_names, original=False, flip_lr=True)

def get_udflip_data(args, current_input_names, current_second_input_names, current_target_names):
  return get_training_data(args, current_input_names, current_second_input_names, current_target_names, original=False, flip_ud=True)

def get_test_data(args, current_input_names, current_second_input_names, current_target_names):
  target_images = load_images(current_target_names, args.test_target_data_dir, args.target_type)
  input_images = tf.random_normal([len(current_input_names), 100]) if args.has_noise_input else \
      load_images(current_input_names, args.test_data_dir, args.input_type)
  if args.second_input_type:
    second_input_images = load_images(current_second_input_names, args.test_second_data_dir, args.second_input_type)
    combined_input_images = np.concatenate([input_images, second_input_images], axis=-1)
    data_set = tf.data.Dataset.from_tensor_slices((combined_input_images, target_images)).batch(args.batch_size)
    del second_input_images
    del combined_input_images
  else:
    data_set = tf.data.Dataset.from_tensor_slices((input_images, target_images)).batch(args.batch_size)
  del input_images
  del target_images
  return data_set

def get_disc_predictions(get_data_method, generator, discriminator, args, input_image_names, second_input_image_names, target_image_names):
  disc_predictions = []
  def process_images(current_input_names, current_second_input_names, current_target_names):
    data_set = get_data_method(args, current_input_names, current_second_input_names, current_target_names)
    for batch in tqdm(data_set, total=len(current_input_names) // args.batch_size):
      inputs, targets = batch
      if generator:
        targets = generator(inputs, training=True)
      if args.conditioned_discriminator:
        disc_input = tf.concat([inputs, targets], axis=-1)
      else:
        disc_input = targets
      disc_predictions.extend(logistic(discriminator(disc_input, training=True)))
    del data_set

  if args.split_data:

    tf.logging.warning("Splitting data into 2 parts")
    split = len(input_image_names) // 2
    process_images(input_image_names[:split], second_input_image_names[:split] if second_input_image_names else None, target_image_names[:split])
    process_images(input_image_names[split:], second_input_image_names[split:] if second_input_image_names else None, target_image_names[split:])

    # tf.logging.warning("Splitting data into 3 parts")
    # split = len(input_image_names) // 3
    # process_images(input_image_names[:split], second_input_image_names[:split] if second_input_image_names else None, \
    #     target_image_names[:split])
    # process_images(input_image_names[split:2*split], second_input_image_names[split:2*split] if second_input_image_names else None, \
    #     target_image_names[split:2*split])
    # process_images(input_image_names[2*split:], second_input_image_names[2*split:] if second_input_image_names else None, \
    #     target_image_names[2*split:])

  else:
    process_images(input_image_names, second_input_image_names, target_image_names)

  return np.mean(disc_predictions), np.std(disc_predictions)

def evaluate_discriminations(args, generator, discriminator,
    training_input_image_names, training_second_input_image_names, training_target_image_names,
    test_input_image_names, test_second_input_image_names, test_target_image_names):
  disc_on_training_mean, disc_on_training_std = get_disc_predictions(get_original_data,
          generator, discriminator, args, training_input_image_names, training_second_input_image_names, training_target_image_names)
  disc_summary = []
  discriminator.summary(print_fn=disc_summary.append)
  disc_summary = "\n".join(disc_summary)
  tf.logging.info("Discriminator model:\n{}".format(disc_summary))

  disc_on_training_minus_10_mean, disc_on_training_minus_10_std = get_disc_predictions(get_minus_10_data,
          generator, discriminator, args, training_input_image_names, training_second_input_image_names, training_target_image_names)
  disc_on_training_plus_10_mean, disc_on_training_plus_10_std = get_disc_predictions(get_plus_10_data,
          generator, discriminator, args, training_input_image_names, training_second_input_image_names, training_target_image_names)
  tf.logging.info("Disc on training: {:.3f}+-{:.3f}, minus 10%: {:.3f}+-{:.3f}, plus 10%: {:.3f}+-{:.3f}".format(
    disc_on_training_mean, disc_on_training_std, disc_on_training_minus_10_mean, disc_on_training_minus_10_std,
    disc_on_training_plus_10_mean, disc_on_training_plus_10_std))

  disc_on_training_lr_flip_mean, disc_on_training_lr_flip_std = get_disc_predictions(get_lrflip_data,
      generator, discriminator, args, training_input_image_names, training_second_input_image_names, training_target_image_names)
  tf.logging.info("Disc on training with LR flip: {:.3f}+-{:.3f}".format(
    disc_on_training_lr_flip_mean, disc_on_training_lr_flip_std))

  disc_on_training_ud_flip_mean, disc_on_training_ud_flip_std = get_disc_predictions(get_udflip_data,
      generator, discriminator, args, training_input_image_names, training_second_input_image_names, training_target_image_names)
  tf.logging.info("Disc on training with UD flip: {:.3f}+-{:.3f}".format(
    disc_on_training_ud_flip_mean, disc_on_training_ud_flip_std))

  if args.compute_test_scores:
    disc_on_test_mean, disc_on_test_std = get_disc_predictions(get_test_data,
        generator, discriminator, args, test_input_image_names, test_second_input_image_names, test_target_image_names)
    tf.logging.info("Disc on test: {:.3f}+-{:.3f}".format(
      disc_on_test_mean, disc_on_test_std))
  else:
    tf.logging.fatal("Skipping test set!")
    disc_on_test_mean = disc_on_test_std = np.nan

  tf.logging.info("On minus 10%: {:.3f}, on plus 10%: {:.3f}, on LR-flipped: {:.3f}, on UD-flipped: {:.3f}, on test: {:.3f}".format(
    disc_on_training_mean-disc_on_training_minus_10_mean, disc_on_training_mean-disc_on_training_plus_10_mean,
    disc_on_training_mean-disc_on_training_lr_flip_mean, disc_on_training_mean-disc_on_training_ud_flip_mean,
    disc_on_training_mean-disc_on_test_mean))

  if args.store_scores:
    results_file = os.path.join("analysis-results", "disc-predictions.csv")
    with open(results_file, "a", buffering=1) as fh:
      writer = csv.writer(fh)
      if os.stat(results_file).st_size == 0:
        metrics = ["training", "minus_10", "plus_10", "lrflip", "udflip", "test"]
        writer.writerow(["eid", "epoch"] + flatten([(metric + "_mean", metric + "_std") for metric in metrics]))
      eid = "G-" + args.eval_dir if generator else args.eval_dir
      writer.writerow([
        eid, args.epoch, disc_on_training_mean, disc_on_training_std,
        disc_on_training_minus_10_mean, disc_on_training_minus_10_std,
        disc_on_training_plus_10_mean, disc_on_training_plus_10_std,
        disc_on_training_lr_flip_mean, disc_on_training_lr_flip_std,
        disc_on_training_ud_flip_mean, disc_on_training_ud_flip_std,
        disc_on_test_mean, disc_on_test_std
        ])

def main(start_time):
  tf.enable_eager_execution()
  configure_logging()

  # handle arguments and config
  args = parse_arguments()
  args.start_time = start_time

  tf.logging.info("Args: {}".format(args))
  args.second_data_dir = args.second_data_dir or args.data_dir
  args.target_data_dir = args.target_data_dir or args.data_dir

  args.match_pattern = None
  args.augmentation_flip_lr = False
  args.augmentation_flip_ud = False
  args.has_noise_input = args.input_type == "noise"
  args.has_colored_input = args.input_type == "image"
  args.has_colored_second_input = args.second_input_type == "image"
  args.has_colored_target = args.target_type == "image"
  args.discriminator_classes = 1
  if os.path.exists(os.path.join("output", args.eval_dir, "checkpoints")):
    args.checkpoint_dir = os.path.join("output", args.eval_dir, "checkpoints")
  else:
    args.checkpoint_dir = os.path.join("old-output", args.eval_dir, "checkpoints")
  model = load_model(args)
  discriminator = model.get_discriminator()
  if args.evaluate_generator:
    generator = model.get_generator()
    load_checkpoint(args, checkpoint_number=(args.epoch+24)//25 if args.epoch else args.epoch, generator=generator, discriminator=discriminator)
  else:
    generator = None
    load_checkpoint(args, checkpoint_number=(args.epoch+24)//25 if args.epoch else args.epoch, discriminator=discriminator)

  # assert not (bool(args.test_data_dir) and bool(args.test_data_samples)), \
  #     "either use a part of the training data for test *OR* some actual test data"

  training_input_image_names = load_image_names(args.data_dir)
  training_second_input_image_names = load_image_names(args.second_data_dir) if args.second_input_type else None
  training_target_image_names = load_image_names(args.target_data_dir)
  if training_second_input_image_names is not None:
    if len(training_input_image_names) > len(training_second_input_image_names):
      tf.logging.info("Input and second input data are different; shuffling inputs before reducing")
      np.random.shuffle(training_input_image_names)
      training_input_image_names = training_input_image_names[:len(training_second_input_image_names)]
    assert len(training_input_image_names) == len(training_second_input_image_names)
  assert len(training_target_image_names) == len(training_input_image_names)
  if args.data_dir != args.target_data_dir or args.second_data_dir != args.target_data_dir:
    tf.logging.info("Input and target data are different; shuffling targets before reducing")
    np.random.shuffle(training_target_image_names)
    training_target_image_names = training_target_image_names[:len(training_input_image_names)]
  if args.compute_test_scores:
    if args.test_data_samples: # remove data that's not actually in the training set from the training data
      if args.split_data:
        raise NotImplementedError()
      tf.logging.warning("Using the first {} unaugmented samples of the training data for testing".format(args.test_data_samples))
      test_input_image_names = training_input_image_names[:args.test_data_samples]
      training_input_image_names = training_input_image_names[args.test_data_samples:]
      if args.second_input_type:
        test_second_input_image_names = training_second_input_image_names[:args.test_data_samples]
        training_second_input_image_names = training_second_input_image_names[args.test_data_samples:]
      else:
        test_second_input_image_names = None
      test_target_image_names = training_target_image_names[:args.test_data_samples]
      training_target_image_names = training_target_image_names[args.test_data_samples:]
      args.test_data_dir = args.data_dir
      args.test_second_data_dir = args.second_data_dir
      args.test_target_data_dir = args.target_data_dir
    else:
      args.test_data_dir = args.data_dir + "-TEST"
      args.test_second_data_dir = args.second_data_dir + "-TEST"
      args.test_target_data_dir = args.target_data_dir + "-TEST"
      test_input_image_names = load_image_names(args.test_data_dir)
      test_second_input_image_names = load_image_names(args.test_second_data_dir) if args.second_input_type else None
      test_target_image_names = load_image_names(args.test_target_data_dir)
      if test_second_input_image_names is not None:
        if len(test_input_image_names) > len(test_second_input_image_names):
          tf.logging.info("TEST input and second input data are different; shuffling inputs before reducing")
          np.random.shuffle(test_input_image_names)
          test_input_image_names = test_input_image_names[:len(test_second_input_image_names)]
        assert len(test_input_image_names) == len(test_second_input_image_names)
      assert len(test_target_image_names) >= len(test_input_image_names)
      if args.test_data_dir != args.test_target_data_dir or args.test_second_data_dir != args.test_target_data_dir:
        tf.logging.info("TEST input and target data are different; shuffling targets before reducing")
        np.random.shuffle(test_target_image_names)
        test_target_image_names = test_target_image_names[:len(test_input_image_names)]
  else:
    test_input_image_names = test_second_input_image_names = test_target_image_names = None

  if args.evaluate_discriminator:
    # evaluate D on real images
    tf.logging.warning("Evaluating D")
    evaluate_discriminations(args, None, discriminator,
      training_input_image_names, training_second_input_image_names, training_target_image_names,
      test_input_image_names, test_second_input_image_names, test_target_image_names)

  if generator:
    # if there's also a generator, evalaute D on generated images
    tf.logging.warning("Evaluating G (evaluating D on generated images)")
    evaluate_discriminations(args, generator, discriminator,
      training_input_image_names, training_second_input_image_names, training_target_image_names,
      test_input_image_names, test_second_input_image_names, test_target_image_names)

  tf.logging.info("Finished evaluation")


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
