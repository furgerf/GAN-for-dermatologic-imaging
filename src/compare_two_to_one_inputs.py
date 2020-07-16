#!/usr/bin/env python

import math
import os
import time
import traceback
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from evaluation import Evaluation
from utils import (load_checkpoint, load_image_names, load_images, load_model,
                   logistic)


def parse_arguments():
  parser = ArgumentParser()

  parser.add_argument("--eval-dir", type=str, required=True,
      help="Directory of the evaluation to test (output)")
  parser.add_argument("--model-name", type=str, required=True,
      help="Name of the model to instantiate")
  parser.add_argument("--epoch", type=int, required=True,
      help="The epoch of the model to load")
  parser.add_argument("--batch-size", type=int, default=16,
      help="The number of samples in one batch")

  parser.add_argument("--data-dir", type=str, required=True,
      help="Directory containing the data set")
  parser.add_argument("--second-data-dir", type=str,
      help="Directory containing the second input data set, if different from the main input data dir")

  parser.add_argument("--conditioned-discriminator", type=bool, default=False,
      help="Specify if the discriminator should discriminate the combination of generation input+output")

  parser.add_argument("--width", type=int, default=3200,
      help="The width of the resulting image (multiple of 100)")
  parser.add_argument("--height", type=int, default=3200,
      help="The height of the resulting image (multiple of 100)")
  parser.add_argument("--description", type=str, default=None,
      help="An optional description of the images")
  parser.add_argument("--image-count", type=int, default=1,
      help="The number of images to generate")

  return parser.parse_args()

def build_inputs(args, training_input_image_names, training_second_input_image_names,
      test_input_image_names, test_second_input_image_names, image_number):
  input_images = []
  second_input_images = []

  input_images.append(load_images(training_input_image_names, args.data_dir, args.input_type, \
      original=True, flip_lr=False, flip_ud=False))
  second_input_images.append(load_images(training_second_input_image_names, args.second_data_dir, args.second_input_type, \
      original=True, flip_lr=False, flip_ud=False))

  input_images.append(load_images(training_input_image_names, args.data_dir, args.input_type, \
      original=False, flip_lr=True, flip_ud=False))
  second_input_images.append(load_images(training_second_input_image_names, args.second_data_dir, args.second_input_type, \
      original=False, flip_lr=True, flip_ud=False))

  input_images.append(load_images(training_input_image_names, args.data_dir, args.input_type, \
      original=False, flip_lr=False, flip_ud=True))
  second_input_images.append(load_images(training_second_input_image_names, args.second_data_dir, args.second_input_type, \
      original=False, flip_lr=False, flip_ud=True))

  input_images.append(load_images(test_input_image_names, args.test_data_dir, args.input_type, \
      original=True, flip_lr=False, flip_ud=False))
  second_input_images.append(load_images(test_second_input_image_names, args.test_second_data_dir, args.second_input_type, \
      original=True, flip_lr=False, flip_ud=False))

  input_images.append(np.zeros(shape=(len(training_input_image_names), *input_images[-1].shape[1:-1], 3), dtype=np.float32))
  second_input_images.append(load_images(training_second_input_image_names, args.second_data_dir, args.second_input_type, \
      original=True, flip_lr=False, flip_ud=False))

  input_images.append(np.zeros(shape=(len(training_input_image_names), *input_images[-1].shape[1:-1], 3), dtype=np.float32))
  second_input_images.append(load_images(training_second_input_image_names, args.second_data_dir, args.second_input_type, \
      original=False, flip_lr=True, flip_ud=False))

  input_images.append(np.zeros(shape=(len(training_input_image_names), *input_images[-1].shape[1:-1], 3), dtype=np.float32))
  second_input_images.append(load_images(training_second_input_image_names, args.second_data_dir, args.second_input_type, \
      original=False, flip_lr=False, flip_ud=True))

  input_images.append(np.zeros(shape=(len(training_input_image_names), *input_images[-1].shape[1:-1], 3), dtype=np.float32))
  second_input_images.append(load_images(test_second_input_image_names, args.test_second_data_dir, args.second_input_type, \
      original=True, flip_lr=False, flip_ud=False))

  return input_images, second_input_images

def generate_samples(args, generator, discriminator, training_input_image_names, training_second_input_image_names,
      test_input_image_names, test_second_input_image_names, image_number):
  labels = ["original", "lrflip", "udflip", "test", "blank", "blank-lrflip", "blank-udflip", "blank-test"]
  input_images, second_input_images = build_inputs(args, training_input_image_names, training_second_input_image_names,
      test_input_image_names, test_second_input_image_names, image_number)

  input_images = np.concatenate(input_images, axis=0)
  second_input_images = np.concatenate(second_input_images, axis=0)
  combined_input_images = np.concatenate([input_images, second_input_images], axis=-1)
  data_set_size = len(input_images)
  data_set = tf.data.Dataset.from_tensor_slices(combined_input_images).batch(args.batch_size)
  del input_images
  del second_input_images
  del combined_input_images

  plt.figure(figsize=(int(args.width/100), int(args.height/100)))
  plt.suptitle("{}: different G inputs".format(args.eval_dir), fontsize=28)
  plt.tight_layout()
  column_count = 2
  rows = math.ceil(data_set_size / column_count)
  images_per_group = 4
  columns = images_per_group*column_count
  image_index = 1
  for inputs in tqdm(data_set, total=data_set_size // args.batch_size):
    generated_samples = generator(inputs, training=True)
    if args.conditioned_discriminator:
      disc_input = tf.concat([inputs, generated_samples], axis=-1)
    else:
      disc_input = generated_samples
    disc_predictions = logistic(discriminator(disc_input, training=True))
    for i in range(inputs.shape[0]):
      plt.subplot(rows, columns, image_index)
      label_index = image_index // images_per_group // len(training_input_image_names)
      in_label_index = (image_index // images_per_group) % len(training_input_image_names)
      Evaluation.plot_image(inputs[i][:, :, :-1], title="{} {}".format(labels[label_index], in_label_index+1), title_size=20)
      image_index += 1
      plt.subplot(rows, columns, image_index)
      Evaluation.plot_image(inputs[i][:, :, -1:])
      image_index += 1
      plt.subplot(rows, columns, image_index)
      Evaluation.plot_image(generated_samples[i], title=disc_predictions[i].numpy(), title_size=20)
      image_index += 1
      plt.subplot(rows, columns, image_index)
      Evaluation.plot_image(generated_samples[i]-inputs[i][:, :, :-1])
      image_index += 1

  figure_file = os.path.join(args.output_dir, "generator_on_inputs{}_{:03d}.png".format(
    "_{}".format(args.description) if args.description else "", image_number+1))
  plt.savefig(figure_file)
  plt.close()

def main(start_time):
  tf.enable_eager_execution()

  # handle arguments and config
  args = parse_arguments()
  args.start_time = start_time

  tf.logging.info("Args: {}".format(args))
  args.second_data_dir = args.second_data_dir or args.data_dir

  args.input_type = "image"
  args.second_input_type = "patho"
  args.has_colored_input = True
  args.has_colored_second_input = False
  args.has_colored_target = True
  args.has_noise_input = False
  args.discriminator_classes = 1
  if os.path.exists(os.path.join("output", args.eval_dir, "checkpoints")):
    args.output_dir = os.path.join("output", args.eval_dir)
  else:
    args.output_dir = os.path.join("old-output", args.eval_dir)
  args.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
  model = load_model(args)
  generator = model.get_generator()
  discriminator = model.get_discriminator()
  load_checkpoint(args, checkpoint_number=args.epoch//25, generator=generator, discriminator=discriminator)

  images_per_type = 2
  training_input_image_names = np.array(load_image_names(args.data_dir))
  training_second_input_image_names = np.array(load_image_names(args.second_data_dir))
  args.test_data_dir = args.data_dir + "-TEST"
  args.test_second_data_dir = args.second_data_dir + "-TEST"
  test_input_image_names = np.array(load_image_names(args.test_data_dir))
  test_second_input_image_names = np.array(load_image_names(args.test_second_data_dir))
  input_indexes = np.random.choice(min(len(training_input_image_names), len(test_input_image_names)),
      args.image_count*images_per_type, replace=False)

  for image_number in range(args.image_count):
    indexes = input_indexes[images_per_type*image_number:images_per_type*(image_number+1)]
    generate_samples(args, generator, discriminator, training_input_image_names[indexes],
        training_second_input_image_names[indexes], test_input_image_names[indexes], test_second_input_image_names[indexes], image_number)

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
