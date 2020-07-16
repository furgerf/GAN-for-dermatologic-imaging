#!/usr/bin/env python

import os
import time
import traceback
from argparse import ArgumentParser
from glob import glob

import numpy as np
import tensorflow as tf
from scipy.misc import imread, imsave

from utils import (get_hand_segmentation_for_image, get_combined_segmentation_for_image,
                   get_patho_segmentation_for_image, hand_subdir, image_subdir, data_subdirs,
                   patho_subdir, combined_subdir)

def parse_arguments():
  parser = ArgumentParser()

  parser.add_argument("--output-dir", required=True, type=str,
      help="Target directory to store the patches in")
  parser.add_argument("--data-dir", type=str, default="training-data",
      help="Input directory from where to load images")

  parser.add_argument("--size", type=int, default=256,
      help="Size of the square patches in pixels")
  parser.add_argument("--step-size", type=float, default=0.5,
      help="Step size to use when looking for patches as a percentage of the patch size")
  parser.add_argument("--min-hand", type=float, default=1.0,
      help="Minimum percentage of hand pixels")
  parser.add_argument("--max-hand", type=float, default=1.0,
      help="Maximum percentage of hand pixels")
  parser.add_argument("--min-patho", type=float, default=0.0,
      help="Minimum percentage of pathology pixels")
  parser.add_argument("--max-patho", type=float, default=1.0,
      help="Maximum percentage of pathology pixels")

  parser.add_argument("--match-pattern", type=str, default=None,
      help="Specify pattern for files to match")
  parser.add_argument("--verify", action="store_true",
      help="Verify data integrity if specified")

  return parser.parse_args()

def verify_data_integrity(image_dir, hand_dir, patho_dir, combined_dir):
  # count images in the directories
  images_count = len(glob("{}/*.png".format(image_dir)))
  hand_count = len(glob("{}/*.png".format(hand_dir)))
  patho_count = len(glob("{}/*.png".format(patho_dir)))
  combined_count = len(glob("{}/*.png".format(combined_dir)))
  tf.logging.info("Image file counts: {}/{}/{}/{} (images/hand/patho/combined)".format(
    images_count, hand_count, patho_count, combined_count))
  assert images_count == hand_count and images_count == patho_count and images_count == combined_count

  # check file names
  for file_name in glob("{}/*.png".format(image_dir)):
    assert os.path.isfile(get_hand_segmentation_for_image(file_name, hand_dir))
    assert os.path.isfile(get_patho_segmentation_for_image(file_name, patho_dir))
    assert os.path.isfile(get_combined_segmentation_for_image(file_name, combined_dir))

  tf.logging.info("There seems to be exactly one hand, pathology, and combined segmentation per image")

def add_patch_to_file_name(file_name, patch_number):
  assert patch_number < 1e5
  return "{}_patch_{:04d}.png".format(os.path.splitext(file_name)[0], patch_number)

def find_patches_in_file(image_file, hand_dir, patho_dir, combined_dir, output_dir, args):
  # pylint: disable=too-many-locals
  patch_size = args.size
  patch_step = int(args.size * args.step_size)
  min_hand = int(patch_size * patch_size * args.min_hand)
  max_hand = int(patch_size * patch_size * args.max_hand)
  min_patho = int(patch_size * patch_size * args.min_patho)
  max_patho = int(patch_size * patch_size * args.max_patho)

  image_filename = os.path.basename(image_file)
  image = imread(image_file)
  hand_file = get_hand_segmentation_for_image(image_file, hand_dir)
  hand_filename = os.path.basename(hand_file)
  hand = imread(hand_file)
  patho_file = get_patho_segmentation_for_image(image_file, patho_dir)
  patho = imread(patho_file)
  patho_filename = os.path.basename(patho_file)
  combined_file = get_combined_segmentation_for_image(image_file, combined_dir)
  combined = imread(combined_file)
  combined_filename = os.path.basename(combined_file)

  partial_patch_count = 0
  non_hand_patch_count = 0
  non_patho_patch_count = 0
  found_patch_count = 0

  for i in range(0, image.shape[0], patch_step):
    for j in range(0, image.shape[1], patch_step):
      hand_patch = hand[i:i+patch_size, j:j+patch_size]
      if hand_patch.shape != (patch_size, patch_size):
        # ignore partial patches
        partial_patch_count += 1
        continue

      if np.count_nonzero(hand_patch) < min_hand or np.count_nonzero(hand_patch) > max_hand:
        # ignore patches that have too few/much hand
        non_hand_patch_count += 1
        continue

      patho_patch = patho[i:i+patch_size, j:j+patch_size]
      if np.count_nonzero(patho_patch) < min_patho or np.count_nonzero(patho_patch) > max_patho:
        # ignore patches that have too few/much patho
        non_patho_patch_count += 1
        continue

      # save patches
      image_patch = image[i:i+patch_size, j:j+patch_size]
      combined_patch = combined[i:i+patch_size, j:j+patch_size]
      imsave("{}/{}/{}".format(output_dir, image_subdir, add_patch_to_file_name(image_filename, found_patch_count)), image_patch)
      imsave("{}/{}/{}".format(output_dir, hand_subdir, add_patch_to_file_name(hand_filename, found_patch_count)), hand_patch)
      imsave("{}/{}/{}".format(output_dir, patho_subdir, add_patch_to_file_name(patho_filename, found_patch_count)), patho_patch)
      imsave("{}/{}/{}".format(output_dir, combined_subdir, add_patch_to_file_name(combined_filename, found_patch_count)), combined_patch)
      found_patch_count += 1

  tf.logging.info("Found {} patches and ignored {}/{}/{} (bad-hand/bad-patho/partial) in file '{}'".format(
    found_patch_count, non_hand_patch_count, non_patho_patch_count, partial_patch_count, image_filename))

  return found_patch_count

def main():
  # handle arguments and config
  args = parse_arguments()
  tf.logging.info("Args: {}".format(args))
  data_dir = os.path.join("data", args.data_dir)
  image_dir = os.path.join(data_dir, image_subdir)
  hand_dir = os.path.join(data_dir, hand_subdir)
  patho_dir = os.path.join(data_dir, patho_subdir)
  combined_dir = os.path.join(data_dir, combined_subdir)

  if args.verify:
    verify_data_integrity(image_dir, hand_dir, patho_dir, combined_dir)

  for sub_dir in data_subdirs:
    assert not os.path.exists(os.path.join(args.output_dir, sub_dir)), \
        "Output directory '{}' exists already, select another!".format(args.output_dir)
    os.makedirs(os.path.join(args.output_dir, sub_dir))

  found_patch_count = 0
  processed_image_count = 0

  for image_file in glob("{}/*{}.png".format(image_dir, args.match_pattern + "*" if args.match_pattern else "")):
    found_patch_count += find_patches_in_file(image_file, hand_dir, patho_dir, combined_dir, args.output_dir, args)
    processed_image_count += 1

  tf.logging.info("Found {} patches in {} images".format(found_patch_count, processed_image_count))

if __name__ == "__main__":
  START_TIME = time.time()
  tf.logging.set_verbosity(tf.logging.INFO)
  try:
    main()
  except Exception as ex:
    tf.logging.fatal("Exception occurred: {}".format(traceback.format_exc()))
  finally:
    tf.logging.info("Finished execution after {:.1f}m".format((time.time() - START_TIME) / 60))
