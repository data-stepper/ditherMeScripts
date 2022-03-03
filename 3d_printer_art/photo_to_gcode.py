#!/usr/bin/python3
# Optional shebang interpreter line (if script shall be executable)


"""
    This script can transform an image into a gcode file which in turn
    can be carved by a 3d printer into a sheet of transparent plastic.
"""

import argparse
import logging
from pathlib import Path
from pprint import pformat
from PIL import Image
import numpy as np


def read_image_to_pil(path):

    with Image.open(path, "r") as img:
        img.load()

    return img


def dots_to_visit_from_ar_image(ar_image):
    indices = np.where(ar_image)
    print(indices)


def main(args):

    logging.info(pformat(args))
    pil_img = read_image_to_pil(args.image_input)
    h, w = pil_img.size

    assert h == w, "Only square images supported"

    downsample_factor = 0.2
    new_w, new_h = int(w * downsample_factor), int(h * downsample_factor)
    # pil_img = pil_img.resize((new_h, new_w))
    pil_img = pil_img.resize((new_h, new_w), resample=Image.LANCZOS)

    dithered = pil_img.convert(mode="1", dither=Image.FLOYDSTEINBERG)

    # dithered.save("/home/bent/git/.for_my_xing_xing/dithered.png")

    ar = np.array(dithered, dtype=np.uint8)
    dots = dots_to_visit_from_ar_image(ar)


def setup_logging(args):

    FORMAT = "%(asctime)s %(levelname)s: %(message)s"

    if args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if args.logfile is not None:

        # Log to file
        logging.basicConfig(filename=args.logfile, level=level, format=FORMAT)

    else:

        logging.basicConfig(level=level, format=FORMAT)


def parse_args():

    parser = argparse.ArgumentParser(description=__doc__)

    ## Notice:
    # Optional arguments start with a '-', positionals do not

    # Verbose logging
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Lowest loglevel",
        default=False,
    )

    # Logfile argument
    parser.add_argument(
        "--logfile",
        type=str,
        dest="logfile",
        default=None,
        metavar="<filepath>",
        help="Write log output to a file",
    )

    # Set some arguments for the script to be configurable
    parser.add_argument(
        "--temperature",
        type=float,
        default=190,
        help="The temperature to which to heat the printer head.",
    )

    parser.add_argument(
        "--num-max-dots",
        type=int,
        default=500,
        help="The maximum number of dots to carve into the plastic.",
    )

    parser.add_argument(
        "--mm-depth-into-plastic",
        type=float,
        default=0.5,
        help="The amount of mm the head will carve the dots into the plastic.",
    )

    parser.add_argument("image_input", type=str, help="Image to process")
    parser.add_argument(
        "-o", "--output-path", type=str, default=None, help="Image to process"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args)
    main(args)
