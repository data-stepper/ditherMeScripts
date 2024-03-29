#!/usr/bin/python3
# Optional shebang interpreter line (if script shall be executable)


"""
    This script can transform an image into a gcode file which in turn
    can be carved by a 3d printer into a sheet of transparent plastic.
"""

import argparse
import logging
from pathlib import Path
from pprint import pformat, pprint
from PIL import Image, ImageOps
import sys
import numpy as np


def read_image_to_pil(path):

    with Image.open(path, "r") as img:
        img.load()

    return img


def normalized_coordinates_to_visit_from_ar_image(
    ar_image,
    image_side_length: int,
    target_side_length_mm: float,
    border_mm: float,
):
    x, y = np.where(ar_image)
    indices = np.stack((x, y), axis=-1)
    indices = np.array(indices, dtype=np.float64)
    indices /= image_side_length - 1  # Zero indexed always

    # Now transform the coordinates appropriately
    effective_side_length = target_side_length_mm - (2 * border_mm)
    indices *= effective_side_length
    indices += border_mm

    globals()["effective_side_length"] = effective_side_length

    # indices += image_side_length

    return indices


def gcode_carve_one_dot(x, y, z_mm_above, z_mm_bottom):
    g = ""

    # Move above the dot
    g += f"G0 X{x:.1f} Y{y:.1f} Z{z_mm_above:.1f} ;\n"

    # Move to the dot
    g += f"G0 X{x:.1f} Y{y:.1f} Z{z_mm_bottom:.1f} ;\n"

    # Wait 1/2 s at the dot to melt it a little
    g += f"G4 P500 ;\n"

    # Move back above the dot
    g += f"G0 X{x:.1f} Y{y:.1f} Z{z_mm_above:.1f} ;\n"

    return g


def gcodes_from_coordinates(coordinates, most_bottom_mm: float) -> str:

    full_gcode_str = ""

    # Dont really auto home since it would touch the plastic
    full_gcode_str += "G28 ;\n"
    full_gcode_str += "G0 X0 Y0 Z5 ;\n"

    # Set the speed to very high
    full_gcode_str += "G0 F3000 ; \n"

    # Heat the hotend and wait until heated
    full_gcode_str += "M109 S180 \n"

    z_mm_above = 8.0 + (1.0 - most_bottom_mm)

    # Parse the coordinates into a list first
    # And sort them for better efficiency
    coordinates = [(x, y) for (x, y) in coordinates]

    rows = []

    current_row = []
    last_x = coordinates[0][0]

    while len(coordinates) > 0:
        x, y = coordinates.pop()

        if last_x == x:
            current_row.append((x, y))

        else:
            rows.append(current_row)
            current_row = []

        last_x = x

    # Now reassemble the original coordinates

    coordinates = []
    pixels_in_row = []

    for i, row in enumerate(rows):
        if len(row) == 0:
            continue

        pixels_in_row.append(len(row))

        if i % 2 == 1:
            coordinates.extend(row)

        else:
            coordinates.extend(row[::-1])

    # Start at the bottom left
    coordinates = coordinates[::-1]

    pixels_in_row = np.array(pixels_in_row)
    logging.info(
        "Pixel distribution by row total {} rows: \n{}".format(
            pixels_in_row.shape, pformat(pixels_in_row)
        )
    )

    esl = globals()["effective_side_length"]

    logging.info(
        "This configuration would give a row height of {} mm".format(
            esl / pixels_in_row.shape[0]
        )
    )

    pixel_density = esl / pixels_in_row

    logging.info(
        "Each row height pixel density: \n{}".format(pformat(pixel_density))
    )

    logging.info(
        "Stats on the pixel density (in mm):\n {} mean {} median {} min {} max"
        " {} std".format(
            pixel_density.mean(),
            np.median(pixel_density),
            pixel_density.min(),
            pixel_density.max(),
            pixel_density.std(),
        )
    )

    for x, y in coordinates:
        full_gcode_str += gcode_carve_one_dot(
            x, y, z_mm_above, z_mm_bottom=most_bottom_mm
        )

    # Heat the hotend and wait until heated
    full_gcode_str += "M104 S0 \n"
    full_gcode_str += "G0 X300 Y300 Z20 ; \n"

    return full_gcode_str


def main(args):

    logging.info(pformat(args))
    pil_img = read_image_to_pil(args.image_input)

    # Don't invert
    # pil_img = ImageOps.invert(pil_img)
    h, w = pil_img.size

    assert h == w, "Only square images supported"

    # Use about 0.2 yields good quality
    downsample_factor = 0.4
    new_w, new_h = int(w * downsample_factor), int(h * downsample_factor)
    pil_img = pil_img.resize((new_h, new_w), resample=Image.LANCZOS)

    dithered = pil_img.convert(mode="1", dither=Image.FLOYDSTEINBERG)
    dithered.save("/home/bent/git/.for_my_xing_xing/dithered2.png")

    ar = np.asarray(dithered, dtype=np.uint8)

    logging.info("Carving total {} dots".format(ar.sum()))

    coordinates = normalized_coordinates_to_visit_from_ar_image(
        ar,
        image_side_length=new_w,
        target_side_length_mm=300.0,
        border_mm=50.0,
    )

    gcodes = gcodes_from_coordinates(
        coordinates, most_bottom_mm=args.mm_depth_into_plastic
    )

    with open("/home/bent/generated_gcode.gcode", "w") as f:
        f.write(gcodes)


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
        default=1.0,
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
