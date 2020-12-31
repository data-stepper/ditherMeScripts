from dataclasses import dataclass
import os
import argparse
import logging
from math import sqrt

DEFAULTS = {
    "fontName": "Menlo-Bold",
    "fontsDirectory": "",  # Empty since fonts are global on macos
    "pixelRatio": 0.6,
    "pixelHeight": 28,
    "thicknessRange": (1, 8),
    "minPixelHeight": None,
    "paperHeight": 841.0,
    "uppercase": False,
    "supersamplingSize": 256,
    "padding": 0.05,  # 5 % padding of the shorter edge per default
}


@dataclass
class _Params:
    pixelHeight: int = DEFAULTS["pixelHeight"]
    thicknessRange: tuple = DEFAULTS["thicknessRange"]
    minPixelRatio: float = DEFAULTS["pixelRatio"]
    minPixelHeight: float = DEFAULTS["minPixelHeight"]
    paperHeight: float = DEFAULTS["paperHeight"]
    uppercase: bool = DEFAULTS["uppercase"]
    padding: float = DEFAULTS["padding"]
    fonts_list: str = DEFAULTS["fontName"]
    final_aspect_ratio: float = sqrt(2)


class Params(_Params):
    """Dataclass storing parameters relevant for dithering."""

    def __getFontName(self):
        return self.__fn

    def __setFontName(self, new_fonts_list: list):

        if len(new_fonts_list) > 0:
            fonts_passed = []

            for font_path in new_fonts_list:
                if font_path.endswith(".ttf") or font_path.endswith(".ttc"):
                    fonts_passed.append(font_path)

                else:
                    logging.warning(
                        f"Font {font_path} didn't end with .ttf or .ttc, not adding it to the used fonts list."
                    )

            if len(fonts_passed) > 0:
                logging.info(
                    f"{len(fonts_passed)} fonts accepted, using the following {fonts_passed}"
                )
                self.__fn = fonts_passed

            else:
                logging.error(
                    f"No font was accepted, aborting now. fonts given were: {new_fonts_list}"
                )
                raise ValueError(f"No font accepted.")

        else:
            raise ValueError("Attempt was made to set font to an empty list.")

    fonts_list = property(fset=__setFontName, fget=__getFontName)


def make_argument_parser():
    parser = argparse.ArgumentParser(
        description="Script for dithering images given a text and an image"
    )

    # STARTFOLD ##### Add all command line arguments supported.

    parser.add_argument(
        "-img",
        "--input-image",
        metavar="<image-input>",
        type=str,
        help="Path to the input image, can be one of the following formats: PNG, JPED, TIFF",
        required=True,
    )

    parser.add_argument(
        "-txt",
        "--in-text",
        metavar="<text-input>",
        type=str,
        help="Path to the input text file, must be a .txt file encoded with utf-8",
        required=True,
    )

    parser.add_argument(
        "-out",
        "--out-filepath",
        metavar="<output-filepath>",
        type=str,
        help="Path where the output file will be written to, must end with '.png', if ommitted it will simply append something to the input image path.",
        required=False,
    )

    parser.add_argument(
        "--pixel-height",
        metavar="<height of character in pixels>",
        type=int,
        help="The height in pixels for each character, will determine final image size (varying this value can easily lead to very large images).",
        default=DEFAULTS["pixelHeight"],
    )

    parser.add_argument(
        "--uppercase",
        help="Whether all letters dithered should be uppercased or not",
        action="store_true",
    )

    parser.add_argument(
        "--font",
        metavar="<one or multiple names/paths to truetype fonts>",
        dest="fontName",
        type=str,
        nargs="+",
        help="Either one or more names or absolute paths to truetype fonts used for dithering.",
        default=[DEFAULTS["fontName"]],
    )

    paperGroup = parser.add_mutually_exclusive_group()

    def paperFormatter(f: str) -> float:
        if f == "A0":
            return 841.0

        if f == "A1":
            return 594.67

        if f == "A2":
            return 420.0

        raise argparse.ArgumentError("Format " + f + " not valid.")

    paperGroup.add_argument(
        "--paper-format",
        dest="paperHeight",
        type=paperFormatter,
        default="A0",  # --------------- Fix this
        # Paper height should be defaultable in DEFAULTS
        # not setting a default value for paper format (like here)
        # results in paperHeight being None
        help="You can specify this instead of '--paper-height' when using one of the supported formats.",
    )

    paperGroup.add_argument(
        "--paper-height",
        type=float,
        metavar="<paper height in mm>",
        dest="paperHeight",
        help="The height of the paper given in millimeters, relevant for calculting final character height or when using '--min-pixel-height' to specify the pixel ratio",
        default=DEFAULTS["paperHeight"],
    )

    parser.add_argument(
        "--num-chars",
        metavar="<Number of characters where to truncate or repeat text>",
        dest="truncate_text",
        type=int,
        help="Whether and where to truncate or repeat the given text to yield a good amount of characters, -1 means don't truncate",
        default=-1,
    )

    pixelGroup = parser.add_mutually_exclusive_group()

    pixelGroup.add_argument(
        "--pixel-ratio",
        metavar="<ratio smallest pixel height to largest>",
        type=float,
        help=f"A float between 0 and 1. The smallert this ratio is the more shades (and quality) the final image will have, but beware that it has an impact to the absolute size of the characters in the final image, possibly yielding unreadable parts of the image (where the original image was very bright). (defaults to {DEFAULTS['pixelRatio']})",
        default=DEFAULTS["pixelRatio"],
    )

    pixelGroup.add_argument(
        "--min-pixel-height",
        dest="minPixelHeight",
        metavar="<height in mm of smallest pixel>",
        type=float,
        help="The final height in millimeters for the smallest pixel's height in the dithered image",
        default=DEFAULTS["minPixelHeight"],
    )

    # Parse logging stuff

    parser.add_argument(
        "--logfile",
        metavar="<path to logfile>",
        type=str,
        help="The path to the logfile (defaults to stdout)",
        default="stdout",
    )

    # Define different loglevels
    mgroup = parser.add_mutually_exclusive_group()
    mgroup.add_argument("-v", "--verbose", help="Verbose logging", action="store_true")
    mgroup.add_argument(
        "-q", "--quiet", help="Don't output anything", action="store_true"
    )

    parser.add_argument(
        "--padding",
        "-p",
        metavar="<percentage to pad the shorter edge>",
        type=float,
        help="Specify padding as a floating point number, is expected to be a percentage value",
        dest="padding",
        default=DEFAULTS["padding"] * 100,
    )

    # Thickness values
    parser.add_argument(
        "--min-thickness",
        dest="min_thickness",
        metavar="<minimum thickness value>",
        type=int,
        help=f"Minimum thickness value for variable thickness font. (defaults to {DEFAULTS['thicknessRange'][0]}",
        default=DEFAULTS["thicknessRange"][0],
    )

    parser.add_argument(
        "--max-thickness",
        dest="max_thickness",
        metavar="<maximum thickness value>",
        type=int,
        help=f"Maximum thickness value for variable thickness font. (defaults to {DEFAULTS['thicknessRange'][1]}",
        default=DEFAULTS["thicknessRange"][1],
    )

    parser.add_argument(
        "--aspect-ratio",
        dest="aspect_ratio",
        metavar="<aspect ratio of final picture>",
        type=float,
        help="Aspect ratio of final output image, needed for padding correctly, always needs to be a positive number greatther than or equal to 1. Defaults to sqrt(2) for (DIN AX)",
        default=sqrt(2),
    )

    return parser


def _exchange_suffix_in_path(path: str, newSuffix: str) -> str:
    """Exchanges the suffix in a path for a new one, newSuffix must start with a dot."""

    assert "." in newSuffix, "new Suffix needs to start with a dot"
    dotIndex = path.rfind(".")
    return path[:dotIndex] + newSuffix


def check_arguments(args):
    assert args.in_text.endswith(
        ".txt"
    ), f'The given text input path: "{args.in_text}" doesn\'t end with .txt'

    imgPath = args.input_image
    outPath = args.out_filepath

    assert (
        imgPath.endswith(".jpg")
        or imgPath.endswith(".png")
        or imgPath.endswith(".jpeg")
        or imgPath.endswith(".tiff")
    ), f'Invalid filepath for image given, only supports .jpg, .png, .jpeg, .tiff path given: "{imgPath}"'

    if outPath:
        if not outPath.endswith(".png"):
            logging.info("Appending .png to the output path as it was not given")
            outPath += ".png"
    else:
        outPath = _exchange_suffix_in_path(imgPath, "_dithered.png")
        logging.info(f"No output path given, saving to {outPath}")

    assert (
        0 < args.pixel_height < 100
    ), f"Given pixelHeight: {args.pixel_height} was not in range (5 to 49)"

    assert (
        0.0 < args.pixel_ratio < 1.0
    ), f"Pixel ratio must strictly be between 0 and 1: {args.pixel_ratio} was given!"

    assert args.paperHeight > 0.0, "Negative paperHeight not allowed!"

    if args.uppercase:
        logging.info("Uppercasing all letters")

    assert 100.0 > args.padding >= 0.0, "Padding value was not in range (0-100%)"

    assert (
        args.max_thickness > args.min_thickness
    ), "Maximum thickness value must be greater than minimum."

    assert args.aspect_ratio >= 1.0, "Aspect ratio must be strictly positive."


def setup_dithering_parameters(parsed_args) -> dict:
    kwargs = {
        "pixelHeight": parsed_args.pixel_height,
        "uppercase": parsed_args.uppercase,
        "fonts_list": parsed_args.fontName,
        "minPixelRatio": parsed_args.pixel_ratio,
        "minPixelHeight": parsed_args.minPixelHeight,
        "paperHeight": parsed_args.paperHeight,
        "padding": parsed_args.padding / 100.0,  # percentage value given here
        "thicknessRange": (parsed_args.min_thickness, parsed_args.max_thickness),
        "final_aspect_ratio": parsed_args.aspect_ratio,
    }
    return kwargs
