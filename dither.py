#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import logging
import sys
from collections import defaultdict
import argparse
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps
import PIL
from math import ceil, sqrt, floor, cos, sin, pi
import pdb
import os
import cv2

# Refactored
from dither_datastructure import AlphabetHolder
from parameters import (
    Params,
    DEFAULTS,
    make_argument_parser,
    check_arguments,
    setup_dithering_parameters,
)


def _truncate_or_repeat_text(source: str, targetLength: int) -> str:
    """Truncates or repeats a string to satisfy target length."""

    if len(source) < targetLength:
        numReps = int(targetLength / ceil(len(source))) + 1
        return (source * numReps)[:targetLength]
    else:
        return source[:targetLength]


def _render_function_cut_with_ratio_top_left(
    image: np.ndarray, ratio: float
) -> np.ndarray:
    """Given a ratio it cuts the image to the top left corner scaling sides proportionally.
    This function is for more efficient rendering purposes."""

    assert ratio > 0.0, "Ratio must be positive! (2518)"

    h, w = image.shape
    newH = int(h * ratio)
    newW = int(w * ratio)

    return image[:newH, :newW]


def _render_function_cut_with_ratio_centered(
    image: np.ndarray, ratio: float
) -> np.ndarray:
    """Given a ratio it cuts the image to the center scaling sides proportionally.
    This function is for more efficient rendering purposes."""

    assert ratio > 0.0, "Ratio must be positive! (2519)"

    h, w = image.shape
    newH = int(h * ratio / 2)
    newW = int(w * ratio / 2)
    cW = int(w / 2)
    cH = int(h / 2)

    return image[cH - newH : cH + newH, cW - newW : cW + newW]


class ArrayDither(Params):
    """Abstraction of the raw parameters, implements all dithering functionality for ndarrays.
    Actual dithering though needs at least the ImageDither class as it needs to resize arrays using PIL."""

    def calculateAlpabet(self, text: str):
        """Builds a set of all different letters used for dithering."""

        self._alphabet = set(list(text))

    # STARTFOLD ##### CORE DITHERING FUNCTIONALITY
    # ----- A big part of the dithering functionality is being refactored to the AlphabetHolder class -----

    def _make_padded_square_letter_box_from_character(self, letter: str) -> np.ndarray:
        """Produces an ndarray of shape [100,100] with a letter inside aligned to the top left corner."""

        img = Image.new("L", (DEFAULTS["supersamplingSize"],) * 2, 255)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), letter, (0), font=self._font)
        draw = ImageDraw.Draw(img)
        ar = np.array(img)
        return ar

    def _reverse_pad_single_square_letter_box(
        self, squareLetterBox: np.ndarray
    ) -> np.ndarray:
        """Cuts whitespace out of a letter box, as defined in '_getSquareLetterBoxForLetter'"""

        x, y = np.where(np.logical_not(squareLetterBox == 255))
        l, t, r, b = (min(x), min(y), max(x), max(y))
        return squareLetterBox[l : r + 1, t : b + 1]

    def _build_letters_dictionary_from_character_alphabet(self):
        """Builds a defaultdict containing all letterboxes for lettes in the alphabet, all sized at maximum height."""

        self._bbox = (
            *((DEFAULTS["supersamplingSize"],) * 2),  # (100, 100)
            0,
            0,
        )  # Init with max characters
        boxes = {}

        for c in self._alphabet:
            ar = self._make_padded_square_letter_box_from_character(c)
            x, y = np.where(np.logical_not(ar == 255))
            if len(x) > 0:  # If we have a visible character
                boxes.update({c: ar})
                l, t, r, b = (min(x), min(y), max(x), max(y))

                self._bbox = (
                    min((l, self._bbox[0])),
                    min((t, self._bbox[1])),
                    max((r, self._bbox[2])),
                    max((b, self._bbox[3])),
                )

        l, t, r, b = self._bbox

        self._letters = {c: ar[l : r + 1, t : b + 1] for c, ar in boxes.items()}

        # Now resize for the correct pixelHeight

        newDim = (list(self._letters.values()))[0]

        # We don't resample here anymore since we want to do that later on
        # when having supersampled the single characters.

        # fac = self.pixelHeight / newDim[1]
        # newDim = (int(fac * newDim[0]), int(fac * newDim[1]))
        # self._letters = {
        #     c: np.array(Image.fromarray(ar).resize(newDim, resample=PIL.Image.LANCZOS))
        #     for c, ar in self._letters.items()
        # }

        ddict = defaultdict(
            lambda: np.full(shape=newDim, fill_value=255, dtype=np.uint8)
        )

        ddict.update(self._letters)
        self._letters = ddict

    def _add_datastructure_for_letters_dictionary(self):
        """Builds a defaultdict containing all the letters in PixelHolder datastructure, ready for dithering."""

        minHeight = int((list(self._letters.values()))[0].shape[1] * self.minPixelRatio)
        d = {}

        for k, v in self._letters.items():
            holder = PixelHolder(v, minHeight, self.pixelHeight)
            d.update({k: holder})

        shape = list(d.values())[0](255).shape[:]

        ddict = defaultdict(
            lambda: (
                lambda x: np.full(
                    shape=shape,
                    fill_value=255,
                    dtype=np.uint8,
                )
            )
        )

        ddict.update(d)
        self._letters = ddict

        logging.info(
            f"Sampled down to 'color' palette with {next(iter(self._letters.values())).shades} shades"
        )

    def _dither_row_from_text_and_brightness_values(
        self, rowText: str, rowPixels: np.ndarray
    ) -> np.ndarray:
        """Dithers a row given enough text and brightness values, returns the concatenated row."""

        row = []

        for pix, letter in zip(rowPixels, rowText):
            # Size the pixel correctly before appending
            row.append(self._letters[letter](pix))
            # row.append(self._sizeAndCenterLetterBox(self._letters[letter], pix))

        return np.concatenate(row, axis=1)

    # ENDFOLD

    def padImage(self, unpadded: np.ndarray) -> np.ndarray:
        """Pads a given dithered image according to the specified percentage."""

        h, w = unpadded.shape
        padH, padW = 0, 0

        sq_2 = self.final_aspect_ratio

        logging.debug(
            f"Padding image sized: {w}x{h} (wxh) with {self.padding:.2%} padding."
        )

        if w > h:
            logging.debug("Detected landscape picture")
            # Landscape mode
            if w > sq_2 * h:
                # panoramic
                # pad the longer edge (w)
                logging.debug("Padding panoramic (longer edge)")
                padW = int(self.padding * (w / 2))
                totalWidth = w + 2 * padW
                totalHeight = int(totalWidth / sq_2)
                padH = int((totalHeight - h) / 2)
            else:
                # pad the shorter edge (h)
                logging.debug("Padding non-panoramic (shorter edge)")
                padH = int(self.padding * (h / 2))
                totalHeight = h + 2 * padH
                totalWidth = int(sq_2 * totalHeight)
                padW = int((totalWidth - w) / 2)
        else:
            logging.debug("Detected portrait picture")
            # Portrait mode
            if h > sq_2 * w:
                # panoramic
                # pad the longer edge (h)
                logging.debug("Padding panoramic (longer edge)")
                padH = int(self.padding * (h / 2))
                totalHeight = h + 2 * padH
                totalWidth = int(totalHeight / sq_2)
                padW = int((totalWidth - w) / 2)
            else:
                logging.debug("Padding non-panoramic (shorter edge)")
                padW = int(self.padding * (w / 2))
                totalWidth = w + 2 * padW
                totalHeight = int(totalWidth * sq_2)
                padH = int((totalHeight - h) / 2)

        padded = np.pad(
            unpadded,
            ((padH,) * 2, (padW,) * 2),
            constant_values=255,
        )

        logging.debug(
            f"Padded with {padW}x{padH} (wxh) padding only, size after padding: {padded.shape[1]}x{padded.shape[0]} (wxh)"
        )

        return padded

    def initAlphabetHolder(self, text: str):
        """Initializes the AlphabetHolder Datastructure."""

        # Load the font here
        # self._font = ImageFont.truetype(
        #     font=self.fontName, size=int((72 / 200) * DEFAULTS["supersamplingSize"])
        # )

        self._letters = AlphabetHolder(
            text,
            self._font,
            self.pixelHeight,
            self.minPixelRatio,
            self.thicknessRange[1],
        )

    def __call__(self, img: np.ndarray, text: str) -> np.ndarray:
        # The Call function should be as pure as possible.
        assert len(img.shape) == 2, "ArrayDither only supports b/w images"
        self.__w, self.__h = img.shape[::-1]
        assert (self.__w * (self.__h - 1)) <= len(text) <= (self.__w * self.__h), (
            "ArrayDither expects img to be correctly sized: "
            + f"w, h {self.__w, self.__h} --> nPix = {self.__w * self.__h} "
            + f"numChars = {len(text)}"
        )

        # Refactored to AlphabetHolder
        # self.calculateAlpabet(text)
        # self._calculateAndExtractBBoxesForAlphabet()
        # self._prepareLetterDictForSizes()

        self.initAlphabetHolder(text)

        rows = []
        logging.info(f"Image downsampled to: {img.shape[1]}x{img.shape[0]} (WxH)")
        rowLength = img.shape[1]

        for i in range(img.shape[0] - 1):
            rows.append(
                self._dither_row_from_text_and_brightness_values(
                    text[i * rowLength : (i + 1) * rowLength], img[i, :]
                )
            )

        rows[-1] = np.pad(
            rows[-1], (0, rows[0].shape[1] - rows[-1].shape[1]), constant_values=255
        )

        unpadded = np.concatenate(rows, axis=0)

        return self.padImage(unpadded) if self.padding > 0.0 else unpadded


class ImagePreprocessing:
    """This class handles all image preprocessing.

    It expects to be given a numpy array (2 dimensional) and returns one of the same
    shape but processed according to the parameters defined here."""

    def __init__(
        self,
        globally_equalize_histogram: bool = False,
        local_equalization_tile_size_percent: float = 0.03,
    ):
        self.globally_equalize_histogram = globally_equalize_histogram
        self.local_equalization_tile_size_percent = local_equalization_tile_size_percent

    def __call__(self, image: PIL.Image):
        ar = np.array(image)

        square_width = sqrt(ar.shape[0] * ar.shape[1])
        tile_size = square_width * self.local_equalization_tile_size_percent
        tile_size = int(tile_size)

        logging.info("CLAHE algorithm uses {} tile size in pixels".format(tile_size))

        clahe = cv2.createCLAHE(tileGridSize=(tile_size, tile_size))

        ar = clahe.apply(ar)

        if self.globally_equalize_histogram:
            ar = cv2.equalizeHist(ar)
        return ar


class ImageDither(ArrayDither):
    """Abstraction of ArrayDither, but implements all dithering relevant functionality.
    When called via given text, and a PIL.Image it produces another PIL.Image
    containing the dithered content.

    Note that this class is tightly coupled with ArrayDither, since it implements
    all the resizing functionality such that ArrayDither doesn't need to use PIL."""

    def __call__(
        self, img: Image.Image, text: str, preprocessor: ImagePreprocessing = None
    ) -> Image.Image:
        newText = text[:]

        if self.uppercase:
            newText = text.upper()

        # super().calculateAlpabet(newText)

        # The font size in points
        fontPts = int((72 / 100) * DEFAULTS["supersamplingSize"])

        # Set the font now
        try:
            self._font = ImageFont.truetype(self.fonts_list[0], fontPts)

            # self._font = [ImageFont.truetype(p) for p in self.fontName]

        except Exception:
            # Fall back to the default font if the given font was not found
            logging.warning(
                f'Couldn\'t load "{self.fonts_list}" from {DEFAULTS["fontsDirectory"]}, falling back to default font'
            )

            try:
                self._font = ImageFont.truetype(DEFAULT_FONT, fontPts)
            except:
                logging.critical(
                    "ERROR: Failed to load default font: "
                    + DEFAULT_FONT
                    + " aborting script now!"
                )
                exit(1)

        # super()._calculateAndExtractBBoxesForAlphabet()
        super().initAlphabetHolder(text)

        ratio = (list(self._letters.values())[0](0)).shape
        ratio = ratio[0] / ratio[1]
        ratio /= float(img.height / img.width)
        logging.debug(f"Skew transformation calculated ratio: {ratio}")
        # Now resize the image for the pixel ratio

        newImg = img.resize(
            (int(img.height * ratio), img.height), resample=PIL.Image.LANCZOS
        )

        logging.info(
            f"Image Skew Transform: {img.width}x{img.height} (original), {newImg.width}x{newImg.height} (skewed)"
        )

        fac = sqrt(len(newText) / (newImg.width * newImg.height))
        nW = int(newImg.width * fac)
        nH = int(ceil(newImg.height * fac))
        ar = np.array(newImg.resize((nW, nH)).convert("L"))

        if self.minPixelHeight is not None:
            # Now the image is skewed, so we can calculate the correct pixel ratio
            # expects A0 image size

            # Here should the final image height in millimeters go (841 is for A0)
            maxPixelHeight = self.paperHeight / ar.shape[0]

            if self.minPixelHeight > maxPixelHeight:
                logging.exception(
                    f"Min pixel height was bigger than maxPixelHeight: {self.minPixelHeight: .2f}mm (min) and {maxPixelHeight: .2f}mm (max)"
                )
                exit(1)

            self.minPixelRatio = float(self.minPixelHeight / maxPixelHeight)

        if preprocessor is not None:
            ar = preprocessor(ar)

        # Image.fromarray(ar).show()
        d = super().__call__(ar, newText[: int(nH * nW)])

        # Manual post processing for rendering purposes.

        # Manually pad for rendering also.
        # d = _padEqually(d, 0.05)

        # Cut the image here by 'hand' for rendering.
        # d = _cutForRatioTopLeft(d, 1.0 / 8.0)
        # d = _cutForRatioCenter(d, 1.0 / 8.0)
        return Image.fromarray(d)


class FileDither(ImageDither):
    """Abstraction of ImageDither, implements file based dithering functionality.
    Also callable with filenames the same content as ImageDither.

    Handles all IO bound exceptions, logs but also throws them when critical."""

    def __call__(
        self,
        inFile: str,
        outFile: str,
        text: str,
        preprocessor: ImagePreprocessing = None,
    ):

        try:
            img = Image.open(inFile)
        except Exception as e:
            logging.exception(f"Error reading image input file: {e}")
            exit(1)

        try:
            d = super().__call__(img, text, preprocessor)
        except Exception as e:
            logging.exception(f"Uncaught exception occurred in ArrayDither: {str(e)}")
            exit(1)

        try:
            d.save(outFile)
        except Exception as e:
            logging.exception(
                f"Error writing dithered image to file at path: {outFile}, error: {e}"
            )
            exit(1)

        # Now calculte the actual height of a character
        height_ratio = self.pixelHeight / d.height
        char_height = height_ratio * self.paperHeight  # in mm

        logging.info(f"Dithered image dimensions: {d.width}x{d.height}")
        logging.info(
            f"char to image height ratio: {height_ratio}, with the specified paperHeight of {self.paperHeight: .2f}mm that gives a maximum of {char_height: .2f}mm character height, and a minimum of {char_height * self.minPixelRatio: .2f}"
        )


def main():
    """Main function, needs arguments to be refactored..."""

    parser = make_argument_parser()

    args = parser.parse_args()

    loglevel = logging.WARNING

    if args.verbose:
        loglevel = logging.DEBUG
    elif args.quiet:
        loglevel = logging.ERROR

    loggingFormat = "%(asctime)s %(levelname)s: %(message)s"

    if args.logfile != "stdout":
        logging.basicConfig(level=loglevel, filename=args.logfile, format=loggingFormat)
    else:
        logging.basicConfig(level=loglevel, format=loggingFormat)

    # And log the actual script call
    logging.debug("dither.py script called with: " + str(sys.argv))

    try:
        check_arguments(args)
    except AssertionError as err:
        logging.exception("Argument error occurred: " + str(err))
        exit(1)

    # Now we can dither
    textfilter_call = (
        "python ./textfilter.py -in {} -out /tmp/filtered_text.txt -v".format(
            args.in_text
        )
    )

    try:
        rval = os.system(textfilter_call)

        if rval != 0:
            logging.error("Error during text filtering occurred: {}".format(rval))

    except Exception as e:
        logging.error(
            "Error during calling the text filter occurred: {}".format(str(e))
        )

    with open("/tmp/filtered_text.txt", "r") as f:
        txt = f.read()  # [:-1]

    logging.debug(
        "Read {} characters from filtered file in tmp directory".format(len(txt))
    )

    if args.truncate_text > -1:
        if len(txt) < 64:
            logging.debug(f"Using short text: '{txt}'")
        txt = _truncate_or_repeat_text(txt, args.truncate_text)

        logging.info(
            f"Trunc or Repeat text to {args.truncate_text} characters, removed {(len(txt) - args.truncate_text) / len(txt): .2%}"
        )

    kwargs = setup_dithering_parameters(args)

    logging.debug("Dithering parameters used:")

    for k, v in kwargs.items():
        logging.debug(f'\t"{k}" = {v}')

    preprocessor = ImagePreprocessing()

    p = FileDither(**kwargs)

    logging.info(
        "dithering "
        + args.input_image
        + " using text: "
        + args.in_text
        + " (with "
        + str(len(txt))
        + ") characters."
    )

    try:
        p(args.input_image, args.out_filepath, txt, preprocessor)
    except Exception as e:
        logging.exception(f"Uncaught error occurred during dithering: {e}")
        exit(1)

    exit(0)  # Success should return code 0


if __name__ == "__main__":
    main()
    quit()
    fonts = [
        ImageFont.truetype(
            "Catamaran-VariableFont_wght.ttf",
            # "~/Downloads/Roboto_Mono/RobotoMono-VariableFont_wght.ttf",
            int((72 / 100) * DEFAULTS["supersamplingSize"]),
        )
    ]
    fonts[0].set_variation_by_axes([100])  # This sets weight of variable font
    text = "the quick brown fox jumps over the lazy dog."
    holder = AlphabetHolder(text.upper(), fonts=fonts, max_thickness=30)

    for n in range(32):
        v = int(n * (255 / 31))
        ar = holder["D"](v)
        pImg = Image.fromarray(ar)
        pImg.show()
