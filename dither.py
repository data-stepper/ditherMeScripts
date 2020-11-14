#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

# STARTFOLD ##### IMPORTS

import logging
import os
import sys
from dataclasses import dataclass
from collections import defaultdict, Counter, namedtuple
import argparse
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import PIL
from math import ceil, sqrt, floor, cos, sin, pi
import pdb

# ENDFOLD

# STARTFOLD ##### SUPPLEMETARY FUNCTIONS

# This is 'stolen' from textfilter.py so maybe consider refactoring here?
def _exchangeSuffixInPath(path: str, newSuffix: str) -> str:
    """Exchanges the suffix in a path for a new one, newSuffix must start with a dot."""

    assert "." in newSuffix, "new Suffix needs to start with a dot"
    dotIndex = path.rfind(".")
    return path[:dotIndex] + newSuffix


def _truncateOrRepeat(source: str, targetLength: int) -> str:
    """Truncates or repeats a string to satisfy target length."""

    if len(source) < targetLength:
        numReps = int(ceil(len(source) / targetLength))
        return (source * numReps)[:targetLength]
    else:
        return source[:targetLength]


# ENDFOLD

# STARTFOLD  ##### SCRIPT CONSTANTS

# The default font which  to fall back if an exception occurs.
# Make sure that this font (path) is always accessible.
DEFAULT_FONT = "Menlo-Bold.ttf"

# Script defaults (CLI and as module) should be defined here
DEFAULTS = {
    "fontName": "Menlo-Bold",
    "fontsDirectory": "",  # Empty since fonts are global on macos
    "pixelRatio": 0.6,
    "pixelHeight": 28,
    "minPixelHeight": None,
    "paperHeight": 841.0,
    "uppercase": False,
    "watermark": False,  # Still needs to be implemented.
    "supersamplingSize": 512,
    "padding": 0.05,  # 5 % padding of the shorter edge per default
}

# Reduces image always to exactly this number of shades
NUM_SHADES = 32

# ENDFOLD

# STARTFOLD ##### DITHER PARAMETERS


@dataclass
class _Params:
    pixelHeight: int = DEFAULTS["pixelHeight"]
    minPixelRatio: float = DEFAULTS["pixelRatio"]
    minPixelHeight: float = DEFAULTS["minPixelHeight"]
    paperHeight: float = DEFAULTS["paperHeight"]
    uppercase: bool = DEFAULTS["uppercase"]
    padding: float = DEFAULTS["padding"]
    # bold: bool = True  # Boldness will be implemented using Supersampling
    fontName: str = DEFAULTS["fontName"]
    watermark: bool = DEFAULTS["watermark"]


class Params(_Params):
    """Dataclass storing parameters relevant for dithering."""

    def __getFontName(self):
        return self.__fn

    def __setFontName(self, newFontName: str):
        # Now look up whether the font with the given name exists

        # Sort of deprecated, fonts will later be implemented using a specified font directory
        if newFontName.endswith(".ttf") or newFontName.endswith(".ttc"):
            if os.path.isfile(newFontName):
                self.__fn = newFontName

            else:
                logging.warning(
                    f"font {newFontName} not a file, falling back to default font"
                )

                self.__fn = DEFAULT_FONT

        self.__fn = newFontName

    fontName = property(fset=__setFontName, fget=__getFontName)


# ENDFOLD

# STARTFOLD ##### PIXEL HOLDER DATASTRUCTURE

# STARTFOLD ##### OLD PIXEL HOLDER
# class PixelHolder:
#     """Datastructure holding all different sizes for a single character.
#     Instances are callable with a brightness level and return the correct Pixel.

#     Initialized with the biggest pixel and a minimum height."""

#     def __init__(self, maxPixel: np.ndarray, minHeight: int, samplingHeight: int):
#         # For debug purposes implementing Supersampling

#         self.minHeight = minHeight
#         self.heightRange = maxPixel.shape[1] - self.minHeight
#         self.__sizedPixels = {
#             h: self._scaleCentered(maxPixel, h)
#             for h in range(self.minHeight, maxPixel.shape[1])
#         }

#         self.__sizedPixels.update({maxPixel.shape[1]: maxPixel})
#         self.shades = len(self.__sizedPixels.keys())

#         # Now downsample to the specified height
#         samplingRatio = maxPixel.shape[1]
#         sampled = {
#             k: self._resampleArray(v, samplingHeight / maxPixel.shape[1])
#             for k, v in self.__sizedPixels.items()
#         }

#         self.__sizedPixels = sampled

#     def __call__(self, value: int) -> np.ndarray:
#         # Accepts pixel value as input

#         ratio = 1.0 - (value / 255.0)
#         ratio = ratio * self.heightRange + self.minHeight

#         return self.__sizedPixels[ceil(ratio)]
# ENDFOLD


def clip(value, lower, upper):
    return lower if value < lower else upper if value > upper else value


# STARTFOLD ##### Array sampling


def _resampleArray(original: np.ndarray, ratio: float) -> np.ndarray:
    """Resamples an ndarray for a given ratio."""

    assert ratio > 0, "Negative ratio doesn't make any sense: " + str(ratio)

    w, h = original.shape
    newHeight = int(h * ratio)
    newWidth = int(w * ratio)
    resampled = np.array(
        Image.fromarray(original).resize(
            (newHeight, newWidth), resample=PIL.Image.LANCZOS
        )
    )

    return resampled


def _scaleCentered(pixel: np.ndarray, height: int) -> np.ndarray:
    """Scales a pixel down to a given height, leaving content centered."""

    scaled = _resampleArray(pixel, height / pixel.shape[1])

    padW = pixel.shape[0] - scaled.shape[0]
    padH = pixel.shape[1] - scaled.shape[1]
    scaled = np.pad(
        scaled,
        ((floor(padW / 2.0),) * 2, (floor(padH / 2.0),) * 2),
        constant_values=255,
    )

    scaled = np.pad(
        scaled,
        (
            (0, pixel.shape[0] - scaled.shape[0]),
            (0, pixel.shape[1] - scaled.shape[1]),
        ),
        constant_values=255,
    )

    return scaled


# ENDFOLD


class PixelHolder:
    """'Dummy' class that just holds some arrays and returns the correct one when called."""

    def __init__(self, arraysOfShades: list):
        self.arrays = arraysOfShades[:]
        self.numShades = len(self.arrays)

        # Resamples brightness values for number of shades
        def samplingFunc(brightness: int) -> int:
            return int((clip(brightness, 0.0, 255.0) / 255.0) * self.numShades)

        self.samplingFunc = samplingFunc

    def __call__(self, brightness: int) -> np.ndarray:
        return self.arrays[self.samplingFunc(brightness)]

    def resampleWithRatio(self, ratio: float):
        sampled = [_resampleArray(ar, ratio) for ar in self.arrays]

        self.arrays = sampled


def thicknessAndHeightForBrightness(brightness: float) -> (float, float):
    """Calculates the normalized thickness and character height for a given normalized brightness value."""

    return (
            cos(pi * ( brightness / 2.0)),
            cos(pi * ( brightness / 2.0)),
            )

    return (
        1.0 - cos(pi * (brightness - 0.25)),
        sin(pi * (brightness - 0.25)),
    )  # Hopefully gives a nice ellipse


class AlphabetHolder:
    """This class holds all the single 'pixels' with correct sizes and thicknesses.

    It is instantiated using a string of either a raw text or an already filtered alphabet.
    Items of it are PixelHolders and can be accessed like a dictionary:

        Example usage:

            a = AlphabetHolder('some text I want to dither.', final_height=12, absolute_height=1.6)
            letter_s = a['s']
            pix_brightest = letter_s(255)
            # Now you have the corresponding ndarray to append to the row."""

    # STARTFOLD ##### CORE DITHERING FUNCTIONALITY

    def _generateSquareLetterBox(self, letter: str, thickness: int) -> np.ndarray:
        """Produces an ndarray of supersamplingSize with a letter aligned to the top left corner,
        using the specified thickness value as stroke thickness."""

        img = Image.new("L", (DEFAULTS["supersamplingSize"],) * 2, 255)
        draw = ImageDraw.Draw(img)
        offset = int(DEFAULTS["supersamplingSize"] * 0.2)
        draw.text(
            (offset, offset),
            letter,
            (0),
            font=self._font,
            stroke_width=thickness,
        )
        draw = ImageDraw.Draw(img)
        ar = np.array(img)
        return ar

    def _removeWhitespaceFromLetters(self):
        """Computes bboxes for all letters with all thicknesses and removes all unnecessary whitespace.
        Places extracted letters back into self._lettes dictionary."""

        # Lazily make tuple of (SS, SS, 0, 0)
        bbox = (*((DEFAULTS["supersamplingSize"],) * 2), 0, 0)

        for c in self._alphabet:
            t = self._letters[c]

            for ar in t:
                # Iterate over the ndarrays with varying thicknesses
                x, y = np.where(np.logical_not(ar == 255))

                if len(x) > 0:  # If we have a visible character
                    l, t, r, b = (min(x), min(y), max(x), max(y))

                    bbox = (
                        min((l, bbox[0])),
                        min((t, bbox[1])),
                        max((r, bbox[2])),
                        max((b, bbox[3])),
                    )

        l, t, r, b = bbox

        logging.debug(f"Bounding box for letters computed: {l}, {t}, {r}, {b} (ltrb)")

        extracted = {}

        for c in self._alphabet:
            tmp = [v[l : r + 1, t : b + 1] for v in self._letters[c]]

            extracted[c] = tmp

        self._letters = extracted

    def _getLetterWithThickness(self, letter: str, thickness: int) -> np.ndarray:
        """Gets the specified letter from the dictionary that already computed the thickness values."""

        return self._letters[letter][thickness]

    def _getSizedLetterWithThickness(
        self, letter: str, thickness: int, size: int
    ) -> np.ndarray:
        """Calulates the letter 'box' for the specified thickness and size."""

        t = self._getLetterWithThickness(letter, int(thickness))
        return _scaleCentered(t, size)

    # Watch out these methods are abstract and intendet to be overridden or replaced !!

    def _scaleThickness(self, normalizedThickness: float):
        raise NotImplementedError

    def _scaleSize(self, normalizedSize: float):
        raise NotImplementedError

    def _getLetterBoxForNormalizedBrightness(
        self, letter: str, brightness: float
    ) -> np.ndarray:
        """Computes the correctly sized and thickness adjusted letter box for the specified brightness value."""

        # Compute the normalized thickness and size values
        thickness, size = thicknessAndHeightForBrightness(1.0 - brightness)

        # Now scale them accordingly
        # At this point the scaling methods should either be overridden or implemented
        sThickness = self._scaleThickness(self, thickness)
        sSize = self._scaleSize(self, size)

        return self._getSizedLetterWithThickness(letter, sThickness, sSize)

    def _initializeLetterBoxes(self):
        """Expects to be called after the thickness adjusted arrays have been computed.
        Will do the actual initialization of the finally used arrays.

        Returns:
            The height of a unscaled letter box."""

        tmp = {}

        for k in self._letters.keys():
            # For each letter build a pixel holder

            arrays = [
                self._getLetterBoxForNormalizedBrightness(k, ( b / NUM_SHADES ))
                for b in range(NUM_SHADES)
            ]

            h = PixelHolder(arrays)
            tmp[k] = h

        self._letters = tmp

        return arrays[0].shape[0]

    def _scaleLetterBoxes(self, ratio: float):
        """Scales all the letter boxes and assigns a default ndarray if an invalid key is given."""

        for _, v in self._letters.items():
            v.resampleWithRatio(ratio)

        h, w = v.arrays[0].shape

        defaultArray = np.ones((h, w), dtype=np.uint8) * 255

        self.defaultArray = lambda x: defaultArray

    # ENDFOLD

    def __init__(
        self,
        text: str,
        final_height: int,
        absolute_height: float,
        pixel_ratio: float,
        font: ImageFont,
        min_thickness: int = 0,
        max_thickness: int = 20,
    ):

        assert (
            DEFAULTS["supersamplingSize"] >= 512
        ), "AlphabetHolder calibrated to work with at least 512 as Supersampling size"

        self._font = font
        self._alphabet = set(list(text))
        self._alphabet.difference_update(set(" "))
        self._letters = {}

        for k in self._alphabet:
            thicknesses = [
                self._generateSquareLetterBox(k, t)
                for t in range(min_thickness, max_thickness)
            ]

            self._letters[k] = thicknesses

        self._removeWhitespaceFromLetters()

        # Set the proper scaling

        T_range = max_thickness - min_thickness

        def scaleT(self, normalized: float) -> int:
            return (int((1.0 - normalized ) * T_range) + min_thickness)

        max_width = list(self._letters.values())[0][0].shape[1]
        min_width = int(max_width * pixel_ratio)
        S_range = max_width - min_width - 1

        def scaleS(self, normalized: float) -> int:
            return int((1.0 - normalized) * S_range) + min_width

        self._scaleThickness = scaleT
        self._scaleSize = scaleS

        unscaledHeight = self._initializeLetterBoxes()

        scalingRatio = final_height / unscaledHeight

        self._scaleLetterBoxes(scalingRatio)

    def __getitem__(self, value: str) -> np.ndarray:
        # The AlphabetHolder class acts like a dictionary.

        try:
            return self._letters[value]
        except:
            return self.defaultArray


# ENDFOLD

# STARTFOLD ##### DITHER CLASSES


class ArrayDither(Params):
    """Abstraction of the raw parameters, implements all dithering functionality for ndarrays.
    Actual dithering though needs at least the ImageDither class as it needs to resize arrays using PIL."""

    def calculateAlpabet(self, text: str):
        """Builds a set of all different letters used for dithering."""

        self._alphabet = set(list(text))

    # STARTFOLD ##### CORE DITHERING FUNCTIONALITY
    # ----- A big part of the dithering functionality is being refactored to the AlphabetHolder class -----

    def _getSquareLetterBoxForLetter(self, letter: str) -> np.ndarray:
        """Produces an ndarray of shape [100,100] with a letter inside aligned to the top left corner."""

        img = Image.new("L", (DEFAULTS["supersamplingSize"],) * 2, 255)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), letter, (0), font=self._font)
        draw = ImageDraw.Draw(img)
        ar = np.array(img)
        return ar

    def _extractLetterOnlyBox(self, squareLetterBox: np.ndarray) -> np.ndarray:
        """Cuts whitespace out of a letter box, as defined in '_getSquareLetterBoxForLetter'"""

        x, y = np.where(np.logical_not(squareLetterBox == 255))
        l, t, r, b = (min(x), min(y), max(x), max(y))
        return squareLetterBox[l : r + 1, t : b + 1]

    def _calculateAndExtractBBoxesForAlphabet(self):
        """Builds a defaultdict containing all letterboxes for lettes in the alphabet, all sized at maximum height."""

        self._bbox = (
            *((DEFAULTS["supersamplingSize"],) * 2),  # (100, 100)
            0,
            0,
        )  # Init with max characters
        boxes = {}

        for c in self._alphabet:
            ar = self._getSquareLetterBoxForLetter(c)
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

    def _prepareLetterDictForSizes(self):
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

    def _assembleRow(self, rowText: str, rowPixels: np.ndarray) -> np.ndarray:
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

        sq_2 = sqrt(2)  # Aspect ratio of DIN A*

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

        # Load the font here
        self._font = ImageFont.truetype(
            font=self.fontName, size=int((72 / 100) * DEFAULTS["supersamplingSize"])
        )

        self._letters = AlphabetHolder(text, self.pixelHeight, 1.6, self.minPixelRatio, self._font)

        rows = []
        logging.info(f"Image downsampled to: {img.shape[1]}x{img.shape[0]} (WxH)")
        rowLength = img.shape[1]
        for i in range(img.shape[0] - 1):
            rows.append(
                self._assembleRow(text[i * rowLength : (i + 1) * rowLength], img[i, :])
            )

        rows[-1] = np.pad(
            rows[-1], (0, rows[0].shape[1] - rows[-1].shape[1]), constant_values=255
        )

        unpadded = np.concatenate(rows, axis=0)

        return self.padImage(unpadded) if self.padding > 0.0 else unpadded


class ImageDither(ArrayDither):
    """Abstraction of ArrayDither, but implements all dithering relevant functionality.
    When called via given text, and a PIL.Image it produces another PIL.Image
    containing the dithered content.

    Note that this class is tightly coupled with ArrayDither, since it implements
    all the resizing functionality such that ArrayDither doesn't need to use PIL."""

    def __call__(self, img: Image.Image, text: str) -> Image.Image:
        newText = text[:]

        if self.uppercase:
            newText = text.upper()

        super().calculateAlpabet(newText)

        # The font size in points
        fontPts = int((72 / 100) * DEFAULTS["supersamplingSize"])

        # Set the font now
        try:
            # Try setting the font for the given fontName
            self._font = ImageFont.truetype(
                f"{DEFAULTS['fontsDirectory']}{self.fontName}", fontPts
            )
        except Exception:
            # Fall back to the default font if the given font was not found
            logging.warning(
                f'Couldn\'t load "{self.fontName}" from {DEFAULTS["fontsDirectory"]}, falling back to default font'
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

        super()._calculateAndExtractBBoxesForAlphabet()

        ratio = (list(self._letters.values())[0]).shape
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

        # Image.fromarray(ar).show()
        d = super().__call__(ar, newText[: int(nH * nW)])
        return Image.fromarray(d)


class FileDither(ImageDither):
    """Abstraction of ImageDither, implements file based dithering functionality.
    Also callable with filenames the same content as ImageDither.

    Handles all IO bound exceptions, logs but also throws them when critical."""

    def __call__(self, inFile: str, outFile: str, text: str):

        try:
            img = Image.open(inFile)
        except Exception as e:
            logging.exception(f"Error reading image input file: {e}")
            exit(1)

        try:
            d = super().__call__(img, text)
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


# ENDFOLD

# STARTFOLD ##### MAIN METHOD


def main():
    """Main function, needs arguments to be refactored..."""

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
        "--font-name",
        metavar="<name or path to truetype font>",
        dest="fontName",
        type=str,
        help="Either a font name or a total or relative path to the .ttf or .ttc file., defaultsto 'Menlo-Bold.ttf' if loading the specified font fails the font will default to the latter.",
        default=DEFAULTS["fontName"],
    )

    paperGroup = parser.add_mutually_exclusive_group()

    def paperFormatter(f: str) -> float:
        if f == "A0":
            return 841.0

        if f == "A1":
            return 594.67

        if f == "A2":
            return 420.0

        raise ArgumentError("Format " + f + " not valid.")

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

    # parser.add_argument(
    #     "--truncate-text",
    #     metavar="<Number of characters where to truncate text>",
    #     type=int,
    #     help="Whether and where to truncate the given text such that it remains readable, -1 means don't truncate",
    #     default=-1,
    # )

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

    # ENDFOLD

    args = parser.parse_args()

    # STARTFOLD ##### Setup logging

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

    # ENDFOLD

    # STARTFOLD ##### Check parsed arguments for given preconditions

    try:
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
            outPath = _exchangeSuffixInPath(imgPath, "_dithered.png")
            logging.info(f"No output path given, saving to {outPath}")

        pixelHeight = args.pixel_height
        assert (
            4 < pixelHeight < 50
        ), f"Given pixelHeight: {pixelHeight} was not in range (5 to 49)"

        pixelRatio = args.pixel_ratio
        assert (
            0.0 < pixelRatio < 1.0
        ), f"Pixel ratio must strictly be between 0 and 1: {pixelRatio} was given!"

        uppercase = args.uppercase  # Well with a bool there can't be too much wrong

        assert args.paperHeight > 0.0, "Negative paperHeight not allowed!"

        if uppercase:
            logging.info("Uppercasing all letters")

        truncate = args.truncate_text

        assert 100.0 > args.padding >= 0.0, "Padding value was not in range (0-100%)"

    except AssertionError as err:
        logging.exception("Argument error occurred: " + str(err))
        exit(1)

    # ENDFOLD

    # Now we can dither
    with open(args.in_text, "r") as f:
        txt = f.read()

    if truncate > -1:
        logging.info(
            f"Trunc or Repeat text to {truncate} characters, removed {(len(txt) - truncate) / len(txt): .2%}"
        )

        txt = txt[:truncate]
        txt = _truncateOrRepeat(txt, truncate)

    kwargs = {
        "pixelHeight": pixelHeight,
        "uppercase": uppercase,
        "fontName": args.fontName,
        "minPixelRatio": pixelRatio,
        "minPixelHeight": args.minPixelHeight,
        "paperHeight": args.paperHeight,
        "padding": args.padding / 100.0,  # percentage value given here
    }

    logging.debug("Dithering parameters used:")

    for k, v in kwargs.items():
        logging.debug(f'\t"{k}" = {v}')

    p = FileDither(**kwargs)

    # p = FileDither(
    #     pixelHeight=pixelHeight,
    #     uppercase=uppercase,
    #     fontName=args.fontName,
    #     minPixelRatio=pixelRatio,
    # )

    logging.info(
        "dithering "
        + imgPath
        + " using text: "
        + args.in_text
        + " (with "
        + str(len(txt))
        + ") characters."
    )

    try:
        p(imgPath, outPath, txt)
    except Exception as e:
        logging.exception(f"Uncaught error occurred during dithering: {e}")
        exit(1)

    exit(0)  # Success should return code 0


# ENDFOLD

if __name__ == "__main__":
    main()

