"""Responsible for precomputing all the character pixels.

This Module mainly builds the AlphabetHolder which stores for each character
the corresponding ndarray sized and thickened properly for each brightness value.
"""
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import PIL
import logging
from math import cos, pi, floor


def clip(value, lower, upper):
    return lower if value < lower else upper if value > upper else value


def _resample_array_with_ratio(original: np.ndarray, ratio: float) -> np.ndarray:
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


def thickness_and_height_for_brightness_function(brightness: float) -> (float, float):
    """Calculates the normalized thickness and character height for a given normalized brightness value."""

    return (
        cos(pi * (brightness / 2.0)),  # Thickness value
        cos(pi * (brightness / 2.0)),  # Size value
    )


def _scale_character_array_centered(pixel: np.ndarray, height: int) -> np.ndarray:
    """Scales a pixel down to a given height, leaving content centered."""

    scaled = _resample_array_with_ratio(pixel, height / pixel.shape[1])

    padW = pixel.shape[0] - scaled.shape[0]
    padH = pixel.shape[1] - scaled.shape[1]
    scaled = np.pad(
        scaled,
        ((floor(padW / 2.0),) * 2, (floor(padH / 2.0),) * 2),
        constant_values=255,
    )

    scaled = np.pad(
        scaled,
        ((0, pixel.shape[0] - scaled.shape[0]), (0, pixel.shape[1] - scaled.shape[1]),),
        constant_values=255,
    )

    return scaled


class PixelHolder:
    """'Dummy' class that just holds some arrays and returns the correct one when called."""

    def __init__(self, arraysOfShades: list):
        self.arrays = arraysOfShades[:]
        self.numShades = len(self.arrays)

        # Resamples brightness values for number of shades
        def samplingFunc(brightness: int) -> int:
            return int((clip(brightness, 0.0, 255.0) / 255.0) * (self.numShades - 1))

        self.samplingFunc = samplingFunc

    def __call__(self, brightness: int) -> np.ndarray:
        return self.arrays[self.samplingFunc(brightness)]

    def resample_arrays_with_ratio(self, ratio: float):
        sampled = [_resample_array_with_ratio(ar, ratio) for ar in self.arrays]

        self.arrays = sampled


class AlphabetHolder:
    """This class holds all the single 'pixels' with correct sizes and thicknesses.

    It is instantiated using a string of either a raw text or an already filtered alphabet.
    Items of it are PixelHolders and can be accessed like a dictionary:

        Example usage:

            a = AlphabetHolder('some text I want to dither.', final_height=12, absolute_height=1.6)
            letter_s = a['s']
            pix_brightest = letter_s(255)
            # Now you have the corresponding ndarray to append to the row."""

    def _generate_square_letter_box(self, letter: str, thickness: int) -> np.ndarray:
        """Produces an ndarray of supersamplingSize with a letter aligned to the top left corner,
        using the specified thickness value as stroke thickness."""

        img = Image.new("L", (self._supersampling_size,) * 2, 255)
        draw = ImageDraw.Draw(img)
        offset = int(self._supersampling_size * 0.2)
        # self._font_list is still a single font only
        # implement the multi font functionality here.
        draw.text(
            (offset, offset), letter, (0), font=self._font_list, stroke_width=thickness,
        )
        draw = ImageDraw.Draw(img)
        ar = np.array(img)
        return ar

    def _unpad_whitespace_from_letters(self):
        """Computes bboxes for all letters with all thicknesses and removes all unnecessary whitespace.
        Places extracted letters back into self._lettes dictionary."""

        # Lazily make tuple of (SS, SS, 0, 0)
        bbox = (*((self._supersampling_size,) * 2), 0, 0)
        largest_characters_area = 0

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

                    area = (b - t) * (r - l)

                    if (
                        area > largest_characters_area
                    ):  # If a character was the new 'largest'
                        largest_character = c
                        largest_characters_area = area

                else:
                    logging.debug(f"Found invisible character: {c} ({ord(c)}) in hex")

        l, t, r, b = bbox

        logging.debug(f"Bounding box for letters computed: {l}, {t}, {r}, {b} (ltrb)")
        logging.debug(
            f"Largest character found: {largest_character}, with area: {largest_characters_area} pix^2"
        )

        extracted = {}

        for c in self._alphabet:
            tmp = [v[l : r + 1, t : b + 1] for v in self._letters[c]]

            extracted[c] = tmp

        self._letters = extracted

    def _get_letter_with_thickness(self, letter: str, thickness: int) -> np.ndarray:
        """Gets the specified letter from the dictionary that already computed the thickness values."""

        return self._letters[letter][thickness]

    def _get_sized_letter_with_thickness(
        self, letter: str, thickness: int, size: int
    ) -> np.ndarray:
        """Calulates the letter 'box' for the specified thickness and size."""

        t = self._get_letter_with_thickness(letter, int(thickness))
        return _scale_character_array_centered(t, size)

    # Watch out these methods are abstract and intendet to be overridden or replaced !!

    def _scale_thickness(self, normalizedThickness: float):
        raise NotImplementedError

    def _scale_size(self, normalizedSize: float):
        raise NotImplementedError

    def _produce_letter_box_with_normalized_brightness_values(
        self, letter: str, brightness: float
    ) -> np.ndarray:
        """Computes the correctly sized and thickness adjusted letter box for the specified brightness value."""

        # Compute the normalized thickness and size values
        thickness, size = thickness_and_height_for_brightness_function(1.0 - brightness)

        # Now scale them accordingly
        # At this point the scaling methods should either be overridden or implemented
        sThickness = self._scale_thickness(self, thickness)
        sSize = self._scale_size(self, size)

        return self._get_sized_letter_with_thickness(letter, sThickness, sSize)

    def _rebuild_letter_boxes_array(self):
        """Expects to be called after the thickness adjusted arrays have been computed.
        Will do the actual initialization of the finally used arrays.

        Returns:
            The height of an unscaled letter box."""

        tmp = {}

        for k in self._letters.keys():
            # For each letter build a pixel holder

            arrays = [
                self._produce_letter_box_with_normalized_brightness_values(
                    k, (b / self._num_shades)
                )
                for b in range(self._num_shades)
            ]

            h = PixelHolder(arrays)
            tmp[k] = h

        self._letters = tmp

        return arrays[0].shape[0]

    def _scaleLetterBoxes(self, ratio: float):
        """Scales all the letter boxes and assigns a default ndarray if an invalid key is given."""

        for _, v in self._letters.items():
            v.resample_arrays_with_ratio(ratio)

        h, w = v.arrays[0].shape

        defaultArray = np.ones((h, w), dtype=np.uint8) * 255

        self.defaultArray = lambda x: defaultArray

    # ENDFOLD

    def __init__(
        self,
        text: str,
        final_height: int,
        absolute_height: float,
        fonts: list,
        min_pixel_ratio: float,
        max_pixel_ratio: float = 1.0,
        min_thickness: int = 0,
        max_thickness: int = 20,
        supersampling_size: int = 256,
        num_shades: int = 64,
    ):

        assert (
            max_pixel_ratio > min_pixel_ratio
        ), "Max pixel ratio should be bigger than min_pixel_ratio"

        self._font_list = fonts
        self._alphabet = set(list(text))
        self._alphabet.difference_update(set(" "))
        self._letters = {}
        self._supersampling_size = supersampling_size
        self._num_shades = num_shades

        for k in self._alphabet:
            thicknesses = [
                self._generate_square_letter_box(k, t)
                for t in range(min_thickness, max_thickness + 1)
            ]

            self._letters[k] = thicknesses

        self._unpad_whitespace_from_letters()

        # Set the proper scaling

        T_range = max_thickness - min_thickness - 1

        logging.debug(f"Thickness range: {T_range}")

        def scaleT(self, normalized: float) -> int:
            return int((1.0 - normalized) * T_range) + min_thickness

        max_width = int(list(self._letters.values())[0][0].shape[1] * max_pixel_ratio)
        min_width = int(max_width * min_pixel_ratio)
        S_range = max_width - min_width - 1

        logging.debug(f"Size range: {S_range}")

        def scaleS(self, normalized: float) -> int:
            return int((1.0 - normalized) * S_range) + min_width

        self._scale_thickness = scaleT
        self._scale_size = scaleS

        unscaledHeight = self._rebuild_letter_boxes_array()

        scalingRatio = final_height / unscaledHeight

        self._scaleLetterBoxes(scalingRatio)

        ltr = list(self._letters.values())[0]
        logging.debug("Testing brightness range for arbitrary character:")
        logging.debug(
            f"Min: {ltr(0).mean() / 255.0} ( ltr(0) ), Max: {ltr(255).mean() / 255.0} ( ltr(255) )"
        )

    def __getitem__(self, value: str) -> np.ndarray:
        # The AlphabetHolder class acts like a dictionary.

        try:
            return self._letters[value]
        except:
            return self.defaultArray

    def values(self):
        return self._letters.values()
