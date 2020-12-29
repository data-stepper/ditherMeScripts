from dataclasses import dataclass
import os

DEFAULTS = {
    "fontName": "Menlo-Bold",
    "fontsDirectory": "",  # Empty since fonts are global on macos
    "pixelRatio": 0.6,
    "pixelHeight": 28,
    "thicknessRange": (1, 4),
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
    fontName: str = DEFAULTS["fontName"]


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

