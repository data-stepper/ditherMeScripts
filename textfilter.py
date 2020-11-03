#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

# STARTFOLD ##### IMPORTS

import os
import sys
import re
from time import time
import argparse
import abc
from math import ceil
from functools import wraps
import logging
import inspect
import pdb

# ENDFOLD

# STARTFOLD ##### SCRIPT CONSTANTS

_SUPPORTED_FORMATS = {"pdf", "html", "txt", "htm", "html-wikisource", "detect"}
_ENGLISH_ALPHABET = (
    ' QWERTZUIOPASDFGHJKLYXCVBNMqwertzuiopasdfghjklyxcvbnm,.;:/()-?!1234567890"$§%&'
)

# ENDFOLD

# STARTFOLD ##### SUPPLEMETARY FUNCTIONS


def _detectFormatFromFile(filename: str) -> str:
    """Finds the suffix from a filename, case-insensitive."""

    return filename[filename.rfind(".") + 1 :].lower()


def _checkValidFormat(formatGiven: str):
    """Checks if a given suffix is supported."""

    if formatGiven not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"{formatGiven} is not supported, supported formats are {_SUPPORTED_FORMATS}"
        )


def _exchangeSuffixInPath(path: str, newSuffix: str) -> str:
    """Exchanges the suffix in a path for a new one, newSuffix must start with a dot."""

    assert "." in newSuffix, "new Suffix needs to start with a dot"
    dotIndex = path.rfind(".")
    return path[:dotIndex] + newSuffix


# ENDFOLD

# STARTFOLD ##### FILTERING CLASSES

# The base class for all other filters
class Filter(abc.ABC):
    """Abtract base class for all text based filters."""

    filterName = "Standard Filter"

    @abc.abstractmethod
    def __call__(self, data: str) -> str:
        pass

    def __str__(self):
        return f"{self.filterName} ({self.__class__.__name__})"


# The Decorator which all string filters shall use
def stringFilter(filterFunc):
    """Decorator for the call methods of Filter classes.
    Handles logging and zero length checking."""

    @wraps(filterFunc)
    def wrapped(*args, **kwargs):
        # Perform zero length check
        # args[1] is the 'data' argument
        fname = str(args[0])

        if len(args[1]) == 0:
            logging.exception(f"Received zero length string in {fname}!")
            exit(1)

        filtered = filterFunc(*args, **kwargs)

        if len(filtered) == 0:
            logging.exception(
                f"Fatal error occurred in {fname}, produced zero length string!"
            )
            exit(1)

        pctChange = 1.0 - (len(filtered) / len(args[1]))

        logging.debug(
            f"{fname}: {len(args[1])} -> {len(filtered)} ({-pctChange:+.2%} change)"
        )

        return filtered

    return wrapped


# STARTFOLD ##### TEXT BASED FILTERS


class alphabetFilter(Filter):
    """Filters text to only keep alphabetic characters (specified in alphabet)."""

    def __init__(self, alphabet: str):
        self.alphabet = set(list(alphabet))
        self.filterName = "AlphabetFilter"

    @stringFilter
    def __call__(self, data: str) -> str:
        badChars = set(list(data))
        badChars.difference_update(self.alphabet)
        newStr = data[:]

        for c in badChars:
            newStr = newStr.replace(c, "")

        return newStr

    @classmethod
    def englishStandardAlphabet(cls):
        inst = cls(_ENGLISH_ALPHABET)
        inst.filterName = "EnglishAlphabet"
        return inst

    @classmethod
    def germanStandardAlphabet(cls):
        inst = cls(_ENGLISH_ALPHABET + "ÄÜÖäüöß")
        inst.filterName = "GermanAlphabet"
        return inst


# STARTFOLD ##### REGEX BASED FILTERS


class RegexFilter(Filter):
    """Filter that replaces regex matches with given replacement string."""

    def __init__(self, pattern: str, replacement: str = ""):
        self.regex = re.compile(pattern)
        self.replacement = replacement
        self.filterName = "RegexFilter"

    @stringFilter
    def __call__(self, data: str) -> str:
        return self.regex.subn(self.replacement, data)[0]

    def __str__(self) -> str:
        return f'RegexFilter ("{str(self.regex).replace("re.compile", "")[2:-2]}")'

    @classmethod
    def multiSpacesLinebreaksFilter(cls):
        """Collapses multiple Spaces and linebreaks into a single one."""

        inst = cls("[\n ]+", " ")
        inst.filterName = "multiSpacesLinebreaksFilter"
        return inst

    @classmethod
    def linebreakWordWrapFilter(cls):
        """Filters broken up words because of a linebreak in the source file."""

        inst = cls("-[\n ]", "")
        inst.filterName = "linebreakWordWrapFilter"
        return inst

    @classmethod
    def lineNumberFilter(cls):
        """Removes line numbering at the end of the line, needs to be used before linebreak filters."""

        inst = cls("( )+[0-9]+\n", "")
        inst.filterName = "lineNumberFilter"
        return inst

    @staticmethod
    def shrinkBrackets():
        """Shrinks brackets like ( content ) to (content) to further remove whitespace."""
        inst = MultiFilter(
            [
                RegexFilter("\( ", "("),
                RegexFilter(" \)", ")"),
                RegexFilter("\[ ", "["),
                RegexFilter(" \]", "]"),
                RegexFilter("{ ", "{"),
                RegexFilter(" }", "}"),
            ]
        )

        inst.filterName = "^BracketShrinker"
        return inst

    @staticmethod
    def shrinkSentenceEnding():
        """Shrinks sentence endings like \" .\" down to just \".\"."""
        inst = MultiFilter(
            [
                RegexFilter(" \.", "."),
                RegexFilter(" \,", ","),
                RegexFilter(" !", "!"),
                RegexFilter(" :", ":"),
                RegexFilter(" \?", "?"),
                RegexFilter(" ;", ";"),
            ]
        )

        inst.filterName = "^EndingsShrinker"
        return inst


class complexRegexFilter(RegexFilter):
    """A regex filter which replaces with a function evaluated on the found match"""

    def __init__(self, pattern: str, replacementFunc):
        self.regex = re.compile(pattern)
        self.replacementFunc = replacementFunc

    @stringFilter
    def __call__(self, data: str) -> str:
        text = data[:]
        newText = []

        prevE = 0

        for match in self.regex.finditer(text):
            s, e = match.span(0)
            newText.append(text[prevE:s])
            newText.append(self.replacementFunc(match.group(0)))
            prevE = e

        return "".join(newText)

    @classmethod
    def sentenceEnding(cls):
        f = lambda s: s[-1]
        return cls(" [!?.:]", f)


# ENDFOLD


class TargetLengthFilter(Filter):
    """A Filter that either cuts or repeats text until it has the specified length."""

    def __init__(self, targetLength: int):
        self.targetLength = targetLength
        self.filterName = "TargetLengthFilter"

    @stringFilter
    def __call__(self, data: str) -> str:
        txt = data[:]
        textLength = len(txt)

        if textLength > self.targetLength:
            return txt[: self.targetLength]
        else:
            factor = int(ceil(self.targetLength / textLength))
            return (txt * factor)[: self.targetLength]
            pass


class MultiFilter(Filter):
    """Filter consisting of multiple filters, called in sequence."""

    def __init__(self, filters: list):
        self.filters = filters
        self.filterName = "MultiFilter"

    @stringFilter
    def __call__(self, data: str) -> str:
        txt = data[:]

        for f in self.filters:
            txt = f(txt)

        return txt

    @classmethod
    def standardTextFilter(cls):
        filters = [
            RegexFilter.lineNumberFilter(),
            RegexFilter.linebreakWordWrapFilter(),
            alphabetFilter.germanStandardAlphabet(),
            RegexFilter.multiSpacesLinebreaksFilter(),
            RegexFilter.shrinkBrackets(),
            RegexFilter.shrinkSentenceEnding(),
            # complexRegexFilter.sentenceEnding(), # This filter doesn not yet work properly
            TargetLengthFilter(1000000),
        ]

        inst = cls(filters)
        inst.filterName = "standardTextFilter"
        return inst


# ENDFOLD

# STARTFOLD ##### FILE BASED FILTERS


class FileFilter:
    """Filter that uses a given Filter to filter from a filePath to a filePath."""

    def __init__(self, applyingFilter: Filter):
        self.f = applyingFilter

    def __call__(self, inputPath: str, outputPath: str):
        with open(inputPath, "r") as inFile:
            txt = inFile.read()

        filtered = self.f(txt)

        with open(outputPath, "w") as outFile:
            outFile.write(filtered)

        return {
            "before": len(txt),
            "after": len(filtered),
            "reduction": f"{(1.0 - (len(filtered) / len(txt))): .2%}",
        }


class TypeConverter(FileFilter):
    """Just like FileFilter, but it can accept all supported suffixes in the filePath."""

    def __init__(self, applyingFilter: Filter):
        self.f = applyingFilter

        # STARTFOLD ##### FILE TYPE CONVERTER FUNCTIONS

        def pdf(inPath: str) -> str:
            newPath = _exchangeSuffixInPath(inPath, ".txt")
            r = os.system(f"ps2ascii {inPath} {newPath}")
            # -----> THIS SCRIPT MAY NOT CORRECTLY WORK ON SOME PDFS
            # MAYBE LOOK FOR A DIFFERENT WAY TO CONVERT??
            if r != 0:
                # Error occurred in ps2ascii.
                raise IOError(
                    f"Error converting pdf file from path: {inPath}, error code {r}"
                )
            return newPath

        # ENDFOLD

        self.formatConverters = {".pdf": pdf, ".txt": lambda x: x}

    def __call__(self, inputPath: str, outputPath: str):
        # check the format given
        assert outputPath.endswith(".txt"), (
            "Output path needs to end with .txt: " + outputPath
        )

        newPath = ""

        for k, v in self.formatConverters.items():
            if inputPath.endswith(k):
                # Format detected
                newPath = v(inputPath)

        if newPath == "":
            raise ValueError(f"Unsupported format given: {inputPath}")

        assert newPath.endswith(".txt"), "Path was converted to non .txt file" + newPath

        super().__call__(newPath, outputPath)


# ENDFOLD

# ENDFOLD

# STARTFOLD ##### MAIN FUNCTION


def main():
    """Main Method needs refactoring such that it accepts the parsed arguments."""

    parser = argparse.ArgumentParser(
        description="Script for filtering and extracting textual data from text sources like ebooks"
    )

    # STARTFOLD ##### Adding arguments
    parser.add_argument(
        "-in",
        "--in-file",
        type=str,
        help="Input Filepath to filter text data from",
        required=True,
    )

    parser.add_argument(
        "-out",
        "--out-file",
        type=str,
        required=True,
        help="Output Filepath to write filtered text (must end with .txt)",
    )

    parser.add_argument(
        "-fmt",
        "--inputFormat",
        type=str,
        default="detect",
        help="Force usage of specified input format",
    )

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
    logging.debug("textfilter.py script called with: " + str(sys.argv))

    # ENDFOLD

    if args.inputFormat == "detect":
        givenFormat = _detectFormatFromFile(args.in_file)
    else:
        givenFormat = args.inputFormat

    _checkValidFormat(givenFormat)

    # fileFilter = FileFilter(MultiFilter.standardTextFilter())
    fileFilter = TypeConverter(MultiFilter.standardTextFilter())

    start = time()
    stats = fileFilter(args.in_file, args.out_file)
    end = time()

    logging.info(f"Successfully filtered, took {end - start: .5f}s")


if __name__ == "__main__":
    main()


# ENDFOLD
