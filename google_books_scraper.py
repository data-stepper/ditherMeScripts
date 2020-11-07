#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
This script can scrape the textual contents of an entire book from google books.
It needs at least one positional argument which is the url of the google books text view link.
It is expected that in that link there is a {} where the script will place the page number.

Also it expects that in the called path there are two directories:
    - raw_scraped
    - filtered_scraped

In these directories, scraped content will be placed as .txt with the book title according to the
scraped html content.
"""

# STARTFOLD ##### IMPORTS

import requests
import re
import logging
from argparse import ArgumentParser

# ENDFOLD

# STARTFOLD ##### MAIN METHOD


def scrape_content(targetURL: str):
    """Scrapes the content from the url and returns the string containing all the book's content."""

    content = ""


def main(args):
    print("url: " + args.targetURL)


# ENDFOLD

if __name__ == "__main__":
    # parse some args
    parser = ArgumentParser(description=__doc__)

    parser.add_argument(metavar="<google books url>", dest="targetURL", type=str)

    args = parser.parse_args()

    main(args)
