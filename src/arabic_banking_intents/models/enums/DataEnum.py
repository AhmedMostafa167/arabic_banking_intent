from enum import StrEnum


class DataVariantEnum(StrEnum):
    OFFICIAL = "official"  # SinaLab's train and validation files; the validation file leaks
    GROUPED = "grouped"  # our split from the corpus, grouped by source question


class TestSetEnum(StrEnum):  # the official test files, kept as published
    MSA = "MSA"
    PAL = "PAL"
    SAUDI = "Saudi"
    MOROCCAN = "Moroccan"
    TUNISIAN = "Tunisian"
