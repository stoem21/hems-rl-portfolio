"""Household split + date constants. Importable everywhere in the backend."""

TRAIN_HOUSEHOLDS = {
    0: [129, 294, 204, 174, 271, 264],
    1: [251, 206, 245, 291, 214, 213],
    2: [143, 207, 241, 177, 114, 250],
    3: [176, 178, 224, 165, 183, 236],
}
TEST_HOUSEHOLDS = {
    0: [1, 56, 36, 3, 64, 9, 95, 52],
    1: [5, 76, 73, 14, 48, 26, 59, 16],
    2: [12, 17, 37, 92, 91, 88, 63, 78],
    3: [55, 33, 25, 57, 18, 100, 45, 97],
}

TRAIN_FLAT = [hh for ids in TRAIN_HOUSEHOLDS.values() for hh in ids]
TEST_FLAT = [hh for ids in TEST_HOUSEHOLDS.values() for hh in ids]

DATE_DATASET_START = "2010-07-01"
DATE_TRAIN_END = "2012-01-01"
DATE_VAL_END = "2012-07-01"  # = test period start


def hh_role(hh_id: int) -> str:
    if hh_id in TRAIN_FLAT:
        return "train"
    if hh_id in TEST_FLAT:
        return "test"
    return "unused"


def hh_cluster(hh_id: int):
    # NB: TRAIN and TEST share cluster keys 0-3, so they must be scanned
    # separately — merging them with {**TRAIN, **TEST} would drop TRAIN.
    for mapping in (TRAIN_HOUSEHOLDS, TEST_HOUSEHOLDS):
        for c, ids in mapping.items():
            if hh_id in ids:
                return c
    return None


def date_period(date_str: str) -> str:
    if date_str < DATE_TRAIN_END:
        return "train"
    if date_str < DATE_VAL_END:
        return "val"
    return "test"
