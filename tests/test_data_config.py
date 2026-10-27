"""Tests for data_config.py — pure logic, zero external dependencies."""
from data_config import (
    hh_role, hh_cluster, date_period,
    DATE_TRAIN_END, DATE_VAL_END,
)


def test_hh_role_train():
    assert hh_role(129) == "train"   # cluster-0 train household
    assert hh_role(251) == "train"   # cluster-1 train household


def test_hh_role_test():
    assert hh_role(1) == "test"      # cluster-0 test household
    assert hh_role(5) == "test"      # cluster-1 test household


def test_hh_role_unused():
    assert hh_role(999) == "unused"
    assert hh_role(0) == "unused"


def test_hh_cluster_train_households():
    assert hh_cluster(129) == 0
    assert hh_cluster(251) == 1
    assert hh_cluster(143) == 2
    assert hh_cluster(176) == 3


def test_hh_cluster_test_households():
    assert hh_cluster(1) == 0
    assert hh_cluster(5) == 1
    assert hh_cluster(12) == 2
    assert hh_cluster(55) == 3


def test_hh_cluster_unknown_returns_none():
    assert hh_cluster(999) is None


def test_date_period_train():
    assert date_period("2011-01-01") == "train"
    assert date_period("2010-07-01") == "train"


def test_date_period_val():
    assert date_period("2012-03-01") == "val"
    assert date_period("2012-06-30") == "val"


def test_date_period_test():
    assert date_period("2012-08-01") == "test"
    assert date_period("2013-01-01") == "test"


def test_date_period_exact_boundaries():
    # DATE_TRAIN_END is the first date of the val period
    assert date_period(DATE_TRAIN_END) == "val"
    # DATE_VAL_END is the first date of the test period
    assert date_period(DATE_VAL_END) == "test"
