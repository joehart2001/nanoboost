import numpy as np
import pytest
from nanoboost.scripts.utils.utils import save_with_pickle, unpickle


def test_round_trip(tmp_path):
    data = {"arr": np.array([1.0, 2.0, 3.0]), "val": 42}
    path = str(tmp_path / "data.pkl")
    save_with_pickle(path, data)
    loaded = unpickle(path)
    assert loaded["val"] == 42
    np.testing.assert_array_equal(loaded["arr"], data["arr"])


def test_round_trip_nested(tmp_path):
    data = [np.ones(100), np.zeros(50)]
    path = str(tmp_path / "events.pkl")
    save_with_pickle(path, data)
    loaded = unpickle(path)
    assert len(loaded) == 2
    np.testing.assert_array_equal(loaded[0], data[0])
