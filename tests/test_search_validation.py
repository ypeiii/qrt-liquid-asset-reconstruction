"""Invalid search controls must not create or touch persistent Optuna state."""

from pathlib import Path

import numpy as np
import optuna
import pytest

from quant_portfolio.data import make_synthetic_panel
from quant_portfolio.search import _validate_search_arguments, run_optuna_search


@pytest.fixture
def panel():
    return make_synthetic_panel(n_days=6, n_test_days=2, n_targets=2, n_features=3)


INVALID_ARGUMENTS = [
    ("initial_trials", 0, "initial_trials"),
    ("initial_trials", True, "initial_trials"),
    ("initial_trials", np.bool_(True), "initial_trials"),
    ("initial_trials", 1.5, "initial_trials"),
    ("trials_per_reopen", -1, "trials_per_reopen"),
    ("trials_per_reopen", False, "trials_per_reopen"),
    ("trials_per_reopen", 2.5, "trials_per_reopen"),
    ("n_splits", 1, "n_splits"),
    ("n_splits", 7, "n_splits"),
    ("n_splits", True, "n_splits"),
    ("n_splits", np.bool_(True), "n_splits"),
    ("n_splits", 3.0, "n_splits"),
    ("seed", -1, "seed"),
    ("seed", 2**32, "seed"),
    ("seed", False, "seed"),
    ("seed", np.bool_(False), "seed"),
    ("seed", 42.0, "seed"),
]


@pytest.mark.parametrize("name,value,message", INVALID_ARGUMENTS)
def test_invalid_arguments_create_no_directory_or_database(panel, tmp_path, name, value, message):
    storage = tmp_path / "not-created" / "study.sqlite3"
    kwargs = {"initial_trials": 1, name: value}
    with pytest.raises(ValueError, match=message):
        run_optuna_search("extra_trees", panel, storage, **kwargs)
    assert not storage.parent.exists()
    assert not storage.exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_trials": False},
        {"trials_per_reopen": 0},
        {"n_splits": 1},
        {"seed": 2**32},
    ],
)
def test_invalid_arguments_do_not_modify_an_existing_study(panel, tmp_path, kwargs):
    database = tmp_path / "existing.sqlite3"
    storage = optuna.storages.RDBStorage(url="sqlite:///" + database.as_posix())
    study = optuna.create_study(study_name="existing", storage=storage, direction="maximize")
    study.set_user_attr("marker", "unchanged")
    storage.remove_session()
    storage.engine.dispose()
    before = database.read_bytes()

    with pytest.raises(ValueError):
        run_optuna_search("extra_trees", panel, database, study_name="existing", **kwargs)

    assert database.read_bytes() == before


def test_numpy_integer_controls_are_normalized_to_builtin_ints(panel):
    values = _validate_search_arguments(
        panel,
        np.int64(3),
        np.int32(2),
        np.int64(5),
        np.uint32(2**32 - 1),
    )
    assert values == (3, 2, 5, 2**32 - 1)
    assert all(type(value) is int for value in values)
