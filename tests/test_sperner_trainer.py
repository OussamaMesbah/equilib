"""Tests for the SpernerTrainer PEFT integration."""

import numpy as np
import pytest
from equilib.sperner_trainer import SpernerTrainer, BaseObjective


def test_base_objective_raises():
    obj = BaseObjective()
    with pytest.raises(NotImplementedError):
        obj(None)


def test_trainer_mock_init():
    trainer = SpernerTrainer("mock", ["a", "b", "c"], [], mock=True)
    assert trainer.n_objs == 3
    assert trainer.mock is True


def test_trainer_evaluate_mixed_model():
    trainer = SpernerTrainer("mock", ["a", "b", "c"], [], mock=True)
    losses = trainer.evaluate_mixed_model(np.array([0.5, 0.3, 0.2]))
    assert len(losses) == 3
    assert all(isinstance(l, float) for l in losses)


def test_trainer_evaluate_caching():
    trainer = SpernerTrainer("mock", ["a", "b", "c"], [], mock=True)
    w = np.array([0.5, 0.3, 0.2])
    losses1 = trainer.evaluate_mixed_model(w)
    losses2 = trainer.evaluate_mixed_model(w)
    assert losses1 == losses2
    assert len(trainer._eval_cache) == 1


def test_trainer_oracle_label():
    trainer = SpernerTrainer("mock", ["a", "b", "c"], [], mock=True)
    label = trainer.oracle_label(np.array([0.5, 0.3, 0.2]))
    assert isinstance(label, int)
    assert 0 <= label < 3


def test_trainer_train_returns_weights():
    trainer = SpernerTrainer("mock", ["a", "b", "c"], [], mock=True)
    result = trainer.train(grid_size=10)
    assert isinstance(result, np.ndarray)
    assert len(result) == 3
    assert np.isclose(result.sum(), 1.0, atol=0.05)


def test_trainer_train_generator():
    trainer = SpernerTrainer("mock", ["a", "b", "c"], [], mock=True)
    gen = trainer.train_generator(grid_size=5)
    # Get first proposal
    weights, phase = next(gen)
    assert len(weights) == 3
    # Send a label back
    try:
        weights, phase = gen.send(0)
        assert len(weights) == 3
    except StopIteration:
        pass  # Solver may converge in one step at low resolution
