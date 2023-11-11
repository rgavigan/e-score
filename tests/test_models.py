import pytest
import importlib
models = importlib.import_module("e-score.models")

# Test Model Retrieval for valid model
def test_get_model():
    model, tokenizer = models.get_model("ProtT5")
    assert model is not None
    assert tokenizer is not None

# Test Model Retrieval for invalid model
def test_get_model_invalid():
    model, tokenizer = models.get_model("InvalidModel")
    assert model is None
    assert tokenizer is None