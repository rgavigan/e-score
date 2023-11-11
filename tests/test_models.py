from src.models import get_model

# Test Model Retrieval for valid model
def test_get_model():
    model, tokenizer = get_model("ProtT5")
    assert model is not None
    assert tokenizer is not None

# Test Model Retrieval for invalid model
def test_get_model_invalid():
    model, tokenizer = get_model("InvalidModel")
    assert model is None
    assert tokenizer is None