import pytest
import importlib
embeddings = importlib.import_module("e-score.embeddings")
models = importlib.import_module("e-score.models")

# Test Embedding Retrieval for valid model
def test_get_embeddings():
    # Get model
    model, tokenizer = models.get_model("ProtT5")
    assert model is not None
    assert tokenizer is not None

    # Get embeddings
    sequences = embeddings.load_fasta("tests/test.fasta")
    emb1, emb2 = embeddings.get_embeddings(sequences[0][1], sequences[1][1], model, tokenizer, "ProtT5")

    # Validate embedding dimensions
    assert len(emb1) == len(sequences[0][1])
    assert len(emb2) == len(sequences[1][1])
    assert len(emb1[0]) == 1024
    assert len(emb2[0]) == 1024

# Test Embedding Retrieval for invalid model
def test_get_embeddings_invalid():
    # Get model
    model, tokenizer = models.get_model("InvalidModel")
    assert model is None
    assert tokenizer is None

    # Get embeddings
    sequences = embeddings.load_fasta("tests/test.fasta")
    emb1, emb2 = embeddings.get_embeddings(sequences[0][1], sequences[1][1], model, tokenizer, "InvalidModel")

    # Validate embedding dimensions
    assert len(emb1) == 0
    assert len(emb2) == 0