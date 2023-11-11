from src.embeddings import get_pair_embeddings, load_fasta, get_fasta_embeddings
from src.models import get_model

# Test Embedding Retrieval for valid model
def test_get_pair_embeddings():
    Model, Tokenizer = get_model("ProtT5")
    # Get embeddings
    sequences = load_fasta("tests/test.fasta")
    embeddings = get_pair_embeddings(sequences[0][1], sequences[1][1], Model, Tokenizer, "ProtT5")

    # Validate embedding dimensions
    assert len(embeddings) == 2
    assert len(embeddings[0]) == len(sequences[0][1])
    assert len(embeddings[1]) == len(sequences[1][1])
    assert len(embeddings[0][0]) == 1024
    assert len(embeddings[1][0]) == 1024

# Test Embedding Retrieval for invalid model
def test_get_pair_embeddings_invalid():
    # Get embeddings
    sequences = load_fasta("tests/test.fasta")
    embeddings = get_pair_embeddings(sequences[0][1], sequences[1][1], None, None, "InvalidModel")

    # Validate no embeddings returned
    assert len(embeddings) == 0

# Test Embedding Retrieval for fasta and valid model
def test_get_fasta_embeddings():
    Model, Tokenizer = get_model("ProtT5")
    # Get embeddings
    sequences = load_fasta("tests/test.fasta")
    embeddings = get_fasta_embeddings("tests/test.fasta", Model, Tokenizer, "ProtT5")

    # Validate embedding dimensions
    assert len(embeddings) == 2
    assert len(embeddings[0]) == len(sequences[0][1])
    assert len(embeddings[1]) == len(sequences[1][1])
    assert len(embeddings[0][0]) == 1024
    assert len(embeddings[1][0]) == 1024