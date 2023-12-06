# Import PyTorch
import torch

from Bio import SeqIO
import re
from src.models import get_model

# Retrieve the device (CPU or GPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_grad_enabled(False)

def load_fasta(path):
    """ Loads the two sequences from the FASTA file """
    fasta_sequences = SeqIO.parse(open(path),'fasta')
    
    # Array of (name, sequence) tuples
    sequences = []

    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        sequences.append((name, sequence.upper()))

    return sequences

def get_embeddings_ESM1b(ESM1b, batch_converter, sequences, n):
    """ Generates Embeddings from ESM1b Model for sequences """
    sequences = sequences[:n]
    data = [("" , sequences[0])]

    _, _, batch_tokens = batch_converter(data)

    # Extract per-residue representations
    with torch.no_grad():
        results = ESM1b(batch_tokens, repr_layers=[33], return_contacts= False)
    token_representations = results["representations"][33]

    final_embeddings = []
    for i in range(len(token_representations)):
        final_embeddings.append(token_representations[i][1:-1])

    return final_embeddings

def get_embeddings_ESM2(ESM2, batch_converter, sequences, n):
    """ Generates Embeddings from ESM2 Model for sequences """
    sequences = sequences[:n]
    data = [("" , sequences[0])]

    _, _, batch_tokens = batch_converter(data)

    # Extract per-residue representations
    with torch.no_grad():
        results = ESM2(batch_tokens, repr_layers=[33], return_contacts= False)
    token_representations = results["representations"][33]

    final_embeddings = []
    for i in range(len(token_representations)):
        final_embeddings.append(token_representations[i][1:-1])

    return final_embeddings

def get_embeddings_T5(T5, tokenizer, sequences, n):
    """ Generates Embeddings from ProtT5 Model for sequences """
    sequence_examples = sequences[:n]

    # Replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

    # Tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding= True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # Generate embeddings
    print("Generating T5 Embeddings")
    with torch.no_grad():
        embedding_repr = T5(input_ids=input_ids, attention_mask=attention_mask)

    last_layer_repr = embedding_repr.last_hidden_state
    final_embeddings = []
    for i in range(len(last_layer_repr)):
        final_embeddings.append(last_layer_repr[i , :len(sequences[i])])

    return final_embeddings

def get_embeddings_ProtBert(Bert , tokenizer , sequences , n):
    """ Generates Embeddings from ProtBert Model for sequences """
    sequence_examples = sequences[:n]

    # this will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

    ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, pad_to_max_length=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():
        embedding = Bert(input_ids=input_ids,attention_mask=attention_mask)[0]

    final_embeddings = []
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][1:seq_len-1]
        final_embeddings.append(seq_emd)

    return final_embeddings

def get_embeddings_ProtAlbert(Albert, Albert_tokenizer, sequences, n):
    """ Generates Embeddings from ProtAlbert Model for sequences """
    sequences = [" ".join(re.sub(r"[UZOB]", "X", sequence)) for sequence in sequences]
    ids = Albert_tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding = 'longest')
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():
        embedding = Albert(input_ids = input_ids , attention_mask = attention_mask)[0]

    features = []
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][1 : seq_len - 1]
        features.append(seq_emd)

    return features

def get_embeddings_XLNet(XLNet, XLNet_tokenizer, sequences, n):
    """ Generates Embeddings from ProtXLNet Model for sequences """
    sequences = [" ".join(re.sub(r"[UZOBX]" , "<unk>", sequence)) for sequence in sequences]
    ids = XLNet_tokenizer.batch_encode_plus(sequences, add_special_tokens = True, padding = 'longest')
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():
        output = XLNet(input_ids = input_ids , attention_mask = attention_mask)
        embedding = output.last_hidden_state

    features = []
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        padded_seq_len = len(attention_mask[seq_num])
        seq_emd = embedding[seq_num][padded_seq_len - seq_len : padded_seq_len - 2]
        features.append(seq_emd)

    return features

def get_pair_embeddings(sequence_1, sequence_2, Model = None, Model_tokenizer = None, model_name = "ProtT5"):
    """ Gets the embeddings for a pair of sequences, sequence_1 and sequence_2, for the provided model name
    
    Args:
        sequence_1: the first protein sequence
        sequence_2: the second protein sequence
        Model: the model
        Model_tokenizer: the model's tokenizer
        model_name: the model's name
    
    Returns:
        embeddings[]: the resulting embeddings
    """
    embeddings = []

    if model_name == 'ProtT5':
        embeddings = get_embeddings_T5(Model, Model_tokenizer, [sequence_1, sequence_2], 2)
    elif model_name == 'ProtBert':
        embeddings = get_embeddings_ProtBert(Model, Model_tokenizer, [sequence_1, sequence_2], 2)
    elif model_name == 'ProtAlbert':
        embeddings = get_embeddings_ProtAlbert(Model, Model_tokenizer, [sequence_1, sequence_2], 2)
    elif model_name == 'ProtXLNet':
        embeddings = get_embeddings_XLNet(Model, Model_tokenizer, [sequence_1, sequence_2], 2)
    elif model_name == 'ESM1b':
        embeddings = get_embeddings_ESM1b(Model, Model_tokenizer, [sequence_1, sequence_2], 2)
    elif model_name == 'ESM2':
        embeddings = get_embeddings_ESM2(Model, Model_tokenizer, [sequence_1, sequence_2], 2)

    # Convert tensors to numpy arrays
    embeddings = [embedding.cpu().numpy() for embedding in embeddings]

    return embeddings

def get_fasta_embeddings(fasta_file, Model = None, Model_tokenizer = None, model_name = "ProtT5"):
    """ Gets the embeddings for a given fasta file for the provided model name
    
    Args:
        fasta_file: the fasta file to get embeddings for
        Model: the model
        Model_tokenizer: the model's tokenizer
        model_name: the model's name
    
    Returns:
        embeddings[]: the resulting embeddings
    """
    sequences = load_fasta(fasta_file)
    embeddings = []

    if model_name == 'ProtT5':
        embeddings = get_embeddings_T5(Model, Model_tokenizer, [sequence[1] for sequence in sequences], len(sequences))
    elif model_name == 'ProtBert':
        embeddings = get_embeddings_ProtBert(Model, Model_tokenizer, [sequence[1] for sequence in sequences], len(sequences))
    elif model_name == 'ProtAlbert':
        embeddings = get_embeddings_ProtAlbert(Model, Model_tokenizer, [sequence[1] for sequence in sequences], len(sequences))
    elif model_name == 'ProtXLNet':
        embeddings = get_embeddings_XLNet(Model, Model_tokenizer, [sequence[1] for sequence in sequences], len(sequences))
    elif model_name == 'ESM1b':
        embeddings = get_embeddings_ESM1b(Model, Model_tokenizer, [sequence[1] for sequence in sequences], len(sequences))
    elif model_name == 'ESM2':
        embeddings = get_embeddings_ESM2(Model, Model_tokenizer, [sequence[1] for sequence in sequences], len(sequences))

    # Convert tensors to numpy arrays
    embeddings = [embedding.cpu().numpy() for embedding in embeddings]
    
    return embeddings

if __name__ == "__main__":
    # Load all FASTA sequences from data/finetuning/cd00012.fasta as a test
    sequences = load_fasta("data/finetuning/cd00012.fasta")
    for sequence in sequences:
        print(sequence)