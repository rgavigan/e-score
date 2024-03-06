# Import Transformer models and tokenizers
from transformers import T5Tokenizer, T5EncoderModel
from transformers import AlbertModel, AlbertTokenizer
from transformers import BertModel, BertTokenizer
from transformers import XLNetModel, XLNetTokenizer
import esm

# Import PyTorch
import torch

# Retrieve the device (CPU or GPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))

torch.set_grad_enabled(False)

def ESM1b_initialize():
    print("Initializing ESM1b")
    
    ESM1b, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    ESM1b.eval()  # disables dropout for deterministic results

    return ESM1b, batch_converter

def ESM2_initialize():
    print("Initializing ESM2")

    ESM2, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    ESM2.eval()  # disables dropout for deterministic results

    return ESM2, batch_converter

def ProtT5_initialize():
    print("Initializing ProtT5")
    transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"

    T5 = T5EncoderModel.from_pretrained(transformer_link).to(device)

    # Run full-precision if using CPU, half-precision if GPU
    T5.float() if device == torch.device('cpu') else T5.half()

    # Set to evaluation model
    T5 = T5.eval()
    T5_tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)

    return T5, T5_tokenizer

def ProtBert_initialize():
    print("Initializing ProtBert")
    transformer_link = "Rostlab/prot_bert"

    Bert_tokenizer = BertTokenizer.from_pretrained(transformer_link, do_lower_case=False)
    Bert = BertModel.from_pretrained(transformer_link)

    Bert = Bert.to(device)
    Bert = Bert.eval()

    return Bert, Bert_tokenizer

def ProtAlbert_initialize():
    print("Initializing ProtAlbert")
    transformer_link = "Rostlab/prot_albert"

    Albert_tokenizer = AlbertTokenizer.from_pretrained(transformer_link, do_lower_case=False)
    Albert = AlbertModel.from_pretrained(transformer_link)

    Albert = Albert.to(device)
    Albert = Albert.eval()

    return Albert, Albert_tokenizer

def XLNet_initialize():
    """ Initializes the ProtXLNet Model """
    print("Initializing ProtXLNet")
    transformer_link = "Rostlab/prot_xlnet"

    XLNet_tokenizer = XLNetTokenizer.from_pretrained(transformer_link, do_lower_case=False)
    XLNet = XLNetModel.from_pretrained(transformer_link, mem_len= 1024)

    XLNet = XLNet.to(device)
    XLNet = XLNet.eval()

    return XLNet, XLNet_tokenizer

def get_model(model_name):
    """ Gets the model and tokenizer for the given model name.
    
    Args:
        model_name (str): Name of the model to get.

    Returns:
        Model, Model_Tokenizer: Model and tokenizer for the given model name.
    """
    Model, Tokenizer = None, None
    if model_name == "ProtT5":
        Model, Tokenizer = ProtT5_initialize()
    elif model_name == "ProtBert":
        Model, Tokenizer = ProtBert_initialize()
    elif model_name == "ProtAlbert":
        Model, Tokenizer = ProtAlbert_initialize()
    elif model_name == "ProtXLNet":
        Model, Tokenizer = XLNet_initialize()
    elif model_name == "ESM1b":
        Model, Tokenizer = ESM1b_initialize()
    elif model_name == "ESM2":
        Model, Tokenizer = ESM2_initialize()
    return Model, Tokenizer
