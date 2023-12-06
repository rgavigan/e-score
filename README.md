<div align="center">
    <h3 align="center">
        E-Score
    </h3>
    <h4 align="center">
        Aligning Protein Sequences Using Embedding Scores 
    </h4>
    <h4 align="center">
        Undergraduate Thesis - Analyzing Results
    </h4>
</div>

<!-- ABOUT -->
## About
The E-Score project focuses on computing Global-regular and Global-end-gap-free alignment between any two protein sequences using their embedding vectors computed by state-of-art pre-trained models. 

Instead of a fixed score between two pairs of amino acids(like BLOSUM matrices), the cosine similarity is calculated between the embedding vectors of two amino acids and used as the context-dependent score.

## Thesis and Proposal
* [Link to Thesis Paper](https://rileygavigan.com/e-score-thesis.pdf)
* [Link to Proposal Paper](https://rileygavigan.com/e-score-proposal.pdf)

## System Requirements
<b>Recommended Python Version:</b> 3.10
<br />
<b>Recommended RAM:</b> 24GB
* Each of the models needs about 8-12GB of RAM and as sequence length increases, RAM requirements do. 

## Installation
General:
```sh
# Virtual Environment Creation
python -m venv venv
source ./venv/bin/activate

# Install Requirements
pip install -r requirements.txt
```
SageMaker Notebook (Jupyter Lab):
* Run the notebook with `!pip install -r requirements.txt` in the **Imports** code block.

## Models
| Models | Embedding Dim | Pre-trained on
| :---         |     :---:     |  :---:     | 
| ProtT5   | 1024   | UniRef50 |
| ProtBert     | 1024       | UniRef100 |
| ProtAlbert  | 4096     | UniRef100 |
| ProtXLNet    | 1024      |  UniRef100  |
| ESM1b  | 1280     | UniRef50 |
| ESM2   | 1280      | UniRef50 |