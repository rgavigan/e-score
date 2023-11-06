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
| ProtT5   | 1024   | Uniref50 |
| ProtBert     | 1024       | Uniref100 |
| ProtAlbert  | 4096     | Uniref100 |
| ProtXLNet    | 1024      |  Uniref100  |
| ESM1b  | 1280     | Uniref50 |
| ESM2   | 1280      | Uniref50 |


<!-- USAGE EXAMPLES -->
## Usage
**Input**: fasta file

**Purpose**: Computes the alignment for the given fasta file based on chosen parameters (see **Parameters and Descriptions**). Calls the `alignment_file_TXT` function.

**Result**: Outputs a text file in the `saving_add` directory containing the computed alignment, score, and alignment visualization.

### Parameters and Descriptions
| Parameter | Description |
| :---         |     :---:     | 
| saving_add   | Path to output directory   | 
| seqs_path    | Path to the FASTA file containing two protein sequences       | 
| scoring_type  | Model for embedding production: ProtT5, ESM2, ProtBert, ProtAlbert, ESM1b, ProtXLNet     | 
| alignment_type    | "Global-regular" or "Global-end-gap-free"     | 
| gap_penalty   |  **default** = -1; Recommended Values: -4, -3, -2, -1.5, -1, -0.5    |
| gap_extension_penalty   |**default** = -0.2; Recommended Values: -1, -0.8, -0.5, -0.3, -0.2, -0.1      | 

The output file will be named: 
`<fasta_file_name>_<scoring_type>_<alignment_type>_<gap_penalty>_<gap_extension_penalty>_Alignment.txt`

### Usage Example
```python
saving_add =  "./content/"
seqs_path = "tests/Test2.fasta"
scoring = "ProtT5" 
alignment_type = "Global-regular" 
gap_penalty = -1
gap_extension_penalty = -0.2

alignment_file_TXT(saving_add = saving_add , seqs_path = seqs_path, scoring = scoring, alignment_type = alignment_type,
                      gap_penalty = gap_penalty, gap_extension_penalty = gap_extension_penalty)
```

Output (Test2_ProtT5_Global-regular_-1_-0.2_Alignment.txt):

```
Seq 1 
>gi|464921|sp|P34981|TRFR_HUMAN
TILLVLIICGLGIVGNIMVVLVVMRTKHMRTPTNCYLVSLAVADLMVLVAAGLPNITDSIYGSWVYGYVGCLCITYLQYLGINASSCSITAFTIERYIAICHPIKAQFLCTFSRAKKIIIFVWAFTSLYCMLWFFLLDLNISTYKDAIVISCGYKISRNYYSPIYLMDFGVFYVVPMILATVLYGFIARILFLNPIPSDPKENSKTWKNDSTHQNTNLNVNTSNRCFNSTVSSRKQVTKMLAVVVILFALLWMPYRTLVVVNSFLSSPFQENWFLLFCRICIYLNSAINPVIYNLMS
Seq 2 
>gi|20455271|sp|Q9NSD7|R3R1_HUMAN
ISVVYWVVCALGLAGNLLVLYLMKSMQGWRKSSINLFVTNLALTDFQFVLTLPFWAVENALDFKWPFGKAMCKIVSMVTSMNMYASVFFLTAMSVTRYHSVASALKSHRTRGHGRGDCCGRSLGDSCCFSAKALCVWIWALAALASLPSAIFSTTVKVMGEELCLVRFPDKLLGRDRQFWLGLYHSQKVLLGFVLPLGIIILCYLLLVRFIADRRAAGTKGGAAVAGGRPTGASARRLSKVTKSVTIVVLSFFLCWLPNQALTTWSILIKFNAVPFSQEYFLCQVYAFPVSVCLAHSNSCLNPVLYCLVR

Alignment Type : Global-regular

Opening Gap Penalty : -1
Extension Gap Penalty : -0.2
Scoring System : ProtT5
Score : 141.1924964427947

Seq 1 : 1     TILLVLIICGLGIVGNIMVVLVVMRTK-HMRTPTNCYLVSLAVADLMVLVAAGLPNITDS    59
                      C LG  GN  V               N     LA  D               
Seq 2 : 1     ISVVYWVVCALGLAGNLLVLYLMKSMQGWRKSSINLFVTNLALTDFQFVLTLPFWAVENA    60

Seq 1 : 60    IYGSWVYGYVGCLCITYLQYLGINASSCSITAFTIERYIAICHPIKAQF-----------   108
                  W  G   C            AS    TA    RY       K              
Seq 2 : 61    LDFKWPFGKAMCKIVSMVTSMNMYASVFFLTAMSVTRYHSVASALKSHRTRGHGRGDCCG   120

Seq 1 : 109   ----LCTFSRAKKIIIFVWAFTSLYCMLWFFLLDLNISTYKDAIVISCGYKI----SRNY   160
                        AK      WA   L                          K         
Seq 2 : 121   RSLGDSCCFSAKALCVWIWALAALASLPSAIFSTTVKVMGEELCLVRFPDKLLGRDRQFW   180

Seq 1 : 161   YSPIYLMDFGVFYVVPMILATVLYGFIARILFLNPIPSDPKENSKTWKNDSTHQNTNLNV   220
                           V P       Y    R           K                   
Seq 2 : 181   LGLYHSQKVLLGFVLPLGIIILCYLLLVRFIADRR-AAGTKGG---------------AA   224

Seq 1 : 221   NTSNRCFNSTVSSRKQVTKMLAVVVILFALLWMPYRTLVV---VNSFLSSPF------QE   271
                  R           VTK    VV  F L W P   L        F   PF        
Seq 2 : 225   VAGGRPTGASARRLSKVTKSVTIVVLSFFLCWLPNQALTTWSILIKFNAVPFSQEYFLCQ   284

Seq 1 : 272   NWFLLFCRICIYLNSAINPVIYNLMS   297
                           NS  NPV Y L  
Seq 2 : 285   VYAFPVSVCLAHSNSCLNPVLYCLVR   310
```
