## Pathway-Informed Spectral Embeddings for Explainable Cancer Genomics

This repository contains the code for our project,  
**"Pathway-Informed Spectral Embeddings for Explainable Cancer Genomics,"**.


![Alt text](images/__overview_framework.jpeg)

## Data Source

The dataset is obtained from the following sources:

- **[reactome pathway database](https://reactome.org/)**  

TREACTOME is an open-source, open access, manually curated and peer-reviewed pathway database.


## Setup and Get Started

1. Install the required dependencies:
   - `pip install -r requirements.txt`

2. Activate your Conda environment:
   - `conda activate gnn`

3. Install PyTorch:
   - `conda install pytorch torchvision torchaudio -c pytorch`

4. Install the necessary Python packages:
   - `pip install pandas`
   - `pip install py2neo pandas matplotlib scikit-learn`
   - `pip install tqdm`
   - `pip install seaborn`

5. Install DGL:
   - `conda install -c dglteam dgl`

6. Download the data from the built gene association graph using the link below and place it in the `data/multiomics_meth/` directory before training:
   - [Download Gene Association Data](https://drive.google.com/file/d/1l7mbTn2Nxsbc7LLLJzsT8y02scD23aWo/view?usp=sharing)

7. To train the model, run the following command:
   - `python main.py --model_type ACGNN --net_type CPDB --score_threshold 0.99 --learning_rate 0.001 --num_epochs 200`

