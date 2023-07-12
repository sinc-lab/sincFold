# **sincFold**: a RNA folding prediction tool based on deep learning.

<p align="center">
<img src="abstract.png" alt="abstract">
</p>

```bibtex
@article{sincFold2023,
  title={sincFold: an end-to-end method to accurately fold RNA sequences with compressed short-long context encodings and binding restrictions},
  author={Leandro A. Bugnon and Leandro Di Persia and Matias Gerard and Jonathan Raad and Santiago Prochetto and Emilio Fenoy and Uciel Chorostecki and Georgina Stegmayer and Diego H. Milone},
  journal={under review},
  year={2023}
}
```

sincFold is a fast and accurate RNA secondary structure prediction method. It is an end-to-end approach that predict the nucleotides contact matrix using only the RNA sequence as input. The model is based on a residual neural network that can learn short and long context features from a small dataset, it can incorporate base-pair constraints and can guarantee the output structure to be valid.  Extensive experiments on several benchmark datasets were made, comparing sincFold against classical methods and new models based on deep learning. We demonstrate that sincFold achieves a very competitive performance in comparison with state-of-the-art methods.

A summary of results can be seen in [this notebook](results/summary.ipynb).

## Folding RNA sequences

We have a [webserver](https://sinc.unl.edu.ar/web-demo/sincfold/) running with the lattest version. This server admits one sequence at a time. Please follow the next
instructions if you want to run the model locally. We provide a model pretrained with validated  RNA datasets (archiveII, RNAstralign, urs-pdb). At the end you can find the instructions to replicate our cross-validation results from scratch.

## Install

This is a Python package. It is recommended to use virtualenv or conda to create a new enviroment and avoid breaking system packages.

To install the package, run:

    pip install sincfold

Alternativelly, you can clone the repository with

    git clone https://github.com/sinc-lab/sincFold
    cd sincFold/

and install with:

    pip install .

If you find issues with pytorch instalation, please refer to [the documentation](https://pytorch.org/). 

## Predicting sequences

To predict the secondary structure of a list of sequences, using the pretrained weights, use 
    
    sincFold pred sample/test.csv -o pred_file.csv

where sample/test.csv is a table with "id" and "sequence", and pred_file.csv adds the "base-pairs" found. Alternativelly, you can use standard fasta files

    sincFold pred sample/test.fasta -o pred_file.fasta

## Training and testing models

A new model can be trained using  
    
    sincFold train sample/train.csv -n 10 -o output_path

The option -n limits the maximum number of epochs to get a quick result. 

Then, a different test set can be evaluated with 

    sincFold test sample/test.csv -w output_path/weights.pmt

The model path (-w) is optional, if ommited the pretrained weights are used.

## Reproducing our results

You can run the complete train and test scheme using the following code (in this case set up benchmarkII and fold 0 data partition). 

```python
import os 
import pandas as pd 

out_path = f"working_path/"
os.mkdir(out_path)

# read dataset and predefined partitions
dataset = pd.read_csv("data/benchmarkII.csv", index_col="id")
partitions = pd.read_csv("data/benchmarkII_splits.csv")

dataset.loc[partitions[(partitions.fold_number==0) & (partitions.partition=="train")].id].to_csv(out_path + "train.csv")
dataset.loc[partitions[(partitions.fold_number==0) & (partitions.partition=="valid")].id].to_csv(out_path + "valid.csv")
dataset.loc[partitions[(partitions.fold_number==0) & (partitions.partition=="test")].id].to_csv(out_path + "test.csv")
```

then call the training and testing functions


    sincFold -d cuda train working_path/train.csv --valid_file working_path/valid.csv -o working_path/output/

    sincFold -d cuda test working_path/test.csv -w working_path/output/weights.pmt

Using a GPU for training is recommended (with the option '-d cuda'). The complete process may take about 3hs using a RTX A5000.