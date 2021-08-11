# Requirements
conda create -n syn python=3.8  
conda activate syn  
conda install numpy tqdm scikit-learn  
conda install nltk
conda install networkx
conda install spacy
conda install pandas
conda inst
pip install Levenshtein  


# Datasets
The datasets are from <http://www.obofoundry.org/>. This website consists hundred of sub datasets which list the [Term] information and its synonym labels. The synonym entries are labeled data that could be utilized for expermients.  
We can use ontologies.jsonld to catch up all the datasets on the website and filter the useful ones later.  
Also, for convenience, I implement the extraction of synonym entries for every Term on single dataset and the split of dataset(under two different settings).  
See data_process.py for more details.

# Run

To run the program, direct yourself under ```/code ``` and run bash script ```run_normco.sh```. You can change parameter settings in the script as well.

# Pipeline

The main process driving the entire experiment is in run_normco.py. It first prepares the data and pre-processes the data with the data generator coded in data_generator.py. Data pre-processing is done in 2 stages where the first is the same as syn (concept & mention extraction, graph construction and splitting), and the second stage is where we fit the data into this specific NormCo pipeline. Note that NormCo uses different datasets with diverse stuctures but we are using the dictionary data all across the pipeline. However, as some nodes may not have enough context information, we treat these as "isolated points" and use only the 'Summation Model' part of NormCo model to learn their embeddings. It is also done in the same way during validation and testing.

# Notes



# TODOs





