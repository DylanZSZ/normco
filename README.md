# Requirements
conda create -n syn python=3.8  
conda activate syn  
conda install numpy tqdm scikit-learn  
pip install Levenshtein  



# Datasets
The datasets are from <http://www.obofoundry.org/>. This website consists hundred of sub datasets which list the [Term] information and its synonym labels. The synonym entries are labeled data that could be utilized for expermients.  
We can use ontologies.jsonld to catch up all the datasets on the website and filter the useful ones later.  
Also, for convenience, I implement the extraction of synonym entries for every Term on single dataset and the split of dataset(under two different settings).  
See data_process.py for more details.

# Run

To run the program, direct yourself under ```/code ``` and run bash script run_normco.sh






