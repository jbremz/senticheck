from tqdm import tqdm_notebook as tqdm
import numpy as np
import os, glob
import pandas as pd

def corpus(file_list):
    '''
    Inputs:
    file_list - the list of paths of the text files to create the corpus
    
    Returns:
    A list of strings containing the corpus
    '''
    corpus = []

    for file_path in tqdm(file_list):
        with open(file_path) as f_input:
            sample = f_input.read()
            corpus.append(sample)

    return corpus

def makeDF(data_path):
    '''
    Inputs:
    data_path - the path of the directory containing the pos/neg directories
    samples - number of text samples to take for each language

    Returns:
    A pandas dataframe with the string and the language

    '''
    sentis = ['pos','neg']

    strings = np.array([])
    sentiments = np.array([])

    for sentiment in tqdm(sentis):
        file_list = glob.glob(os.path.join(data_path, sentiment,"*.txt"))
        the_corpus = corpus(file_list)
        strings = np.concatenate([strings, the_corpus])
        sentiments = np.concatenate([sentiments, [sentiment]*len(the_corpus)])
    
    return pd.DataFrame(np.array([strings, sentiments]).T, columns=['string','sentiment'])