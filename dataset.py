from multiprocessing import Pool
import ujson
import pandas as pd 
import swifter
import config, plot
import os
from urllib.request import urlretrieve
import shutil
import re
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import Tuple
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def preprocess_text(text: str, remove_stopwords: bool=False, stemming: bool=True) -> str:
    """
    Args:
        text (str): the input text you want to clean
        remove_stopwords (bool): whether or not to remove stopwords
        stemming (bool): whether or not to stem the words
    Returns:
        str: the cleaned text
    """

    # remove links
    text = re.sub(r"http\S+", "", text)
    # remove special chars and keeping numbers
    text = re.sub("[^A-Za-z0-9]+", " ", text)

    # remove stopwords
    if remove_stopwords:
        # 1. tokenize
        tokens = nltk.word_tokenize(text)
        # stemming
        if stemming:
            ps = PorterStemmer()
            tokens = [ps.stem(token) for token in tokens]

        # 2. check if stopword
        tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
        # 3. join back together
        text = " ".join(tokens)
        
    # return text in lower case and stripped of whitespaces
    text = text.lower().strip()
    return text


def load_json(part_id: int) -> list:
    """
    Args:
        part_id (int): part id of the dataset

    Returns:
        list: list of tuples (key, text)
    """
    part_id_path = f'part-{part_id:06}/part-{part_id:06}.json'
    part_data = ujson.load(open(os.path.join(config.DATASET_DIR, part_id_path), encoding='utf-8'))
    rows = []
    for key in part_data:
        rows.append((os.path.join(f'part-{part_id:06}', key), part_data[key]['p']))
        
    return rows
        

def download_and_extract(part_id: int) -> None:
    """
    Args:
        part_id (int): part id of the dataset
    """
    if not os.path.exists(os.path.join(config.DATASET_DIR, f'part-{part_id:06}')):
        part_url = f'https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images/part-{part_id:06}.zip'
        zip_file_name = os.path.join(config.DATASET_DIR, f'part-{part_id:06}.zip')
        urlretrieve(part_url, zip_file_name)
        # unzip the file
        shutil.unpack_archive(zip_file_name, os.path.join(config.DATASET_DIR, f'part-{part_id:06}'))
        

def load_dataset(stemming=True) -> pd.DataFrame:
    """
    Returns:
        pd.DataFrame: the cleaned dataset combined with embeddings
    """
    with Pool(processes=8) as pool:
        # download and extract files in parallel
        pool.map(download_and_extract, config.PART_ID)  
        # load JSON files in parallel and yield rows
        rows = [item for sublist in pool.map(load_json, config.PART_ID) for item in sublist]

    metadata_df = pd.DataFrame(rows, columns=['image', 'prompt'])
    # clean the text
    metadata_df['cleaned'] = metadata_df['prompt'].swifter.apply(lambda x: preprocess_text(x, remove_stopwords=True, stemming=stemming))
    # sentence transformer models
    model_st1 = SentenceTransformer('all-mpnet-base-v2')
    # model_st2 = SentenceTransformer('all-MiniLM-L6-v2')
    # model_st3 = SentenceTransformer('paraphrase-mpnet-base-v2')
    # get the embeddings
    metadata_df['emb1'] = metadata_df['cleaned'].swifter.apply(lambda x: model_st1.encode(x))
    # metadata_df['emb2'] = metadata_df['cleaned'].swifter.apply(lambda x: model_st2.encode(x))
    # metadata_df['emb3'] = metadata_df['cleaned'].swifter.apply(lambda x: model_st3.encode(x))
    
    return metadata_df
