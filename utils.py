import umap
import hdbscan
from sklearn.metrics import silhouette_score
import config, plot
import numpy as np
import pandas as pd
from typing import Tuple, List
import random
import spacy
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import collections
import itertools
import argparse

nlp = spacy.load("en_core_web_sm")

def generate_clusters(embeddings: np.ndarray,
                      n_neighbors: int,
                      n_components: int, 
                      min_cluster_size: int,
                      random_state: int) -> hdbscan.HDBSCAN:
    """
    Args:
        embeddings (np.ndarray): sentence transformer embeddings
        n_neighbors (int)
        n_components (int)
        min_cluster_size (int)
        random_state (int, optional). Defaults to config.SEED.

    Returns:
        hdbscan.HDBSCAN: the clusters
    """
    
    umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors, 
                                n_components=n_components, 
                                metric='cosine', 
                                random_state=random_state)
                            .fit_transform(embeddings))

    clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                               metric='euclidean', 
                               gen_min_span_tree=True,
                               cluster_selection_method='eom').fit(umap_embeddings)

    return clusters


def score_clusters(X: np.ndarray, 
                   clusters: hdbscan.HDBSCAN, 
                   prob_threshold: float=0.05) -> Tuple[int, float, float, float]:
    """
    Args:
        X (np.ndarray): embeddings
        clusters (hdbscan.HDBSCAN): clusters
        prob_threshold (float, optional). Defaults to 0.05:float.

    Returns:
        Tuple[int, float, float, float]: label_count, cost, silhouette_coef, DBCV_score
    """

    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_num = len(cluster_labels)
    cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold)/total_num)
    silhouette_coef = silhouette_score(X, cluster_labels)
    DBCV_score = clusters.relative_validity_
    
    return label_count, cost, silhouette_coef, DBCV_score


def random_search(args: argparse.Namespace,
                  df: pd.DataFrame,  
                  space: dict, 
                  num_evals: int) -> pd.DataFrame:
    """
    Randomly search hyperparameter space and limited number of times and a summary of the results
    Args:
        args: argparse object
        df (pd.DataFrame): loaded dataframe
        space (dict[str, list]): search space
        num_evals (int): number of times to run the search

    Returns:
        pd.DataFrame: a summary of the results
    """
    
    results = []
    for i in range(num_evals):
        n_neighbors = random.choice(space['n_neighbors'])
        n_components = random.choice(space['n_components'])
        min_cluster_size = random.choice(space['min_cluster_size'])
        
        clusters = generate_clusters(df['emb1'], 
                                     n_neighbors=n_neighbors, 
                                     n_components=n_components, 
                                     min_cluster_size=min_cluster_size, 
                                     random_state=config.SEED)

        cluster_dict = {'label_st1': clusters}
        df_combined = combine_results(df[["prompt", "cleaned", "image"]], cluster_dict)
        agreement_scores = plot.create_agreement_plot(df_combined, args.topks)
        jaccard_similarity_scores = plot.create_jaccard_similarity_plot(df_combined, args.topks)
        label_count, cost, silhouette_coef = score_clusters(df['emb1'], clusters, prob_threshold=0.05)
        
        results.append([i, n_neighbors, n_components, min_cluster_size, 
                        label_count, cost, silhouette_coef, agreement_scores, jaccard_similarity_scores])

    print(results)    
    result_df = pd.DataFrame(results, columns=['run_id', 'n_neighbors', 'n_components', 'min_cluster_size', 
                                    'label_count', 'cost', 'silhouette_score', 'agreement_scores', 'jaccard_similarity_scores'])
    
    return result_df.sort_values(by='silhouette_score')


def grid_search(args: argparse.Namespace, , df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        args: argparse object
        df (pd.DataFrame): loaded dataframe

    Returns:
        pd.DataFrame: a summary of the results
    """
    
    results = []
    li = [config.N_NEIGHBORS, config.N_COMPONENTS, config.MIN_CLUSTER_SIZE]
    for configuration in itertools.product(*li):
        n_neighbors, n_components, min_cluster_size = configuration
        clusters = generate_clusters(df['emb1'],
                                        n_neighbors=n_neighbors,
                                        n_components=n_components,
                                        min_cluster_size=min_cluster_size,
                                        random_state=config.SEED)
        cluster_dict = {'label_st1': clusters}
        df_combined = combine_results(df[["prompt", "cleaned", "image"]], cluster_dict)
        agreement_scores = plot.create_agreement_plot(args, df_combined, args.topks)
        jaccard_similarity_scores = plot.create_jaccard_similarity_plot(args, df_combined, args.topks)
        label_count, cost, silhouette_coef, DBCV_score = score_clusters(df['emb1'], clusters, prob_threshold=0.05)
        
        results.append([n_neighbors, n_components, min_cluster_size, label_count, cost, 
                        silhouette_coef, DBCV_score, agreement_scores, jaccard_similarity_scores])
    
    print(results) 
    columns = ['n_neighbors', 'n_components', 'min_cluster_size', 'label_count', 'cost', 'silhouette_score', 
                        'DBCV_score', 'agreement_scores', 'jaccard_similarity_scores']
    result_df = pd.DataFrame(results, columns=columns)
    return result_df.sort_values(by='silhouette_score')


def most_common(lst: list, n_words: int) -> list:
    """
    Get most common words in a list of words
    
    Args:
        lst: list, each element is a word
        n_words: number of top common words to return
    
    Returns:
        counter.most_common(n_words): counter object of n most common words
    """
    counter = collections.Counter(lst)
    return counter.most_common(n_words)


def extract_labels(prompts: list, topk: int=1) -> str:
    """
    Extract labels from documents in the same cluster by concatenating most common verbs, objs, and nouns
    Args:
        prompts (list): prompts in the same cluster
        topk (int, optional): _description_. Defaults to 1.

    Returns:
        str: meta label
    """
    def flatten(A):
        rt = []
        for i in A:
            if isinstance(i,list): rt.extend(flatten(i))
            else: rt.append(i)
        return rt

    tokens = []
    for prompt in prompts:  
        doc = nlp(prompt)
        for token in doc:
            if not token.is_stop:
                tokens.append(token.text.lower())
    

    result = [word[0] for word in most_common(tokens, topk)] if tokens else []
    label = ' '.join(set(flatten(result)))
    return label


def combine_results(df_ground: pd.DataFrame, cluster_dict: dict) -> pd.DataFrame:
    """
    Returns dataframe of all documents and each model's assigned cluster

    Args:
        df_ground: dataframe of original documents with associated ground truth labels
        cluster_dict: dict, keys as column name for specific model and value as best clusters HDBSCAN object

    Returns:
        df_combined: dataframe of all documents with labels from best clusters for each model
    """

    df_combined = df_ground.copy()
    labels_array = np.empty((len(df_ground), len(cluster_dict)), dtype=np.int32)

    for idx, (key, value) in enumerate(cluster_dict.items()):
        labels_array[:, idx] = value.labels_

    df_combined[list(cluster_dict)] = labels_array
    
    return df_combined


def get_group(df: pd.DataFrame, category_col: str, category: int) -> pd.DataFrame:
    """
    Returns documents of a single category
    
    Args:
        df: pandas dataframe of documents
        category_col: str, column name corresponding to categories or clusters
        category: int, cluster number to return
    
    Returns:
        single_category: pandas dataframe with documents from a single category
    """
    single_category = df.loc[df[category_col] == category].copy()
    return single_category


def apply_and_summarize_labels(df: pd.DataFrame, category_col: str, topk: int=1) -> pd.DataFrame:
    """
    Assign groups to original documents and provide group counts

    Args:
        df: pandas dataframe of original documents of interest to cluster
        category_col: str, column name corresponding to categories or clusters
        topk: int, number of top words to extract from each cluster

    Returns:
        summary_df: pandas dataframe with model cluster assignment, number
                    of documents in each cluster and derived labels
    """
    
    numerical_labels = df[category_col].unique()
    
    # create dictionary of the numerical category to the generated label
    label_dict = {}
    for label in numerical_labels:
        current_category = list(get_group(df, category_col, label)['cleaned'])
        label_dict[label] = extract_labels(current_category, topk)
        
    # create summary dataframe of numerical labels and counts
    summary_df = (df.groupby(category_col)['cleaned'].count()
                    .reset_index()
                    .rename(columns={'cleaned':'count'})
                    .sort_values('count', ascending=False))
    
    # apply generated labels
    summary_df['label'] = summary_df.swifter.apply(lambda x: label_dict[x[category_col]], axis=1)
    
    return summary_df


def compute_image_pair_similarity(img1: Image.Image, 
                                  img2: Image.Image, 
                                  model: CLIPModel, 
                                  processor: CLIPProcessor) -> float:
    """
    Args:
        img1 (Image.Image): first image
        img2 (Image.Image): second image
        model (CLIPModel): CLIP Model
        processor (CLIPProcessor): CLIP Processor

    Returns:
        float: cosine similarity between the two embeddings
    """
    
    def get_image_embd(img: Image.Image) -> torch.Tensor:
        """
        Args:
            img (Image.Image): an image

        Returns:
            torch.Tensor: image embedding
        """
        processed_img = processor(
            text=None,
            images=img,
            return_tensors='pt'
        )['pixel_values'].to(config.DEVICE)

        img_embs = model.get_image_features(processed_img)
        return img_embs

    generated_emb = get_image_embd(img1)
    original_emb = get_image_embd(img2)
    
    # Calculate the cosine similarity between the embeddings
    similarity_score = torch.nn.functional.cosine_similarity(original_emb, generated_emb)
    return similarity_score.item()
