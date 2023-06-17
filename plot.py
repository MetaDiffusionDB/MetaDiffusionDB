import umap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import hdbscan
import torch
from PIL import Image
import os
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
import config, utils
from typing import List, Tuple
import argparse
import itertools


def plot_clusters(embeddings: np.ndarray, clusters: hdbscan.HDBSCAN, n_neighbors: int=15, min_dist: float=0.1) -> None:
    """_summary_

    Args:
        embeddings (np.ndarray): sentence transformer embeddings
        clusters (hdbscan.HDBSCAN): HDSCAN object
        n_neighbors (int, optional): UMAP hyperparameter n_neighbors. Defaults to 15.
        min_dist (float, optional): UMAP hyperparameter min_dist. Defaults to 0.1.
    """
    umap_data = umap.UMAP(n_neighbors=n_neighbors, 
                          n_components=2, 
                          min_dist = min_dist,  
                          random_state=SEED).fit_transform(embeddings)

    point_size = 100.0 / np.sqrt(len(embeddings))
    
    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    result['labels'] = clusters.labels_

    fig, ax = plt.subplots(figsize=(14, 8))
    outliers = result[result.labels == -1]
    clustered = result[result.labels != -1]
    plt.scatter(outliers.x, outliers.y, color = 'lightgrey', s=point_size)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=point_size, cmap='jet')
    plt.show()
    fig.savefig(os.path.join(config.PLOT_DIR, f"umap_{len(config.PART_ID)}k.png"))


def create_sample_subplots(df: pd.DataFrame, cluster_num: int, part_id: int) -> None:
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(24, 12))
    fig.subplots_adjust(hspace=0.5)

    # Flatten the axes array
    axes = axes.flatten()

    # Loop over each subplot and display the image
    for i, ax in enumerate(axes):
        # Turn off axis ticks and labels
        ax.axis('off')
        # Display the image in the subplot
        img_name = df[df["label_st1"]==cluster_num]["image"].tolist()[i]
        img = Image.open(os.path.join(config.DATASET_DIR, os.path.join(f'part-{part_id:06}', img_name)))
        ax.imshow(img)

    # Show and save the plot
    plt.show()
    plt.savefig(f"part-{part_id:06}_cluster-{cluster_num:02}.png")


def plot_distance_topk(args: argparse.Namespace,
                       df_combined: pd.DataFrame, 
                       model: CLIPModel,
                       processor: CLIPProcessor,
                       pipe: StableDiffusionPipeline, 
                       latents: torch.Tensor, 
                       part_id: int) -> None:
    """
    Args:
        args (argparse.Namespace): arguments
        df_combined (pd.DataFrame): dataframe with labels
        model (CLIPModel): CLIP Model
        processor (CLIPProcessor): CLIP Processor
        pipe (StableDiffusionPipeline): stable diffusion pipeline
        latents (torch.Tensor): stable diffusion latents
        part_id (int): part id of the dataset
    """

    def helper_plot(similarities: List[List[float]]) -> None:
        """
        Args:
            similarities (List[List[float]]): list of similarities
        """

        # sim_types = ["max", "min", "avg"]
        sim_types = ["avg"]
        for similarity, sim_type in zip(similarities, sim_types):
            plt.plot(args.topks,  [1 - i for i in similarity], label=sim_type)
        
        plt.legend()
        plt.title("Distance/TopK plot")
        plt.xlabel("TopK")
        plt.ylabel("1 - Cosine Similarity")
        if not os.path.exists(os.path.join(config.PLOT_DIR, f"results_{len(config.PART_ID)}k")):
            os.makedirs(os.path.join(config.PLOT_DIR, f"results_{len(config.PART_ID)}k"))
        
        config_name = '_'.join(list(map(str, (config.best_cluster.values()))))
        filename = f"distance_topk_{args.inp_sample_size}_{args.sd_sample_size}_search_space{args.search_space}_{config_name}.png"
        absolute_path = os.path.join(config.PLOT_DIR, f"results_{len(config.PART_ID)}k")
        plt.savefig(os.path.join(absolute_path, filename))
    

    max_similarities_by_topk, min_similarities_by_topk, avg_similarities_by_topk = [], [], []
    for k in args.topks:
        cluster_summary = utils.apply_and_summarize_labels(df_combined, 'label_st1', topk=k)
        df_all = cluster_summary.merge(df_combined, on='label_st1')
        
        total_sim_min, total_sim_max, total_sim_avg = 0, 0, 0
        # count the number of data points in each cluster expect noise         
        num_clusters = args.inp_sample_size * (len(df_all["label_st1"].unique()) - 1) # without noise 

        for label in df_all["label_st1"].unique()[1:]:
            temp = df_all[df_all["label_st1"]==label]
            temp = temp.sample(n=args.inp_sample_size, random_state=config.SEED, replace=True)
            # form a matrix of size (sample_size, sd_sample_size)
            sim_matrix = np.zeros((len(temp), args.sd_sample_size))
            meta_prompt = temp["label"].tolist()[0]
            
            with torch.autocast("cuda"):
                    generated_img = pipe([meta_prompt]*args.sd_sample_size, 
                                         latents=latents, 
                                         num_inference_steps=config.NUM_INFERENCE_STEPS,
                                         height=config.HEIGHT,
                                         width=config.WIDTH)
            
            # Iterate over the images in the pipeline output
            for i, image in enumerate(generated_img.images):
                # Save the image to a file
                path = os.path.join(config.PLOT_DIR, f'images_evolution_k_{config.DEVICE}/top_{k}/label_{label}/sd_generated/')
                if not os.path.exists(path):
                    os.makedirs(path)
                    
                file_path = os.path.join(config.PLOT_DIR, f"images_evolution_k_{config.DEVICE}/top_{k}/label_{label}/sd_generated/image_{i}.png")
                image.save(file_path)
            
            # save meta_label in a readme file
            readme_file_path = os.path.join(config.PLOT_DIR, f'images_evolution_k_{config.DEVICE}/top_{k}/label_{label}/readme.txt')
            if not os.path.exists(readme_file_path):
                with open(readme_file_path, 'w') as f:
                    f.write(f"meta_label: {meta_prompt}")
            
            # iterate over the sample
            for i in range(len(temp)):
                img_name = temp["image"].tolist()[i]
                original_img = Image.open(os.path.join(config.DATASET_DIR, img_name))
                # Save the image to a file
                path = os.path.join(config.PLOT_DIR, f'images_evolution_k_{config.DEVICE}/top_{k}/label_{label}/original_dataset/')
                if not os.path.exists(path):
                    os.makedirs(path)
                    
                file_path = os.path.join(config.PLOT_DIR, f"images_evolution_k_{config.DEVICE}/top_{k}/label_{label}/original_dataset/image_{i}.png")
                original_img.save(file_path)
                    
                # compute pairwise similarity between the original image and generated images
                for j in range(args.sd_sample_size):
                    sim_matrix[i][j] = utils.compute_image_pair_similarity(original_img, generated_img[0][j], model, processor)
                    
            
            for i in range(len(temp)):    
                # compute max, min, mean row-wise
                total_sim_max += np.max(sim_matrix[i])
                total_sim_min += np.min(sim_matrix[i])
                total_sim_avg += np.mean(sim_matrix[i])
                                

        # compute average similarity over all clusters
        avg_similarities_by_topk.append(total_sim_avg/num_clusters)
        min_similarities_by_topk.append(total_sim_min/num_clusters)
        max_similarities_by_topk.append(total_sim_max/num_clusters)

    # plot the results, distance/topk
    # helper_plot([max_similarities_by_topk, min_similarities_by_topk, avg_similarities_by_topk])
    config_name = '_'.join(list(map(str, (config.best_cluster.values()))))
    with open(os.path.join(config.PLOT_DIR, f"max_similarities_by_topk{args.inp_sample_size}_{args.sd_sample_size}_{config_name}.txt"), "w") as f:
        for item in max_similarities_by_topk:
            # write each item on a new line
            f.write("%s\n" % item)
    
    with open(os.path.join(config.PLOT_DIR, f"min_similarities_by_topk{args.inp_sample_size}_{args.sd_sample_size}_{config_name}.txt"), "w") as f:
        for item in min_similarities_by_topk:
            # write each item on a new line
            f.write("%s\n" % item)
    
    with open(os.path.join(config.PLOT_DIR, f"avg_similarities_by_topk{args.inp_sample_size}_{args.sd_sample_size}_{config_name}.txt"), "w") as f:
        for item in avg_similarities_by_topk:
            # write each item on a new line
            f.write("%s\n" % item)
        

def create_agreement_plot(args,
                          df_combined: pd.DataFrame, 
                          show: bool=False, 
                          create_hist: bool=False,
                          to_save: bool=False) -> None:
    """
    Args:
        args (argparse.Namespace): arguments
        df_combined (pd.DataFrame): dataframe with labels
        show (bool, optional): show the plot. Defaults to False.
        create_hist (bool, optional): create histogram plot. Defaults to False.
        to_save (bool, optional): save the plot. Defaults to False.
    """
    results = []
    aggregator = lambda meta_p, p: 100 * len(set(meta_p.split()).intersection(p.split())) / (len(meta_p.split()) + 1e-8)
    for topk in args.topks:
        cluster_summary = utils.apply_and_summarize_labels(df_combined, 'label_st1', topk=topk)
        df_all = cluster_summary.merge(df_combined, on='label_st1')
        # we compute the metric for each cluster and then average over all clusters
        grouped_df = df_all.groupby('label_st1')
        num_groups = len(grouped_df)
        aggregate_results = 0
        hist_plot_results = []
        
        for group_label, group_df in grouped_df:
            if group_label != -1:
                meta_prompts = group_df.iloc[0]['label']
                prompts = group_df['cleaned'].values

                # Calculate the metric using vectorized operations
                metrics = np.vectorize(aggregator)(meta_prompts, prompts)
                avg_metric = np.mean(metrics)

                hist_plot_results.append(avg_metric)
                aggregate_results += avg_metric


        # plot the histogram
        if create_hist:
            plt.figure()
            plt.hist(hist_plot_results, bins=num_groups)
            plt.savefig(os.path.join(config.PLOT_DIR, f"agreement_histogram_{topk}.png"))
            
        # Compute the final average metric over all the groups
        final_avg_metric = aggregate_results / (num_groups - 1)
        results.append(final_avg_metric)
        print("Average metric value: ", final_avg_metric)

    if to_save:
        # plot the results, agreement/topk, save the plot
        plt.figure()
        plt.plot(topks, results) 
        plt.title("Agreement/TopK plot")
        plt.xlabel("TopK")
        plt.ylabel("Aggrement")
        if not os.path.exists(os.path.join(config.PLOT_DIR, f"results_{len(config.PART_ID)}k")):
            os.makedirs(os.path.join(config.PLOT_DIR, f"results_{len(config.PART_ID)}k"))
            
        absolute_path = os.path.join(config.PLOT_DIR, f"results_{len(config.PART_ID)}k")
        plt.savefig(os.path.join(absolute_path, f"agreement_topk_{args.run_name}.png"))
        
    if show:
        plt.show()
    
    return results


def create_jaccard_similarity_plot(args,
                                   df_combined: pd.DataFrame, 
                                   show: bool=False,
                                   to_save: bool=False) -> None:
    """
    Args:
        args (argparse.Namespace): arguments
        df_combined (pd.DataFrame): dataframe with labels
        show (bool, optional): show the plot. Defaults to False.
        to_save (bool, optional): save the plot. Defaults to False.
    """
    def jaccard_similarity(str1: str, str2: str) -> float:
        """Compute the jaccard similarity between two strings
        Args:
            str1 (str): first string
            str2 (str): second string
    
        Returns:
            float: jaccard similarity
        """
        list1, list2 = str1.split(), str2.split()
        intersection = len(set(list1).intersection(list2))
        union = (len(set(list1)) + len(set(list2))) - intersection
        return float(intersection) / union

    results = []
    for topk in topks:
        cluster_summary = utils.apply_and_summarize_labels(df_combined, 'label_st1', topk=topk)
        df_all = cluster_summary.merge(df_combined, on='label_st1')
        unique_meta_labels = df_all["label"].unique()
        n_combinations = len(list(itertools.combinations(unique_meta_labels, 2)))
        jaccard_per_k = 0 
        for combination in itertools.combinations(unique_meta_labels, 2):
            jaccard_per_k += jaccard_similarity(*combination)
            
        results.append(jaccard_per_k/n_combinations)
    
    if to_save:
        plt.figure()
        plt.plot(topks, results) 
        plt.title("Jaccard similarity/TopK plot")
        plt.xlabel("TopK")
        plt.ylabel("Jaccard similarity")
        if not os.path.exists(os.path.join(config.PLOT_DIR, f"results_{len(config.PART_ID)}k")):
            os.makedirs(os.path.join(config.PLOT_DIR, f"results_{len(config.PART_ID)}k"))
            
        absolute_path = os.path.join(config.PLOT_DIR, f"results_{len(config.PART_ID)}k")
        plt.savefig(os.path.join(absolute_path, f"jaccard_topk_{args.run_name}.png"))
        
    if show:
        plt.show()

    return results