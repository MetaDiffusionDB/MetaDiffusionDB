import numpy as np
import random
from transformers import CLIPProcessor, CLIPModel, set_seed
from diffusers import StableDiffusionPipeline
import config, dataset, utils, plot
import torch
import argparse
import os
import pandas as pd

np.random.seed(config.SEED)
random.seed(config.SEED)
set_seed(config.SEED)


def parse_args():
    parser = argparse.ArgumentParser()                  
    parser.add_argument('--topks', nargs='+', type=int, help='topk values to compare')
    parser.add_argument('--plot_types', nargs='+', type=str, help='plot types to generate')
    parser.add_argument('--search_space', type=int, help='number of times to run the search', default=1)
    parser.add_argument('--inp_sample_size', type=int, help='number of samples to take from each cluster', default=2)
    parser.add_argument('--sd_sample_size', type=int, help='number of generated images by stable diffusion', default=2)
    parser.add_argument('--parameter-tuning', action='store_true', help='whether to run parameter tuning')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # loading the dataset
    df = dataset.load_dataset(stemming=True)
    print("Dataset loaded")
    print("Length of dataset: ", len(df))
    df = df[df['cleaned'].astype(bool)]
    print("Length of dataset after removing empty text: ", len(df))
    
    # grid search
    if args.parameter_tuning:
        print("Running grid search...")
        space = {
        'n_neighbors': config.N_NEIGHBORS,
        'n_components': config.N_COMPONENTS,
        'min_cluster_size': config.MIN_CLUSTER_SIZE,
        'random_state': config.SEED}
        
        grid_search_results = utils.grid_search(args, df)
        # save the results
        grid_search_results.to_csv(os.path.join(config.PLOT_DIR, f"grid_search_results_{len(config.PART_ID)}k.csv"), index=False)
        
    if args.plot_types:
        config_name = '_'.join(list(map(str, (config.BEST_CLUSTER.values()))))
        if not os.path.exists(os.path.join(config.PLOT_DIR, f"df_combined_{len(config.PART_ID)}k_{config_name}.csv")):
            best_cluster = utils.generate_clusters(df['emb1'].tolist(), 
                                                n_neighbors=config.BEST_CLUSTER['n_neighbors'], 
                                                n_components=config.BEST_CLUSTER['n_components'], 
                                                min_cluster_size=config.BEST_CLUSTER['min_cluster_size'], 
                                                random_state=config.SEED)
            
            cluster_dict = {'label_st1': best_cluster}
            df_combined = utils.combine_results(df[["prompt", "cleaned", "image"]], cluster_dict)
            df_combined.to_csv(os.path.join(config.PLOT_DIR, f"df_combined_{len(config.PART_ID)}k_{config_name}.csv"), index=False)
    
        else:
            print("Loading combined dataframe...")
            df_combined = pd.read_csv(os.path.join(config.PLOT_DIR, f"df_combined_{len(config.PART_ID)}k_{config_name}.csv"))
            
        # plotting
        for plot_type in args.plot_types:
            if plot_type == 'distance':
                # stable diffusion
                if config.DEVICE == "cuda":
                    pipe = StableDiffusionPipeline.from_pretrained(config.STABLE_DIFFUSION_ID, revision="fp16", torch_dtype=torch.float16)
                
                else:
                    pipe = StableDiffusionPipeline.from_pretrained(config.STABLE_DIFFUSION_ID)
                                                                   
                # disabling nsfw
                pipe.safety_checker = lambda images, clip_input: (images, False)
                pipe = pipe.to(config.DEVICE)

                # Diffusiondb dataset
                model = CLIPModel.from_pretrained(config.CLIP_MODEL_ID).to(config.DEVICE)
                # CLIP processor
                processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_ID)

                # reproducible latent space for stable diffusion
                generator = torch.Generator(device=config.DEVICE)
                latents = None
                seeds = []
                for _ in range(args.sd_sample_size):
                # Get a new random seed, store it and use it as the generator state
                    seed = generator.seed()
                    seeds.append(seed)
                    generator = generator.manual_seed(seed)
                    
                    image_latents = torch.randn(
                        (1, pipe.unet.in_channels, config.HEIGHT // 8, config.WIDTH // 8),
                        generator=generator,
                        device=config.DEVICE
                    )
                    latents = image_latents if latents is None else torch.cat((latents, image_latents))
                
                print("Plotting distance/topk plot...")
                plot.plot_distance_topk(args, df_combined, model, processor, pipe, latents, config.PART_ID[0])

            elif plot_type == 'agreement':
                print("Plotting agreement/topk plot...")
                plot.create_agreement_plot(args, df_combined, args.topks, to_save=True)
            
            elif plot_type == 'jaccard':
                print("Plotting jaccard/topk plot...")
                plot.create_jaccard_similarity_plot(args, df_combined, args.topks, to_save=True)
            
            else:
                raise ValueError("Invalid plot type. Please choose from distance, agreement, jaccard")



if __name__ == "__main__":
    main()
