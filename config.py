
DATASET_DIR = "your-directory-to-save-diffusiondb-dataset-images"
PLOT_DIR = "your-directory-to-save-results"
SEED = 42
DEVICE = "cuda"
PART_ID = list(range(1, 101)) #100k
STABLE_DIFFUSION_ID = "CompVis/stable-diffusion-v1-4"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
HEIGHT, WIDTH = 512, 512
# for hyperparameter tuning
N_NEIGHBORS = [15, 30, 40]
N_COMPONENTS = [2, 4, 8, 16, 32]
MIN_CLUSTER_SIZE = [7, 10, 25, 50]
NUM_INFERENCE_STEPS = 20

# best hyperparameters
BEST_CLUSTER = {
    'n_neighbors': 15,
    'n_components': 32,
    'min_cluster_size': 7,
}
