import torch
from plantclef.config import get_device

print(f"PyTorch Version: {torch.__version__}")
device = get_device()
print(f"Using device: {device}")


import pandas as pd
from pathlib import Path

# Get list of stored filed in cloud bucket
root = Path().resolve().parents[0]
print(root)


# path to data
data_path = f"{root}/plantclef-tutorial/data/parquet"
train_path = f"{data_path}/train_pytorch_webinar_filtered"
test_path = f"{data_path}/test_2025_pytorch_webinar"

# read train/test data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# display data
print(f"Train DF shape: {train_df.shape}")
print(f"Test DF shape: {test_df.shape}")

display(train_df.head(3))
display(test_df.head(3))


# limit to 2 samples for testing
limit_train_df = train_df.head(2)
print(f"Limit DF shape: {limit_train_df.shape}")

from plantclef.embed.workflow import torch_pipeline

# extract embeddings
embeddings, logits = torch_pipeline(
    limit_train_df,
    batch_size=2,
    use_grid=False,
    cpu_count=1,
)

# embeddings shape
embeddings.shape

# create embeddings dataframe
cols = ["image_name", "data", "species", "species_id"]
embeddings_df = limit_train_df[cols].copy()
embeddings_df["embeddings"] = embeddings.tolist()
embeddings_df.head(2)


from plantclef.plotting import plot_images_from_binary

plot_images_from_binary(
    embeddings_df,
    data_col="data",
    label_col="species",
    grid_size=(1, 2),
    crop_square=True,
    figsize=(8, 4),
)




from plantclef.plotting import plot_embeddings

plot_embeddings(
    embeddings_df,
    data_col="embeddings",
    label_col="species",
    grid_size=(1, 2),
    figsize=(8, 4),
)




# set params
USE_GRID = True
GRID_SIZE = 3  # 3x3 grid of tiles
CPU_COUNT = 1  # custom cpu_count
TOP_K = 5  # top-K logits for each tile

# select images from test set
image_names = ["CBN-Pyr-03-20230706.jpg", "CBN-can-E6-20230706.jpg"]
test_image_df = test_df[test_df["image_name"].isin(image_names)]

# get embeddings and logits
embeddings, logits = torch_pipeline(
    test_image_df,
    batch_size=2,
    use_grid=USE_GRID,
    grid_size=GRID_SIZE,
    cpu_count=CPU_COUNT,
    top_k=TOP_K,
)


# embeddings shape
embeddings.shape  # (2, 9, 768)


import torch
# create embeddings dataframe
def explode_embeddings_logits(
    df: pd.DataFrame,
    embeddings: torch.Tensor,
    logits: list,
    cols: list = ["image_name", "data"],
) -> pd.DataFrame:
    # create dataframe
    pred_df = df[cols].copy()
    pred_df["embeddings"] = embeddings.cpu().tolist()
    pred_df["logits"] = logits
    # explode embeddings
    explode_df = pred_df.explode(["embeddings", "logits"], ignore_index=True)
    # assign tile number for each image
    explode_df["tile"] = explode_df.groupby("image_name").cumcount()
    return explode_df


explode_df = explode_embeddings_logits(test_image_df, embeddings, logits)
explode_df.head(9)



from plantclef.plotting import plot_image_tiles

# show image tiles
plot_image_tiles(
    explode_df,
    data_col="data",
    grid_size=3,
)




from plantclef.plotting import plot_embed_tiles

plot_embed_tiles(
    explode_df,
    data_col="embeddings",
    grid_size=3,
    figsize=(15, 8),
)



# plot grid embeddings
plot_embeddings(
    explode_df,
    data_col="embeddings",
    label_col="tile",
    grid_size=(3, 3),
    figsize=(8, 8),
)



print(f"Length logits: {len(logits)}")


# display logits of first tile
explode_df["logits"].iloc[0]


# display logits for each tile
for i in range(9):
    logits = explode_df["logits"].iloc[i]
    logits_formatted = {k: round(v, 3) for k, v in logits.items()}
    print(f"Tile {i+1}: {logits_formatted}")
    
    
    
import os

cpu_count = os.cpu_count()
print(f"CPU count: {cpu_count}")


# params
USE_GRID = True
GRID_SIZE = 3  # 3x3 grid of tiles
CPU_COUNT = 1  # custom cpu_count
TOP_K = 5  # top-K logits for each tile

# get embeddings and logits
test_embeddings, test_logits = torch_pipeline(
    test_df,
    batch_size=10,  # 10 imamges per batch
    use_grid=USE_GRID,
    grid_size=GRID_SIZE,
    cpu_count=CPU_COUNT,
    top_k=TOP_K,
)




print(test_embeddings.shape)
print(len(test_logits))


# explode full embeddings and logits
test_explode_df = explode_embeddings_logits(
    test_df,
    test_embeddings,
    test_logits,
)


print(test_explode_df.shape)
test_explode_df.head(9)

plot_embed_tiles(
    test_explode_df,
    data_col="embeddings",
    grid_size=3,
)