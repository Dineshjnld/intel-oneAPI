import os
import pandas as pd
from fastai.vision.all import *
from fastai.vision import models
from fastai.metrics import error_rate, accuracy
import warnings
import time

warnings.filterwarnings("ignore")
set_seed(42)

print('Modules loaded')

# Generate data paths with labels
data_dir = '../input/brats-2019-traintestvalid/dataset/train'
filepaths = []
labels = []

folds = os.listdir(data_dir)
for fold in folds:
    foldpath = os.path.join(data_dir, fold)
    filelist = os.listdir(foldpath)
    for file in filelist:
        fpath = os.path.join(foldpath, file)
        filepaths.append(fpath)
        labels.append(fold)

# Concatenate data paths with labels into one dataframe
Fseries = pd.Series(filepaths, name='filepaths')
Lseries = pd.Series(labels, name='labels')
df = pd.concat([Fseries, Lseries], axis=1)

df
dls = ImageDataLoaders.from_df(df,
                                fn_col=0,  # filepaths
                                label_col=1,  # labels
                                valid_pct=0.2,
                                folder='',
                                item_tfms=Resize(224))
dls.show_batch(max_n=16)
learn = vision_learner(dls, 'efficientnet_b3', metrics=[accuracy, error_rate], path='.')
learn.summary()

# Benchmarking variables
processed_images = 0
total_latency = 0.0

# Run inference on each image and measure performance
for i, (image, label) in enumerate(dls.valid_ds):
    # Preprocess the image
    start_time = time.time()
    pred_class, _, _ = learn.predict(image)
    end_time = time.time()

    # Update benchmarking variables
    processed_images += 1
    total_latency += (end_time - start_time) * 1000  # Convert to milliseconds

    # Print the result for the current image
    print(f"Image {i+1}: {pred_class}, Ground Truth: {label}")

# Calculate throughput and average latency
throughput = processed_images / (total_latency / 1000)  # Images per second
average_latency = total_latency / processed_images  # Milliseconds per image

# Print benchmarking measures
print("Benchmarking Measures:")
print(f"Processed Images: {processed_images}")
print(f"Total Latency (ms): {total_latency}")
print(f"Throughput (images/sec): {throughput}")
print(f"Average Latency (ms): {average_latency}")
