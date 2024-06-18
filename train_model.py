from fastai.vision.all import *
from pathlib import Path

# Set the path for train and test datasets
path = Path('.')

# Define transformations
batch_tfms = [*aug_transforms(size=299, min_scale=0.75), Normalize.from_stats(*imagenet_stats)]

# Load the data and apply transformations
dls = ImageDataLoaders.from_folder(path, train='train', valid='test', 
                                   item_tfms=Resize(299), 
                                   batch_tfms=batch_tfms, 
                                   bs=32)

learn = vision_learner(dls, resnet50, metrics=accuracy)

# Train model with 20 epochs
learn.fine_tune(20)

# Save the model
learn.export('simple_cnn_fastai_resnet50.pkl')

print('Finished Training and model saved as simple_cnn_fastai.pkl')