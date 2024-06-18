from fastai.vision.all import load_learner, PILImage
from pathlib import Path
import requests
from io import BytesIO

# Load the models
learn_resnet32 = load_learner(Path('simple_cnn_fastai_resnet32.pkl'))
learn_resnet50 = load_learner(Path('simple_cnn_fastai_resnet50.pkl'))
learn_resnet101 = load_learner(Path('simple_cnn_fastai_resnet101.pkl'))

# Hardcoded path
# image = 'test/Bentley Arnage Sedan 2009/003222.jpg'

# Download image from URL
image_url = input("Please enter an image URL for prediction: ")
response = requests.get(image_url)
image = PILImage.create(BytesIO(response.content))

# Predict using ResNet32
pred32, pred_idx32, probs32 = learn_resnet32.predict(image)
print('ResNet32 - Predicted class:', pred32)
print('ResNet32 - Probability: {:.2f}%'.format(probs32[pred_idx32].item() * 100))

# Predict using ResNet50
pred50, pred_idx50, probs50 = learn_resnet50.predict(image)
print('ResNet50 - Predicted class:', pred50)
print('ResNet50 - Probability: {:.2f}%'.format(probs50[pred_idx50].item() * 100))

# Predict using ResNet101
pred101, pred_idx101, probs101 = learn_resnet101.predict(image)
print('ResNet101 - Predicted class:', pred101)
print('ResNet101 - Probability: {:.2f}%'.format(probs101[pred_idx101].item() * 100))