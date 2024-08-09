import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

categories = [
    "cyberbullying",
    "sexual content",
    "inciting racial/religious tension",
    "violent content"
]

def zero_shot_classification(text, image_path=None, categories=[]):
    if image_path:
        # Prepare inputs with both text and image
        inputs = processor(text=[text], images=[Image.open(image_path)], return_tensors="pt", padding=True, truncation=True)
    else:
        # Prepare inputs with only text
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)

    # Get model outputs
    outputs = model(**inputs)

    if image_path:
        # Compute cosine similarity between text and image features
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=-1)  # Convert logits to probabilities
    else:
        # Use only text features for classification
        logits_per_text = outputs.logits_per_text
        probs = logits_per_text.softmax(dim=-1)  # Convert logits to probabilities

    # Get the category with the highest probability
    best_category = categories[torch.argmax(probs)]

    return best_category, probs

text = "Eh fuck you jing kai you stupd bitch"

# # Example with both text and image
# image_path = "path/to/image.jpg"
# category_with_image, probabilities_with_image = zero_shot_classification(text, image_path, categories)
# print(f"Detected category with image: {category_with_image}")
# print(f"Probabilities with image: {probabilities_with_image}")

# Example with only text
category_without_image, probabilities_without_image = zero_shot_classification(text, categories=categories)
print(f"Detected category without image: {category_without_image}")
print(f"Probabilities without image: {probabilities_without_image}")