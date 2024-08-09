# Load model directly
import torch
from PIL import Image
from transformers import pipeline
from transformers import ViltProcessor, ViltModel
from torchvision import transforms

# Load the processor and model
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#for text input only
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

categories = [
    "Safe content that does not contain harmful or inappropriate elements.",
    "harassment involving insults and threats",
    "Explicit sexual content including nudity, pornography, or sexually suggestive language.",
    "Content that incites or provokes hatred, violence, or discrimination based on race, religion, or ethnicity.",
    "Content depicting physical violence, threats, or promotion of violent behavior.",
]

# Function to classify text into categories
def classify_text(text):
    result = classifier(text, candidate_labels=categories)
    return {"best_category": result['labels'][0], "probs": result['scores']}

def classify_image_and_text(text, image_path):
    # Load and process the image and text
    image = Image.open(image_path)
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)

    # Get the embeddings from the model
    with torch.no_grad():
        outputs = model(**inputs)

    # For simplicity, we'll take the mean of the last hidden state across tokens
    pooled_output = outputs.last_hidden_state.mean(dim=1)

    # Simulate a classification by comparing the pooled output with some randomly initialized category vectors
    # In a real-world scenario, you'd train these vectors on a labeled dataset
    category_tensors = torch.randn(len(categories), pooled_output.size(-1))
    logits = torch.matmul(pooled_output, category_tensors.T).squeeze()

    # Get probabilities
    probs = torch.nn.functional.softmax(logits, dim=0)
    best_category = categories[torch.argmax(probs)]

    return {"best_category": best_category, "probs": probs}

text = "lol"

# # Example with both text and image
image_path = "./1950306.jpg"

if image_path:
    result = classify_image_and_text(text, image_path)
    print(f"Detected category with image: {result['best_category']}")
    print(f"Probabilities with image: {result['probs']}")
else:
    result = classify_text(text)

    print(f"Detected category without image: {result['best_category']}")
    print(f"Probabilities without image: {result['probs']}")