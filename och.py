# Load model directly
import torch
from PIL import Image
from transformers import pipeline
from transformers import ViltProcessor, ViltModel
from torchvision import transforms
import pandas as pd
# Load the processor and model
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#for text input only
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",device='cuda')

categories = [
    "safe",
    "harassment",
    "sexual",
    "racism",
    "violence.",
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

file_path = "C:/Users/firda/Documents/UiPath/Scrape Forum/Comments2.xlsx"

df = pd.read_excel(file_path)
print("read excel")
# Ensure that the 'Output' column exists
# if 'Output' not in df.columns:
#     df['Output'] = pd.NA  # Initialize the Output column if it does not exist
df = df.dropna(how='all')

# Process each row (starting from index 1 to skip header row)
for index, row in df.iterrows():
    print("iterating")
    if index == 0:
        continue  # Skip header row
    
    # Check if the output column is empty or NaN (indicating that it hasn't been processed)
    if pd.isna(row['Output']):
        text = row[0]
        if text:  # Assuming the first column contains the text
        # Replace 'path/to/your/image.jpg' with the actual image path or modify as needed
        # Run the AI model on the text in the first column and save the result
            output = classify_text(text)
            print(f"{text}| {output['best_category']}\n")

            df.at[index, 'Output'] = output['best_category']
            df.to_excel(file_path, index=False)