# Load model directly
import torch
from PIL import Image
from transformers import pipeline
# from transformers import ViltProcessor, ViltModel
# from torchvision import transforms
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)


pipe = pipeline("text-generation", model="mistralai/Mistral-Nemo-Instruct-2407", device='cuda')

categories = [
    "safe",
    "harassment",
    "sexual-harrasment",
    "Racism",
    "violence",
]

# Define the prompt
def create_prompt(content):
    return (f"Please classify the following content into one of the categories below:\n\n"
            "1. **Online harassment involving insults, threats, or repeated unwanted contact**: "
            "Content that includes any form of harassment such as insults, threats, or repeated unwanted communication.\n"
            "2. **Explicit sexual imagery or suggestive language**: Content that contains explicit sexual images or suggestive language of a sexual nature.\n"
            "3. **Provoking hatred based on race or religion**: Content that incites hatred or violence towards individuals based on their race or religion.\n"
            "4. **Depictions of physical violence or threats**: Content that depicts physical violence or threats of violence.\n"
            "5. **Safe content with no harmful elements detected**: Content that does not fit into any of the above categories and is deemed safe and free from harmful elements.\n\n"
            f"Content: {content}\n\n"
            "Category:")

# Function to classify text into categories
def classify_text(text):
    print("classifying text")
    #for text input only
    prompt = create_prompt(text)
    messages = [
        {"role": "user", "content": prompt},
    ]
    return pipe(messages)
# def classify_image_and_text(text, image_path):
#     # Load and process the image and text
#     image = Image.open(image_path)
#     inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)

#     # Get the embeddings from the model
#     with torch.no_grad():
#         outputs = model(**inputs)

#     # For simplicity, we'll take the mean of the last hidden state across tokens
#     pooled_output = outputs.last_hidden_state.mean(dim=1)

#     # Simulate a classification by comparing the pooled output with some randomly initialized category vectors
#     # In a real-world scenario, you'd train these vectors on a labeled dataset
#     category_tensors = torch.randn(len(categories), pooled_output.size(-1))
#     logits = torch.matmul(pooled_output, category_tensors.T).squeeze()

#     # Get probabilities
#     probs = torch.nn.functional.softmax(logits, dim=0)
#     max_prob = torch.max(probs)

#     best_category = categories[torch.argmax(probs)]

#     # Check if the maximum probability is below the threshold
#     if max_prob < 0.9:
#         best_category = "Safe content that does not contain harmful or inappropriate elements."
#     print(f"{text}\n{best_category}\n")
#     return {"best_category": best_category, "probs": probs}

# # text = "lol"

# # # Example with both text and image
# image_path = "./1950306.jpg"

# if image_path:
#     result = classify_image_and_text(text, image_path)
#     print(f"Detected category with image: {result['best_category']}")
#     print(f"Probabilities with image: {result['probs']}")
# else:
#     result = classify_text(text)

#     print(f"Detected category without image: {result['best_category']}")
#     print(f"Probabilities without image: {result['probs']}")

print("reading file")
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
            print(f"{text}| {output}\n")

            df.at[index, 'Output'] = output
            df.to_excel(file_path, index=False)


# print(torch.cuda.is_available())
