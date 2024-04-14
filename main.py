import gradio as gr
import os
from pathlib import Path
from PIL import Image
import torch
import clip
import yaml
import pandas as pd
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from pprint import pprint as print

categories = {}
# Configuration loading and validation
def load_config(path):
    try:
        with open(path) as file:
            config = yaml.full_load(file)
        # Validate necessary sections are present
        necessary_keys = ['categories', 'config']
        for key in necessary_keys:
            if key not in config:
                raise ValueError(f'Missing necessary config section: {key}')
        return config
    except FileNotFoundError:
        print("Error: config.yml file not found.")
        raise
    except ValueError as e:
        print(str(e))
        raise
    
config = load_config('config.yml')
categories = config['categories']
    
    
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Initialize models and processor
processor = AutoProcessor.from_pretrained(config['config']['models']['blip']['model_name'])
blip_model = Blip2ForConditionalGeneration.from_pretrained(config['config']['models']['blip']['model_name'], torch_dtype=torch.float16)

blip_model.to(device)
model, preprocess = clip.load(config['config']['models']['clip']['model_name'], device=device)

current_index = 0

# Load categories from a YAML configuration


# Precompute category embeddings
for category_name, category_details in categories.items():
    print(f"Precomputing embeddings for category: {category_name}; {category_details}")
    embeddings_tensor = model.encode_text(clip.tokenize(category_details['description']).to(device))
    category_details['embeddings'] = embeddings_tensor.detach().cpu().numpy()

def load_image(path):
    try:
        image = Image.open(path)
        image_input = preprocess(image).unsqueeze(0).to(device)
        return image, image_input
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None, None

def predict_category(image_input, caption_input=None):
    if image_input is None:
        return None, None
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        if caption_input is not None:
            caption_input = clip.tokenize(caption_input).to(device)
            text_features = model.encode_text(caption_input)
            image_features = torch.cat([image_features, text_features])
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.cpu().numpy()
        best_category = None
        best_similarity = -1
        for category_name, category_details in categories.items():
            similarity = (image_features * category_details['embeddings']).sum()
            if similarity > best_similarity:
                best_similarity = similarity
                best_category = category_name
        return best_category, image_features
        
        
image_dir = Path(config['config']['paths']['images'])
image_files = [f for f in image_dir.glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]

images_df = pd.DataFrame(columns=['image_path', 'image_embedding', 'predicted_category', 'generated_text'])
for image_path in image_files:
    img, image_input = load_image(image_path)
    if img is not None:
        blip_input = processor(img, return_tensors="pt").to(device, torch.float16)
        # Ensure generation settings are compatible
        predicted_ids = blip_model.generate(**blip_input, max_new_tokens=10)
        generated_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        predicted_category, image_features = predict_category(image_input, generated_text)
        generated_text = generated_text.replace(" ", "_") + image_path.suffix

        new_row = {
            'image_path': str(image_path),
            'image_embedding': image_features if image_features is not None else None,
            'predicted_category': predicted_category,
            'generated_text': generated_text
        }
        # Using direct indexing to add to the DataFrame
        index = len(images_df)
        images_df.loc[index] = new_row
        

print(images_df.head())
# Gradio interface setup and launch
def next_image_and_prediction(user_choice):
    global current_index
    images_df.loc[current_index, 'predicted_category'] = user_choice
    current_index = (current_index + 1) % len(images_df)
    if current_index < len(images_df):
        next_img_path = images_df.loc[current_index, 'image_path']
        predicted_category = images_df.loc[current_index, 'predicted_category']
        predicted_filename = images_df.loc[current_index, 'generated_text']
        print(f"Next image: {next_img_path}, Predicted category: {predicted_category}")
        return next_img_path, predicted_category, predicted_filename
    else:
        return None, "No more images"

def move_images_to_category_folder():
    for index, row in images_df.iterrows():
        image_path = Path(row['image_path'])
        category_name = row['predicted_category']
        if category_name in categories:
            category_path = Path(categories[category_name]['path'])
            category_dir = Path(config['config']['paths']['output']) / category_path
            category_dir.mkdir(parents=True, exist_ok=True)
            new_image_path = category_dir / row['generated_text']
            image_path.rename(new_image_path)
            print(f"Moved {image_path} to {new_image_path}")
        else:
            print(f"Category {category_name} not found in categories.")





with gr.Blocks() as blocks:
    image_block = gr.Image(label="Image", type="filepath", height=300, width=300)
    filename = gr.Textbox(label="Filename", type="text")
    next_button = gr.Button("Next Image")
    category_dropdown = gr.Dropdown(label="Category", choices=list(categories.keys()), type="value")
    submit_button = gr.Button("Submit")
    submit_button.click(fn=move_images_to_category_folder, inputs=[], outputs=[])
    next_button.click(fn=next_image_and_prediction, inputs=category_dropdown, outputs=[image_block, category_dropdown, filename])

    if not images_df.empty:
        img_path, predicted_category = images_df.loc[0, ['image_path', 'predicted_category']]
        image_block.value = img_path
        category_dropdown.value = predicted_category

blocks.launch()
