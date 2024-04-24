import gradio as gr
import os
from pathlib import Path

import torch

from pprint import pprint as print
import gc

import argparse


from data_handling import (CategoryData,
                           ImageData,
                           load_categories,
                           load_config)


from model_handler import (load_models,
                           generate_text,
                           predict_category)

import clip





# blip model load arguments
parser = argparse.ArgumentParser()
parser.add_argument("--allow-blip", action="store_true", help="Allow loading the BLIP model")
parser.add_argument("--config", type=str, default="config.yml", help="Path to the configuration file")
parser.add_argument("--use-category-names", action="store_true", help="Use category names for the embeddings instead of descriptions")

args = parser.parse_args()



# Configuration loading and validation

config = load_config(args.config)

    
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


print("Loading clip model")


current_index = 0

# Load categories from a YAML configuration
model, preprocess, processor, blip_model = load_models(config,
                                                       device,
                                                       allow_blip=args.allow_blip)
categories = load_categories(config['categories'],
                             model,
                             device,
                             use_category_names=args.use_category_names)


for _, category in categories.items():
    print(f"Category: {category.name}")
  
image_dir = Path(config['config']['paths']['images'])
image_files = [f for f in image_dir.glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]

images = []
for i, image_path in enumerate(image_files):
    images.append(ImageData(image_path))

for i, img in enumerate(images):
    if img.raw_image is not None:
        
        
        clip_input = preprocess(img.raw_image).unsqueeze(0).to(device)
        generated_text = generate_text(img.raw_image, i, processor, blip_model, device, args.allow_blip)
        images[i].generated_text = generated_text

        predicted_category, image_features = predict_category(clip_input,model, categories, generated_text)
        predicted_path = generated_text.replace(" ", config['config']['formatting']['separator']) + image_path.suffix

        images[i].predicted_category = categories[predicted_category]
        images[i].image = image_features
        images[i].predicted_filename = predicted_path
        print(f"Predicted category for {images[current_index].image_path}: {predicted_category}")
        


# Gradio interface setup and launch
def next_image_and_prediction(user_choice):
    global current_index
    print(f"User choice: {user_choice}")
    try:
        # Ensure user_choice is the name of the category, not the CategoryData object
        images[current_index].predicted_category = categories[user_choice]
    except Exception as e:
        print(f"Error setting category: {e}")
        
    current_index += 1
    if current_index >= len(images):
        current_index = 0
    
    print(f"Processing image {current_index}: {images[current_index].image_path}")
    return images[current_index].image_path, images[current_index].predicted_category.name,  images[current_index].predicted_filename
    
    

def move_images_to_category_folder(rename = False):
    for img in images:
        if img.predicted_category is not None:
            if not os.path.exists(f"./{config['config']['paths']['output']}/{img.predicted_category.path}"):
                os.makedirs(f"./{config['config']['paths']['output']}/{img.predicted_category.path}")
            new_path = f"./{config['config']['paths']['output']}/{img.predicted_category.path}/{img.predicted_filename}"
            if rename:
                os.rename(img.image_path, new_path)
            else:
                os.replace(img.image_path, new_path)
            print(f"Moved {img.image_path} to {new_path}")
        
    next_image_and_prediction(None)

def undo_move_images():
    for img in images:
        if img.predicted_category is not None:
            new_path = f"{image_dir}/{img.predicted_filename}"
            os.replace(f"{img.predicted_category.path}/{img.predicted_filename}", new_path)
            print(f"Moved {img.predicted_category.path}/{img.predicted_filename} to {new_path}")
        
    next_image_and_prediction(None)


#unloading the model
del model
if args.allow_blip:
    del blip_model
    del processor
gc.collect()



with gr.Blocks() as blocks:
    image_block = gr.Image(label="Image", type="filepath", height=300, width=300)
    filename = gr.Textbox(label="Filename", type="text")
    rename_toggle = gr.Checkbox(label="Rename")
    next_button = gr.Button("Next Image")
    category_dropdown = gr.Dropdown(label="Category", choices=list(categories.keys()), type="value")
    submit_button = gr.Button("Submit")
    submit_button.click(fn=move_images_to_category_folder, inputs=[rename_toggle], outputs=[])
    next_button.click(fn=next_image_and_prediction, inputs=category_dropdown, outputs=[image_block, category_dropdown, filename])
    reset_button = gr.Button("Undo")
    reset_button.click(fn=undo_move_images, inputs=[], outputs=[])
    if not images:
        print("No images found.")
    else:
        image_block.value = images[0].image_path
        filename.value = images[0].predicted_filename
        category_dropdown.value = images[0].predicted_category
        
        
        next_image_and_prediction(None)
        
blocks.launch(debug=True)
