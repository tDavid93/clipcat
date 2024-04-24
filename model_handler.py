from transformers import AutoProcessor, Blip2ForConditionalGeneration
import clip


import torch
import numpy as np

def load_models(config, device, allow_blip=False):
    if allow_blip:
        print("Loading blip model")
        # Initialize models and processor
        processor = AutoProcessor.from_pretrained(config['config']['models']['blip']['model_name'])
        blip_model = Blip2ForConditionalGeneration.from_pretrained(config['config']['models']['blip']['model_name'], torch_dtype=torch.float16)
        blip_model.to(device)
        model, preprocess = clip.load(config['config']['models']['clip']['model_name'], device=device)
        return model, preprocess, processor, blip_model
    else:
        model, preprocess = clip.load(config['config']['models']['clip']['model_name'], device=device)
        return model, preprocess, None, None
    
    

def generate_text(image_input, image_id, processor, blip_model, device, allow_blip=False):
    if allow_blip:
        with torch.no_grad():
            blip_input = processor(image_input, return_tensors="pt").to(device, torch.float16)
            predicted_ids = blip_model.generate(**blip_input, max_new_tokens=10)
            generated_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            return generated_text
    return f"image_{image_id}"



def predict_category(image_input, model, categories, caption_input=None):
    if image_input is None:
        return None, None
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize the image features

        best_category = None
        best_similarity = -1  # Initialize to the lowest possible similarity
        for _, category in categories.items():
            category_embedding_norm = category.embeddings / np.linalg.norm(category.embeddings)  # Normalize the category embeddings
            similarity = np.dot(image_features.squeeze().cpu().numpy(), category_embedding_norm[0])  # Compute cosine similarity
            if similarity > best_similarity:
                best_similarity = similarity
                best_category = category.name
        return best_category, best_similarity 