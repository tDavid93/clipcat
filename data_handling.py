import yaml
from PIL import Image
from pathlib import Path
import clip


class CategoryData:
    name = None
    description = None
    embeddings = None
    path = None
    
    def __init__(self,name, description, path):
        self.name = name
        self.description = description
        self.path = path
        
    @property
    def __dict__(self):
        return {
            'name': self.name,
            'description': self.description,
            'path': self.path
        }
class ImageData:
    image_embedding = None
    predicted_category = None
    generated_text = None
    image_path = None
    raw_image = None
    predicted_filename = None

    
    
    def __init__(self,image_path):
        self.image_path = image_path
        try:
            self.raw_image = Image.open(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            self.raw_image = None
    
    @property
    def __dict__(self):
        return {
            'image_embedding': self.image_embedding,
            'predicted_category': self.predicted_category,
            'generated_text': self.generated_text,
            'image_path': self.image_path,
            'predicted_filename': self.predicted_filename
        }
    
def load_config(path) -> dict:
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
    
def load_categories(config, model, device, use_category_names = False) -> dict:
    categories = {}
    for category_name, category_details in config.items():
        if use_category_names:
            categories[category_name] = CategoryData(category_name, category_name, category_details['path'])
        else:
            categories[category_name] = CategoryData(category_name, category_details['description'], category_details['path'])
    for _, category in categories.items():
        print(f"Computing embeddings for {category.name}...")
        embeddings_tensor = model.encode_text(clip.tokenize(category.description).to(device))
        categories[category.name].embeddings = embeddings_tensor.detach().cpu().numpy()

    
    return categories
        
  