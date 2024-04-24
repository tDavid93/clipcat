# ClipCat

## Overview
ClipCat is an AI-powered tool designed to automatically categorize images using advanced machine learning models, specifically leveraging CLIP and BLIP technologies. It sorts images into predefined categories based on visual and textual attributes identified by the AI.

## Features
- **Image Categorization:** Automatically categorizes images into specific groups based on content and context.
- **Interactive Interface:** Utilizes Gradio for a user-friendly interface that allows users to view, sort, and confirm categorizations.
- **State-of-the-Art Models:** Incorporates OpenAI's CLIP model and Salesforce's BLIP model for robust image and text processing.

## Installation

To set up ClipCat, follow these steps:

1. Clone the repository:
```bash
git clone [https://github.com/tDavid93/clipcat]
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
## Configuration
The project uses a `config.yml` file to manage the models and paths for processing. Here’s the current configuration setup:
    

```yaml
config:
    models:
    clip:
        model_name: "ViT-B/32"
    blip:
        model_name: "Salesforce/blip2-opt-2.7b"
categories:
    nature:
        title: "Nature"
        
        description: "Lush green forests, majestic mountains, and diverse wildlife."
        path: "nature"
    urban:
        title: "Urban"
        description: "Skylines, streets, and architectural features of urban settings."
        path: "urban"
    abstract:
        title: "Abstract"
        description: "Non-representational art that uses shapes, colors, and forms to achieve its effect."
        path: "abstract"
    portraits:
        title: "Portraits"
        description: "Focused images capturing the mood, personality, and expression of individuals."
        path: "portraits"
```

## Usage
To start the Gradio interface and begin categorizing images, run:

Copy code
python main.py
This command launches an interactive web application where you can view and categorize images, cycle through the dataset, and confirm or adjust the categorizations made by the AI.

## Contributing
Contributions are welcome! Please read the contributing guidelines before submitting pull requests to the project.

