import torch
import pandas as pd
from torch import nn
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import gradio as gr
import classes_for_multimodal_bias_classification as cfmbc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
MAX_LEN = 512

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model_BABE = cfmbc.BertClass()
checkpoint_bert = torch.load(
    "/Users/ritikrmohapatra/Documents/GitHub/Multi-Model-Bias-Detection-and-Debiasing-the-News/EDA/BABE_fine_tuned_model.pt",
    map_location="cpu"
)
model_BABE.load_state_dict(checkpoint_bert, strict=False)
model_BABE.eval()
if device.type == "cuda":
    model_BABE = model_BABE.to(device)

model_NBS = cfmbc.load_model(drop_proj=0.43797, drop_fus=0.08885)
checkpoint_nbs = torch.load(
    "/Users/ritikrmohapatra/Documents/GitHub/Multi-Model-Bias-Detection-and-Debiasing-the-News/EDA/fine_tuned_model_nbs.pt",
    map_location="cpu"
)
model_NBS.load_state_dict(checkpoint_nbs, strict=False)
model_NBS.eval()
if device.type == "cuda":
    model_NBS = model_NBS.to(device)


def predict_babe(model, article_text):
    model.eval()
    inputs = text_tokenizer(
        article_text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(
            inputs['input_ids'].to(device),
            inputs['attention_mask'].to(device),
            inputs['token_type_ids'].to(device)
        )
        conf = torch.sigmoid(outputs).cpu().item()
    return conf





def predict_nbs(model, article_text, image_pil):
    model.eval()
    text_inputs = text_tokenizer(
        article_text,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    image_input = image_transform(image_pil.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(
            pixel_values=image_input,
            input_ids=text_inputs['input_ids'].to(device),
            attention_mask=text_inputs['attention_mask'].to(device)
        )
        conf = torch.sigmoid(outputs).cpu().item()
    return conf

def generate_wordcloud_fig(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud of Article Text')
    return fig

def predict_bias(article_text, image_pil=None, show_individual=True, show_explanations=True):
    if not article_text.strip():
        return "Please enter some text!", "N/A", None, None, None

    babe_conf = predict_babe(model_BABE, article_text)
    babe_pred = babe_conf > 0.5

    if image_pil is not None:
        nbs_conf = predict_nbs(model_NBS, article_text, image_pil)
        nbs_pred = nbs_conf > 0.5
        final_conf = 0.8 * babe_conf + 0.2 * nbs_conf
        final_pred = final_conf > 0.5
    else:
        nbs_conf, nbs_pred = 0.0, False
        final_conf, final_pred = babe_conf, babe_pred

    label = "Biased" if final_pred else "Not Biased"
    confidence_text = f"Model is {final_conf*100:.2f}% confident"
    wordcloud_fig = generate_wordcloud_fig(article_text) if show_explanations else None
    babe_result = None
    nbs_result = None

    if show_individual:
        babe_result = f"Text Model: {['Not Biased','Biased'][babe_pred]} ({babe_conf*100:.2f}%)"
        if image_pil:
            nbs_result = f"Multimodal Model: {['Not Biased','Biased'][nbs_pred]} ({nbs_conf*100:.2f}%)"
    return label, confidence_text, wordcloud_fig, babe_result, nbs_result

with gr.Blocks() as demo:
    with gr.Tab("About"):
        gr.Markdown("# Mutlimodal News Bias Classifier")
        gr.Markdown("Detect if a news article is biased using text only and multimodal models.")
        gr.Markdown("## Project Overview")
        gr.Markdown("""
        This project detects bias in news articles using:
        - **BABE Dataset**: Text-only model
        - **NBS+ Dataset**: Multimodal model (Text + Images)

        **About the Project:**  
        This project aims to design and implement an AI-powered news bias detection system that leverages BERT, pre-trained on the GoodNews dataset. The system is fine-tuned in two ways: one using the BABE dataset for text-only bias detection, and the other combining BERT with a ResNet model, fine-tuned on the News Bias Plus (NBS+) dataset to create a multimodal model. Additionally, these models are used to test the performance of popular large language models (LLMs) in generating debiased news articles.

        Key features of the project include:

           - Comprehensive NLP feature engineering for analyzing text.

           - Masked Language Model (MLM) training of BERT on a large corpus of unlabelled articles from the GoodNews dataset.

           - Training and evaluation of machine learning and deep learning models using extensive classification metrics such as Accuracy, Precision, Recall, F1-score, and ROC AUC.

           - Fusion of text and multimodal models, combining a text-based bias classifier with an image+caption bias classifier to create a more robust multimodal bias detection system.

           - Evaluation of popular LLMs for generating debiased versions of news articles.

           - Using the models as a testing metric for LLM-generated summaries, alongside Grammarly and cosine similarity for grammar and content similarity checks.

           - Ethical consideration of potential biases and emphasis on model explainability, revisited in the evaluation phase.

        By addressing technical gaps and offering a practical solution for news bias detection, this research contributes to the broader field of computational media security and sets the stage for future improvements in responsible AI for journalism.
        """)
    
    with gr.Tab("Prediction"):
        with gr.Row():
            article_input = gr.Textbox(label="Enter Article Text", lines=10, placeholder="Paste your article here...")
            image_input = gr.Image(type="pil", label="Upload Image (Optional)")
        with gr.Row():
            show_individual = gr.Checkbox(label="Show individual model predictions", value=True)
            show_explanations = gr.Checkbox(label="Show explanation visualizations", value=True)
        predict_button = gr.Button("Predict Bias")
        final_opt = gr.Label(label="Final Prediction")
        conf = gr.Textbox(label="Confidence Score")
        wordcloud_opt = gr.Plot(label="Word Cloud")
        babe_opt = gr.Textbox(label="Text-only Model Result")
        nbs_opt = gr.Textbox(label="Multimodal Model Result")
        predict_button.click(
            fn=predict_bias,
            inputs=[article_input, image_input, show_individual, show_explanations],
            outputs=[final_opt, conf, wordcloud_opt, babe_opt, nbs_opt]
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
