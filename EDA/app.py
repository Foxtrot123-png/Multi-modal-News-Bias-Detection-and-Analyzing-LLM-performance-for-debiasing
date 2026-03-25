import os
import torch
import pandas as pd
from torch import nn
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer, BertModel, AutoModel, GPT2Tokenizer, GPT2LMHeadModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import gradio as gr
import classes_for_multimodal_bias_classification as cfmbc
import numpy as np
import google.generativeai as genai
import language_tool_python
import random
import torch.nn.functional as F

# 1. Environment & Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Initialize Models (Using Relative Paths)
text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
MAX_LEN = 512

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load BABE Model
model_BABE = cfmbc.BertClass()
checkpoint_bert = torch.load("BABE_fine_tuned_model.pt", map_location="cpu")
model_BABE.load_state_dict(checkpoint_bert, strict=False)
model_BABE.eval()

# Load NBS Model
model_NBS = cfmbc.load_model(drop_proj=0.43797, drop_fus=0.08885)
checkpoint_nbs = torch.load("fine_tuned_model_nbs.pt", map_location="cpu")
model_NBS.load_state_dict(checkpoint_nbs, strict=False)
model_NBS.eval()

if device.type == "cuda":
    model_BABE = model_BABE.to(device)
    model_NBS = model_NBS.to(device)

# 3. Debiasing Setup (GPT-2 & Gemini)
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token
model_gpt2.config.pad_token_id = tokenizer_gpt2.eos_token_id

# Initialize LanguageTool (Automatic Java setup on HF)
tool = language_tool_python.LanguageTool('en-US')

# 4. Data Loading
nbs_data = pd.read_csv('No_Corrupted_NBS.csv')
biased_headlin = nbs_data[nbs_data['multimodal_label_y'] == 'Likely']['headline']
Non_biased_headlin = nbs_data[nbs_data['multimodal_label_y'] == 'Unlikely']['headline']

# 5. Helper Functions
# NEW CORRECTED VERSION
def predict_babe(model, article_text):
    inputs = text_tokenizer(
        article_text, 
        max_length=MAX_LEN, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt',
        return_token_type_ids=True # Ensure this is explicitly requested
    )
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'].to(device), 
            attention_mask=inputs['attention_mask'].to(device), 
            token_type_ids=inputs['token_type_ids'].to(device)
        ) 
        conf = torch.sigmoid(outputs).cpu().item()
    return conf

def predict_nbs(model, article_text, image_pil):
    text_inputs = text_tokenizer(article_text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    image_input = image_transform(image_pil.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(pixel_values=image_input, input_ids=text_inputs['input_ids'].to(device), attention_mask=text_inputs['attention_mask'].to(device))
        conf = torch.sigmoid(outputs).cpu().item()
    return conf

def gemini_lm(prompt_type, article):
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        return "Error: GEMINI_API_KEY missing from Secrets."
    
    # THIS IS THE PART YOU NEED TO ADD:
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        prompt = build_prompt(prompt_type, article)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"

def run_gpt2(prompt_type, article):
    prompt = build_prompt(prompt_type, article)
    inputs = tokenizer_gpt2(prompt, return_tensors="pt", truncation=True, max_length=800)
    output = model_gpt2.generate(**inputs, max_new_tokens=100)
    return tokenizer_gpt2.decode(output[0], skip_special_tokens=True)

def build_prompt(prompt_type, article):
    # Simplified prompt builder for brevity
    prompts = {
        "zero": f"Article: {article}\nRewrite in neutral tone.",
        "single": f"Article: {article}\nRewrite neutrally like this: {Non_biased_headlin.iloc[0]}",
        "multi": f"Article: {article}\nRewrite using these examples...",
        "role": f"You are a neutral editor. Rewrite this: {article}"
    }
    return prompts.get(prompt_type, prompts["zero"])

def cosine_similarity_texts(text1, text2):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, padding=True)
    inputs2 = tokenizer(text2, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        emb1 = model(**inputs1).last_hidden_state[:, 0, :]
        emb2 = model(**inputs2).last_hidden_state[:, 0, :]
    return F.cosine_similarity(emb1, emb2).item()

def generate_wordcloud_fig(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# 6. Main Prediction Function
def predict_bias(article_text, image_pil, show_individual, show_explanations, show_debiasing):
    if not article_text.strip():
        return "Empty text", "0%", None, "", "", pd.DataFrame(), pd.DataFrame()

    babe_conf = predict_babe(model_BABE, article_text)
    
    if image_pil:
        nbs_conf = predict_nbs(model_NBS, article_text, image_pil)
        final_conf = (babe_conf + nbs_conf) / 2
    else:
        nbs_conf = 0.0
        final_conf = babe_conf

    label = "Biased" if final_conf > 0.5 else "Not Biased"
    prob = f"{final_conf*100:.2f}%"
    
    wc = generate_wordcloud_fig(article_text) if show_explanations else None
    b_res = f"Text Model: {babe_conf*100:.1f}%"
    n_res = f"Multimodal Model: {nbs_conf*100:.1f}%" if image_pil else "N/A"

    # Debiasing Logic
    res_gemini = []
    res_gpt2 = []
    if show_debiasing:
        for p in ["zero", "single", "multi", "role"]:
            # Gemini
            d_gem = gemini_lm(p, article_text)
            score_gem = cosine_similarity_texts(article_text, d_gem)
            err_gem = len(tool.check(d_gem))
            res_gemini.append([p, score_gem, err_gem, d_gem])
            # GPT2
            d_gpt = run_gpt2(p, article_text)
            score_gpt = cosine_similarity_texts(article_text, d_gpt)
            err_gpt = len(tool.check(d_gpt))
            res_gpt2.append([p, score_gpt, err_gpt, d_gpt])

    df_gem = pd.DataFrame(res_gemini, columns=["Mode", "Similarity", "Grammar Errors", "Text"])
    df_gpt = pd.DataFrame(res_gpt2, columns=["Mode", "Similarity", "Grammar Errors", "Text"])

    return label, prob, wc, b_res, n_res, df_gem, df_gpt

# 7. UI Layout
with gr.Blocks() as demo:
    gr.Markdown("# Multimodal News Bias Classifier")
    
    with gr.Row():
        txt = gr.Textbox(label="Article Text", lines=5)
        img = gr.Image(type="pil", label="Image")
        
    with gr.Row():
        ind = gr.Checkbox(label="Individual Scores", value=True)
        exp = gr.Checkbox(label="WordCloud", value=True)
        deb = gr.Checkbox(label="Run Debiasing (Slow)", value=False)

    btn = gr.Button("Analyze")
    
    with gr.Column():
        out_lbl = gr.Label(label="Result")
        out_prob = gr.Textbox(label="Confidence")
        out_wc = gr.Plot()
        out_babe = gr.Textbox(label="BABE Score")
        out_nbs = gr.Textbox(label="NBS Score")
        out_df_gem = gr.Dataframe(label="Gemini Debiasing")
        out_df_gpt = gr.Dataframe(label="GPT-2 Debiasing")

    btn.click(predict_bias, [txt, img, ind, exp, deb], [out_lbl, out_prob, out_wc, out_babe, out_nbs, out_df_gem, out_df_gpt])

demo.launch(server_name="0.0.0.0", server_port=7860)