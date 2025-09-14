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
from transformers import BertModel,AutoModel
import numpy as np

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

#--------------------Debiase Part --------------

#https://huggingface.co/gpt2
from transformers import GPT2Tokenizer, GPT2LMHeadModel


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

def run_gpt2(prompt_type,article):
    prompt = build_prompt(prompt_type,article)
    inputs = tokenizer(prompt, return_tensors="pt",truncation=True,
        max_length=800)
    output = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)


#https://ai.google.dev/gemini-api/docs/multimodal
import google.generativeai as genai


def gemini_lm(prompt_type,article):
    prompt = build_prompt(prompt_type,article)
    import google.generativeai as genai
    genai.configure(api_key="AIzaSyDNH9TmB1bzdcJGO3yI931Js7W2HJsA5Y4")

    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text


import random

index = random.randint(1, 2185)

nbs = pd.read_csv('/Users/ritikrmohapatra/Documents/GitHub/Multi-Model-Bias-Detection-and-Debiasing-the-News/EDA/No_Corrupted_NBS.csv')



biased_headlin = nbs[nbs['multimodal_label_y'] == 'Likely']['headline']
Non_biased_headlin = nbs[nbs['multimodal_label_y'] == 'Unlikely']['headline']

rand_idx_1 = random.randint(0, len(biased_headlin)-1)
rand_idx_2 = random.randint(0, len(Non_biased_headlin)-1)
rand_idx_3 = random.randint(0, len(Non_biased_headlin)-1)
rand_idx_4 = random.randint(0, len(biased_headlin)-1)

#https://huggingface.co/docs/transformers/main/en/generation_strategies


exam_sig = f"""
Uniased: " {Non_biased_headlin.iloc[rand_idx_2]} "
Biased: "{biased_headlin.iloc[rand_idx_1]} " """ 

exam_multi = f"""
Unbiased: "{Non_biased_headlin.iloc[rand_idx_2]}"
Biased: "{biased_headlin.iloc[rand_idx_4]}"

Unbiased: "{Non_biased_headlin.iloc[rand_idx_3]}"
Biased: "{biased_headlin.iloc[rand_idx_1]}"
"""

def build_prompt(prompt_type=str,article=''):
    if prompt_type == "zero":
        print(article)
        return f"""
Article: {article}

Rewrite the article and caption in a neutral and unbiased tone.
"""
    elif prompt_type == "single":
        print(article)

        return f"""
Article: {article}

Here is an example of debiasing:
{exam_sig}

Now, rewrite the article and caption above in a neutral and unbiased tone.
"""
    elif prompt_type == "multi":
        print(article)
        return f"""
Article: {article}

Here are examples of rewriting biased text into neutral form:
{exam_multi}

Now rewrite the article and caption above in the same neutral style.
"""
    elif prompt_type == "role":
        print(article)
        return f"""You are a professional news editor who is neutral in every scenario.Your task is to remove emotionally charged or biased language and rewrite the following article and caption in a strictly neutral,unbiased and factual tone.

Article: {article}
"""
    else:
        raise ValueError("Invalid prompt_type")



#https://pytorch.org/docs/stable/nn.functional.html#cosine-similarity
import torch.nn.functional as F

def cosine_similarity_texts(text1, text2, model_name: str = "bert-base-uncased"):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, padding=True)
    inputs2 = tokenizer(text2, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        emb1 = model(**inputs1).last_hidden_state[:, 0, :]  
        emb2 = model(**inputs2).last_hidden_state[:, 0, :]

    cos_sim = F.cosine_similarity(emb1, emb2).item()
    return cos_sim
  


import language_tool_python
tool = language_tool_python.LanguageTool('en-US', remote_server='http://localhost:8081')

#Terminal - languagetool-server









def predict_bias(article_text, image_pil=None, show_individual=True, show_explanations=True,show_debiasing=False):
    if not article_text.strip():
        return "Please enter some text!", "N/A", None, None, None

    babe_conf = predict_babe(model_BABE, article_text)
    babe_pred = babe_conf > 0.4

    if image_pil is not None:
        nbs_conf = predict_nbs(model_NBS, article_text, image_pil)
        nbs_pred = nbs_conf > 0.6179447412814503
        final_conf = 0.5 * babe_conf + 0.5 * nbs_conf
        final_pred = final_conf > 0.5
    else:
        nbs_conf, nbs_pred = 0.0, False
        final_conf, final_pred = babe_conf, babe_pred

    label = "Biased" if final_pred else "Not Biased"
    confidence_text = f"Probability Biased: {final_conf*100:.2f}% "
    wordcloud_fig = generate_wordcloud_fig(article_text) if show_explanations else None
    babe_result = None
    nbs_result = None

    if show_individual:
        babe_result = f"Text Model: ({babe_conf*100:.2f}% Biased-> Prediction:{['Not Biased','Biased'][babe_pred]} )"
        if image_pil:
            nbs_result = f"Multimodal Model: ({nbs_conf*100:.2f}% Biased -> Prediction:{['Not Biased','Biased'][nbs_pred]} )"


    df_gpt2 = pd.DataFrame({'index': ['cosine_score','error_grammer', 'debiased_text']})
    df_gpt2['zero']=0
    df_gpt2['single']=0
    df_gpt2['multi']=0
    df_gpt2['role']=0


    df_gemini = pd.DataFrame({
        'index': ['cosine_score','error_grammer', 'debiased_text']
    })
    df_gemini['zero']=0
    df_gemini['single']=0
    df_gemini['multi']=0
    df_gemini['role']=0


    prompt_types = ["zero", "single", "multi", "role"]

    for i in prompt_types:
        if show_debiasing:
            gpt2_d = run_gpt2(i,article_text)
            gemini_d = gemini_lm(i,article_text)

            gpt2_score = cosine_similarity_texts(article_text, gpt2_d)
            gemini_score = cosine_similarity_texts(article_text, gemini_d)

            er_gem = tool.check(gemini_d)
            er_count_gem = len(er_gem)

            er_gpt2 = tool.check(gpt2_d)
            er_count_gpt2 = len(er_gpt2)


            scor_gemini=cosine_similarity_texts(article_text,gemini_d)
            scor_gpt2=cosine_similarity_texts(article_text,gpt2_d)
            df_gemini.loc[df_gemini['index'] == 'cosine_score', i] =scor_gemini
            df_gemini.loc[df_gemini['index'] == 'error_grammer', i] =er_count_gem
            df_gemini.loc[df_gemini['index'] == 'debiased_text', i] = gemini_d


            df_gpt2.loc[df_gpt2['index'] == 'cosine_score', i] =scor_gpt2 
            df_gpt2.loc[df_gpt2['index'] == 'error_grammer', i] =er_count_gpt2
            df_gpt2.loc[df_gpt2['index'] == 'debiased_text', i] = gpt2_d
    return label, confidence_text, wordcloud_fig, babe_result, nbs_result,df_gemini,df_gpt2












with gr.Blocks() as demo:
    with gr.Tab("About"):
        gr.Markdown("# Mutlimodal News Bias Classifier")
        gr.Markdown("Detect if a news article is biased using text only and multimodal models.")
        gr.Markdown("## Project Overview")
        gr.Markdown("""
        This project detects bias in news articles using:
        - **BABE Dataset**: Text-only model
        - **NBS+ Dataset**: Multimodal model (Text + Images)
        - **GoodNews Dataset**: Pre-training dataset for BERT

        **About the Project:**  
        This project aims to design and implement an Deep Learning powered news bias detection system that leverages BERT, pre-trained on the GoodNews dataset. The system is fine-tuned in two ways: one using the BABE dataset for text-only bias detection, and the other combining BERT with a ResNet model, fine-tuned on the News Bias Plus (NBS+) dataset to create a multimodal model. Additionally, these models are used to test the performance of popular large language models (LLMs) in generating debiased news articles.

        Key features of the project include:

           - Comprehensive NLP feature engineering for analyzing text.

           - Masked Language Model (MLM) training of BERT on a large corpus of unlabelled articles from the GoodNews dataset.

           - Training and evaluation of machine learning and deep learning models using extensive classification metrics such as Accuracy, Precision, Recall, F1-score, and ROC AUC.

           - Fusion of text and multimodal models combining a text-based bias classifier with an image+caption bias classifier to create a more robust multimodal bias detection system.

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
            show_debiasing = gr.Checkbox(label="Show Debiased Articles", value=False)
        predict_button = gr.Button("Predict Bias")
        final_opt = gr.Label(label="Final Prediction")
        conf = gr.Textbox(label="Probability")
        wordcloud_opt = gr.Plot(label="Word Cloud")
        babe_opt = gr.Textbox(label="Text-only Model Result")
        nbs_opt = gr.Textbox(label="Multimodal Model Result")
        df_gemini = gr.Dataframe(label="Gemini Results", headers=["index", "zero", "single", "multi", "role"]) if show_debiasing else None
        df_gpt2 = gr.Dataframe(label="GPT-2 Debiasing Results", headers=["index", "zero", "single", "multi", "role"]) if show_debiasing else None
        predict_button.click(
            fn=predict_bias,
            inputs=[article_input, image_input, show_individual, show_explanations,show_debiasing],
            outputs=[final_opt, conf, wordcloud_opt, babe_opt, nbs_opt,df_gemini,df_gpt2]
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

