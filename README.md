Multi-Modal Model for Biased News Classification and Debiasing  

Overview  

This project provides a deep learning–based system for detecting bias in news articles (text only model +multimodal model(text+ images)) and debiasing them using LLMs.  

Text-only classifier → BERT fine-tuned on BABE dataset.  

Multimodal classifier → BERT (text) + ResNet-34 (images) with cross-attention.  

Fusion model → Combines text-only and multimodal predictions using XGBoost for batch prediction or Concatenation for Single Values Prediction

LLM Debiasing → GPT-2 and GEMINI used to rewrite biased news into more neutral text.  

Dashboard → FastAPI + Gradio app for on-the-fly bias detection.  

Datasets

BABE → Text with expert-annotated bias labels.  

GoodNews → NYT articles + images (for pretraining).  
NewBiasDataset → Text + image pairs with bias labels (cleaned version: No_Corrupted_NBS.csv).  

Models & Files  
  
Text-only model → Babe_Dataset.ipynb  

Multimodal model → News_Media_Dataset.ipynb  

Fusion model → Fusion.ipynb  

LLM debiasing → LLMs.ipynb  

Dashboard → news_bias_app.py  

Saved models:  

BABE_fine_tuned_model.pt  

fine_tuned_model_nbs.pt  

best_model_state.pt  

Performance  

Text-only (BERT) → F1 = 0.82  

Multimodal (BERT + ResNet) → F1 = 0.65  

Fusion (XGBoost) → F1 = 0.65  

Installation
git clone https://github.com/yourusername/news-bias-classifier.git  
cd news-bias-classifier  
pip install -r requirements.txt  

Usage  

Run notebooks for training:  

jupyter notebook Babe_Dataset.ipynb  
jupyter notebook News_Media_Dataset.ipynb  


Run dashboard:  

python news_bias_app.py  

Results & Conclusion  

Deep learning models effectively detect news bias.  

Due to  limited dataset and computational power Multi Modal and ensembled performed bad

LLMs moderately successful at debiasing while preserving grammar and content.  

FastAPI+Gradio app demonstrates practical deployment.  
