Multi-Modal Model for Biased News Classification and Debiasing it using LLMs
===============================================================Project Overview===============================================================

In today’s digital age, getting information is easier than ever, but differentiating between factual reporting from biased narratives remains a challenge. This project tells how deep learning–based Natural Language Processing (NLP) can classify biased versus unbiased news articles. The primary objective is to design a detailed news bias detection system, while the secondary objective shows how Large Language Models (LLMs) can be applied for debiasing through prompting.
The methodology involved the pre-training of the Transformer Model (Bert) on GoodNews dataset and then using this pre-trained model as a base for the Text Bias Classification Model. This pre-trained model is also used with ResNet Model to create a model for multimodal classifier. After that LLMs were used for debiasing news articles. Exploratory Data Analysis (EDA) was conducted to understand feature distributions and means while model performance was assessed using Accuracy, Precision, Recall, and F1-Score metrics.
Results show that while the ensemble model has better than baseline performance (F1 = 0.653) , the text-only model achieved an F1 of 0.82 which is comparable to state-of-the-art classifiers. LLMs also showed moderate debiasing capability considering the similarity and grammatical accuracy of the generated text. This project highlights the potential of combining pre-trained transformers, multimodal fusion, and LLM prompting for practical bias detection and debiasing.

===============================================================Datasets===============================================================
The project utilizes 3 distinct  for training and evaluation:

BABE Dataset: A publicly available dataset from Kaggle containing news article text with expert-annotated bias labels. This dataset was used for training the text-only model.

Good News Dataset: Retrieved from the New York Times API, this dataset contains unlabelled images and their associated news articles. It was used for pre-training the BERT .

NewBiasDataset: Available on Zenodo, this dataset provides both images and text with their respective bias labels.

No_Corrupted_NBS.csv: A clean version of the NewBiasDataset which does not contain path to corrupted images.


===============================================================Text-Only Model (BABE)===============================================================
File: Babe_Dataset.IPYNB

Model: BERT Transformer

Methodology: This file contains  EDA, training, validation, and hyperparameter optimization.

Performance Metrics:

Accuracy: 75.37%

Precision (Biased/Non-Biased): 0.83

Recall (Biased/Non-Biased): 0.82

F1-Score (Biased/Non-Biased): 0.82

===============================================================Multimodal Model===============================================================
File: News_Media_Dataset.IPYNB

Model: BERT (Text) + ResNet-34 (Image)

Methodology: A multimodal approach using pre-trained weights to prevent random initialization. The model employs cross-attention for the fusion of text and image features. This file also covers EDA, training, validation, and hyperparameter optimization.

Performance Metrics:

Accuracy: 63.85%

Precision (Biased/Non-Biased): 0.705

Recall (Biased/Non-Biased): 0.597

F1-Score (Biased/Non-Biased): 0.65

===============================================================Model Fusion===============================================================
File: Fusion.IPYNB

Methodology: Predictions from the text-only and multimodal models are combined using an XGBoost classifier to produce a final, combined prediction.

Performance Metrics:

Precision (Biased/Non-Biased): 0.698

Recall (Biased/Non-Biased): 0.61

F1-Score (Biased/Non-Biased): 0.65
===============================================================LLM Debiasing===============================================================
File: LLMs.IPYNB

Models: GPT-2 and GEMINI

Methodology: This file is dedicated to evaluating the debiasing capabilities of LLMs, assessing their performance in transforming biased text into more neutral content.

===============================================================Saved Models===============================================================


BABE_fine_tuned_mdoel.pt: The saved weights for the text-only BERT model.

Model_config: Checkpoints and weights for the pre-trained BERT model.

fine_tuned_model_nbs.pt: The saved weights of the fine-tuned multimodal model from News_Media_Dataset.ipynb.

best_model_state.pt: The best model retrieved during the hyperparameter tuning of the multimodal model on the NewBiasDataset.

===============================================================Dashboard===============================================================
File: news_bias_app.py

Functionality:  dashboard for on-the-fly bias detection. For single-input predictions, it uses parametric concatenation of the models predictions instead of XGBoost. For batch inputs, the XGBoost fusion model is used.

