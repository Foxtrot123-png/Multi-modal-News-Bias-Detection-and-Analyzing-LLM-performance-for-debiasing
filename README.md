# Multi-Modal News Bias Detection & LLM Debiasing

My MSc dissertation. The question I wanted to answer: can you train a model to spot bias in news articles — and then fix it?

Turned out yes, mostly. Here's what I built and what I found.

---

## The idea

Most bias detection tools only look at words. But news bias isn't just about what you say — it's about how you frame it. The photo you pick. The headline you write. The quotes you include or leave out.

So I built a system that looks at both text and images together. Then once it finds bias, it uses an LLM to rewrite the article in a more neutral way.

---

## What I built

**Text classifier** — fine-tuned BERT on the BABE dataset (expert-labelled news articles). Hit F1 of 0.82 which I was pretty happy with.

**Multi-modal classifier** — added ResNet-34 for images and connected them with a cross-attention layer so the model could weigh both signals together. F1 dropped to 0.65 — mainly because the dataset was too small for the multi-modal approach to really shine.

**Fusion model** — XGBoost combining both classifiers. Useful for batch analysis of large article sets.

**LLM debiasing** — used GPT-2 and Gemini to rewrite biased articles. The goal was to keep all the facts but strip out the loaded framing. It worked better than I expected, especially Gemini.

**Dashboard** — FastAPI + Gradio app where you can paste in an article and get a bias score plus a rewritten neutral version.

---

## Results

| Model | F1 |
|-------|----|
| BERT text-only | 0.82 |
| BERT + ResNet multi-modal | 0.65 |
| XGBoost fusion | 0.65 |

The text model beat the multi-modal one — not because images don't matter but because I didn't have enough labelled image-text pairs to train it properly. With a bigger dataset I think multi-modal wins.

---

## Stack

BERT, ResNet-34, XGBoost, GPT-2, Gemini API, FastAPI, Gradio, PyTorch, HuggingFace

---

## Run it

```bash
git clone https://github.com/Foxtrot123-png/Multi-modal-News-Bias-Detection-and-Analyzing-LLM-performance-for-debiasing
cd Multi-modal-News-Bias-Detection-and-Analyzing-LLM-performance-for-debiasing
pip install -r requirements.txt

# train
jupyter notebook Babe_Dataset.ipynb

# run dashboard
python news_bias_app.py
```

---

## Honest reflections

The multi-modal result was disappointing at first but it taught me something useful — cross-attention is only as good as the data you feed it. The architecture was right, the dataset was the constraint.

The LLM debiasing part surprised me most. I expected it to hallucinate or lose facts. It mostly didn't. Gemini in particular was good at keeping the substance while changing the tone.

If I were doing this again I'd spend 80% of the time on data collection and 20% on the model. Every problem I hit came back to data quality.

---

## What's next

Bigger dataset. Fine-tuned Gemini instead of prompting it. Video news clips eventually.

---

Built by Ritik R Mohapatra — ex-Deloitte, MSc Data Science Distinction, Herts 2025.

[LinkedIn](https://www.linkedin.com/in/ritik-r-mohapatra) · [GitHub](https://github.com/Foxtrot123-png) · [Live project](https://ritik-ai-twin.streamlit.app/)
