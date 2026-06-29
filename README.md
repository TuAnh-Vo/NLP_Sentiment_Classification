# NLP Sentiment Classification 

Three-class sentiment classification (Negative `-1`, Neutral `0`, Positive `+1`) over a merged corpus of Amazon product reviews (Ni et al. 2019) and SemEval-2017 Task 4 tweets (Rosenthal et al. 2017). The deliverable is a reproducible pipeline plus a fine-tuned transformer served behind a Flask/gunicorn REST API on Hugging Face Spaces, scored on a private test set by macro-F1.

**VU Amsterdam — Data Science Project 2026 (E_EDS2_DSPT)**
Group members: Anh Vo Nhu Tu · Igor Mamyan · Kavya Kinnera · Mika van Straaten

---

## Results

| Route | Model | Test Macro-F1 | Neutral F1 | Review F1 | Tweet F1 |
|-------|-------|:---:|:---:|:---:|:---:|
| A | TF-IDF + Logistic Regression | 0.6969 | 0.54 | 0.6837 | 0.5640 |
| B | FastText-300 + mean pooling + LR | 0.6322 | 0.48 | 0.6146 | 0.5096 |
| B ext | Sentence Transformer (MiniLM) + LR | 0.6395 | 0.48 | 0.6124 | 0.5630 |
| C | BERT-base-uncased (fine-tuned) | 0.7514 | 0.60 | 0.7388 | 0.7075 |
| C | BERTweet-base + twitter-RoBERTa ensemble | 0.7580 | 0.60 | 0.7420 | 0.7424 |
| **C** | **BERTweet-large (fine-tuned)** | **0.7644** | **0.61** | **0.7490** | **0.7130** |

**Leaderboard (deployed BERTweet-large API):** 2nd on Reviews (0.74) · 4th on Tweets (0.63)

Two findings recur across every route: the **Neutral class** is the binding bottleneck (it has no vocabulary of its own — it is the absence of strong sentiment), and the **review-vs-tweet domain gap** shrinks as model capacity grows, from 0.12 in Route A to 0.0004 for the Route C ensemble.

---

## Deployed APIs (Hugging Face Spaces)

| Model | Endpoint |
|-------|----------|
| BERTweet-large *(best)* | https://modelling-giants-bertweet-large.hf.space/ |
| BERTweet-base + twitter-RoBERTa ensemble | https://modelling-giants-ensemble-threshold-mg.hf.space/ |
| BERT-base | https://modellinggiants-bertje.hf.space/ |

```bash
# metadata
curl https://modelling-giants-bertweet-large.hf.space/

# prediction
curl -X POST https://modelling-giants-bertweet-large.hf.space/ \
  -H "Content-Type: application/json" \
  -d '{"items": [{"id": "1", "text": "I absolutely love this product!"}]}'
```

---

## Approach

The project is organised into three independent modelling routes of increasing representational power. Each route loads the shared split (`data_split.csv`) on its own and applies its own preprocessing, so no decision in one route leaks into another. Preprocessing intensity decreases as model complexity increases: heavy normalization for the count-based route, minimal cleaning for the transformers that depend on casing, punctuation, and word order.

**Route A — TF-IDF + classical classifiers**
Sparse TF-IDF features (unigrams + bigrams, capped at 30k terms) with chi-squared feature selection. Classifiers: Complement Naïve Bayes (baseline), Logistic Regression, LinearSVC. Two cleaning pipelines ("light" / "rich") are compared. Winner: light + TF-IDF + Logistic Regression (C=0.5, balanced class weights).

**Route B — static word embeddings**
Dense document vectors from GloVe-Twitter-200, Word2Vec-300, and FastText-300, pooled by mean or IDF-weighted mean. Classifiers: Logistic Regression, LinearSVC. Extension: contextual sentence embeddings from `all-MiniLM-L6-v2` (frozen encoder + LR head). Winner: FastText-300 + mean pooling + LR (C=0.1, balanced class weights).

**Route C — fine-tuned transformers**
End-to-end fine-tuning of transformer encoders on the sentiment objective:
- `bert-base-uncased` — general-purpose baseline
- BERTweet-base + twitter-RoBERTa **late-fusion ensemble**; posteriors combined by log-average, with per-class threshold offsets (δ) tuned by coordinate ascent on validation macro-F1 to correct Neutral under-prediction
- `vinai/bertweet-large` — RoBERTa-large pre-trained on 873M tweets; **best single model**

---

## Repository layout

```
.
├── EDA.ipynb                        # DB query, cleaning, corpus stats, writes data_split.csv
├── route_a.ipynb                    # TF-IDF + chi2 + classifiers; pickles route_a.model
├── route_b.ipynb                    # Static embeddings (GloVe / Word2Vec / FastText) + classifiers
├── route_b_extension.ipynb          # Sentence transformer (MiniLM) + classifiers
├── route_c_bert_base_uncased.ipynb  # BERT-base fine-tuning
├── route_c_ensemble.ipynb           # BERTweet-base + twitter-RoBERTa ensemble training
├── route-c-ensemble-deploy.ipynb    # Ensemble deployment packaging
├── route-c-bertweet-large.ipynb     # BERTweet-large fine-tuning + deployment
├── data_split.csv                   # Shared 70/15/15 split, stratified on sentiment × Type
├── app.py                           # Flask REST API
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Data

The source database `nlp-data.db` is **not** redistributed — it lives on the VU Compute Hub at `/local/DSPT/data/nlp-data.db` (read-only) and is queried directly by `EDA.ipynb`.

`data_split.csv` is the derived split artefact (columns: `clean_eda`, `sentiment`, `Type`, `split`) and is committed so all routes reproduce without database access. SQL joins return N = 255,083 documents (205,000 Amazon reviews + 50,083 SemEval tweets). The split is stratified jointly on `sentiment × Type` (70/15/15): n_train = 178,557, n_val = n_test = 38,263.

---

---

## Reproduce

Set up Python 3.11 and install dependencies:
```bash
pip install -r requirements.txt
```
Run order:
1. `EDA.ipynb` — optional if using the committed `data_split.csv`; otherwise regenerates it.
2. `route_a.ipynb` — trains, cross-validates, evaluates, pickles `route_a.model`.
3. `route_b.ipynb` — downloads embedding vectors, evaluates all embedding × pooling × classifier combinations.
4. `route_b_extension.ipynb` — encodes the corpus with MiniLM, trains LR/SVC heads.
5. `route_c_bert_base_uncased.ipynb` — fine-tunes BERT-base **(GPU required)**.
6. `route_c_ensemble.ipynb` — fine-tunes the two ensemble members, tunes threshold offsets **(GPU required)**.
7. `route-c-ensemble-deploy.ipynb` — packages the ensemble for Hugging Face Spaces.
8. `route-c-bertweet-large.ipynb` — fine-tunes BERTweet-large, packages and uploads **(GPU required)**.

> **GPU note:** Route C training was done on Kaggle (T4, free tier). VU Compute Hub nodes are CPU-only and unsuitable for Route C training, though they can serve the pickled model.

---

---

## Deployment contract

Every packaged `.model` file is a pickled dict with exactly two keys:

```python
deployment = {
    "vectorizer": <fitted encoder>,   # TfidfVectorizer / embedding wrapper / transformer wrapper
    "classifier": <trained model>,
}
```
---

## Key engineering decisions

| Decision | Why |
|----------|-----|
| Stratify on `sentiment × Type` | Tweets are ~45% Neutral vs ~20% for reviews; a plain random split would not preserve both axes |
| Macro-F1 as primary metric | Neutral (24.9% of data) is the bottleneck; accuracy would hide poor Neutral recall |
| Fit IDF / scalers / thresholds on train only | Prevents corpus statistics from leaking into val/test |
| `transform`, not `fit_transform`, at inference | Re-fitting per request destroys learned weights |
| Light beats rich preprocessing (Route A) | IDF already suppresses low-information terms; extra cleaning removes signal |
| fp16 storage, fp32 inference (BERTweet-large) | Halves file size (~713 MB); upcast to fp32 before CPU forward passes |
| Ensemble via log-average of posteriors | Geometric mean is conservative — high confidence only when both members agree |
| Per-class threshold offsets (δ) | `argmax(log p_ens(c) + δ_c)`, δ tuned on validation; corrects systematic Neutral under-prediction |

---

## References

- Ni, J., Li, J., & McAuley, J. (2019). Justifying recommendations using distantly-labeled reviews and fine-grained aspects. *EMNLP*.
- Rosenthal, S., Farra, N., & Nakov, P. (2017). SemEval-2017 Task 4: Sentiment analysis in Twitter. *SemEval-2017*.
- Sebastiani, F. (2002). Machine learning in automated text categorization. *ACM Computing Surveys* 34(1), 1–47.
- Nguyen, D. Q., Vu, T., & Nguyen, A. T. (2020). BERTweet: A pre-trained language model for English tweets. *EMNLP (Demos)*.
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. *EMNLP*.
- Mikolov, T., et al. (2013). Efficient estimation of word representations in vector space. *ICLR Workshop*.
- Bojanowski, P., et al. (2017). Enriching word vectors with subword information. *TACL* 5, 135–146.
- Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP*.
- Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT*.

- Nguyen, D. Q., Vu, T., & Nguyen, A. T. (2020). BERTweet: A pre-trained
  language model for English tweets. *EMNLP (Demos)*.
