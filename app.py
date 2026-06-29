"""
Sentiment Classification API — Flask + gunicorn

Loads a packaged model file and serves three-class sentiment predictions
(Negative -1 / Neutral 0 / Positive +1) over a simple JSON protocol.

Deployment contract (per the DSPT case manual):
    The .model file is a pickled dict with exactly two keys:
        {"vectorizer": <fitted encoder>, "classifier": <trained model>}

GET  /   -> returns model + environment metadata
POST /   -> body {"items": [{"id": ..., "text": ...}, ...]}
            returns {"items": [{"id": ..., "label": <int>}, ...]}
"""

import os
import pickle
import platform

from flask import Flask, jsonify, request
from flask_cors import CORS

# ────────────────────────────────────────────────────────────
# Custom classes used inside the pickle MUST be importable here,
# BEFORE pickle.load runs, or deserialization will fail in this
# separate API process.
#
#   from sentiment_deploy import BERTweetClassifier
#
# Route A (plain sklearn) needs no extra import.
# ────────────────────────────────────────────────────────────

# ── Configuration ───────────────────────────────────────────
GROUP_ID      = "modelling-giants"
MODEL_FILE    = "bertweet_large.model"   # path to your packaged model
MODEL_VERSION = "v1.0"


def batch_predict(model, items):
    """Predict a sentiment label for each item in the batch.

    IMPORTANT: use transform(), never fit_transform(). fit_transform
    re-fits the encoder on every request and destroys the learned
    vocabulary / IDF weights, silently corrupting predictions.
    """
    results = []
    for item in items:
        X = model["vectorizer"].transform([item["text"]])
        label = model["classifier"].predict(X)
        results.append({"id": item["id"], "label": int(label[0])})
    return results


# ── App setup ───────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

meta_data = {
    "groupID": GROUP_ID,
    "modelFile": MODEL_FILE,
    "modelVersion": MODEL_VERSION,
    "pythonVersion": platform.python_version(),
}


@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        items = request.json["items"]
        return jsonify({"items": batch_predict(model, items)})
    return jsonify({"meta": meta_data})


if __name__ == "__main__":
    # Cloud platforms (e.g. Hugging Face) inject PORT; default to 8000 locally.
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
