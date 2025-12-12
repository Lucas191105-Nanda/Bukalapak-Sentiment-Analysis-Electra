# ============================================================================
# SENTIMENT ANALYZER - UPLOAD DATASET VERSION
# - No scraping / no Apify
# - User uploads CSV or JSON containing a column named 'review'
# - Model: Electra (local) - change MODEL_PATH to your model directory
# ============================================================================

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
import torch
from transformers import ElectraForSequenceClassification, ElectraTokenizerFast
import pandas as pd
import json
import io

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "electra_model_local"  # <-- ganti jika perlu
BATCH_SIZE = 16

# ============================================================================
# LOAD TOKENIZER & MODEL
# ============================================================================
print("Loading tokenizer and model...")
try:
    tokenizer = ElectraTokenizerFast.from_pretrained(MODEL_PATH)
    model = ElectraForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Model loaded on: {device}")
except Exception as e:
    print("❌ Failed to load model/tokenizer:", e)
    raise


# ============================================================================
# PREDICTION HELPERS
# ============================================================================

def predict_batch(texts):
    """Predict sentiment labels for a list of texts.
    Assumes model outputs logits and class 1 = positive, 0 = negative.
    """
    preds = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
            logits = out.logits
            batch_preds = torch.argmax(logits, dim=1).cpu().tolist()

        preds.extend(batch_preds)
    return preds


def analyze_reviews(reviews):
    """Analyze a list of review texts and return statistics + samples."""
    if not reviews:
        return {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "positive_percent": 0,
            "negative_percent": 0,
            "sample_positive": [],
            "sample_negative": [],
        }

    preds = predict_batch(reviews)

    positives = [r for r, p in zip(reviews, preds) if p == 1]
    negatives = [r for r, p in zip(reviews, preds) if p == 0]

    total = len(reviews)
    pos = len(positives)
    neg = len(negatives)

    return {
        "total": total,
        "positive": pos,
        "negative": neg,
        "positive_percent": round((pos / total) * 100, 2) if total > 0 else 0,
        "negative_percent": round((neg / total) * 100, 2) if total > 0 else 0,
        "sample_positive": positives[:3],
        "sample_negative": negatives[:3]
    }


# ============================================================================
# FASTAPI APP - Upload-based flow
# ============================================================================
app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def home():
    html = """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Upload Dataset - Sentiment Analyzer</title>
        <style>
          * { box-sizing: border-box; }
          body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial; background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); min-height:100vh; padding:20px }
          .container { max-width:720px; margin:40px auto; background:white; padding:30px; border-radius:12px; box-shadow:0 20px 40px rgba(0,0,0,0.15) }
          h2 { color:#ee4d2d }
          p { color:#444 }
          .note { font-size:13px; color:#666 }
          input[type=file] { width:100%; padding:10px; margin:18px 0 }
          button { background:#ee4d2d; color:white; border:none; padding:12px 18px; border-radius:8px; cursor:pointer; font-weight:600 }
        </style>
      </head>
      <body>
        <div class="container">
          <h2>📁 Upload Dataset Reviews</h2>
          <p>Upload CSV atau JSON. File harus berisi kolom <b>review</b> (case-sensitive).</p>

          <form action="/analyze-file" enctype="multipart/form-data" method="post">
            <input type="file" name="file" accept=".csv,.json" required />
            <button type="submit">🔍 Analyze</button>
          </form>

          <div style="margin-top:18px;" class="note">Contoh CSV:
            <pre>review\n"Bagus kualitasnya"\n"Tidak sesuai foto"</pre>
          </div>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/analyze-file", response_class=HTMLResponse)
async def analyze_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename.lower()

        # --- CSV ---
        if filename.endswith('.csv'):
            try:
                df = pd.read_csv(io.BytesIO(content))
            except Exception as e:
                raise HTTPException(400, f"Gagal membaca CSV: {e}")

        # --- JSON ---
        elif filename.endswith('.json'):
            try:
                data = json.loads(content.decode('utf-8'))
            except Exception as e:
                raise HTTPException(400, f"Gagal membaca JSON: {e}")

            # Support a few JSON shapes
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # If root contains array under a key (e.g. {"reviews": [...]})
                if any(isinstance(v, list) for v in data.values()):
                    # pick the first list-like value
                    for v in data.values():
                        if isinstance(v, list):
                            df = pd.DataFrame(v)
                            break
                else:
                    # Single dict -> make single-row dataframe
                    df = pd.DataFrame([data])
            else:
                raise HTTPException(400, "Format JSON tidak dikenali atau rusak")

        else:
            raise HTTPException(400, "Format file harus .csv atau .json")

        # Ensure 'review' column exists
        if 'review' not in df.columns:
            # try common alternatives
            possible = [c for c in df.columns if 'review' in c.lower() or 'text' in c.lower() or 'comment' in c.lower()]
            if possible:
                # choose first match and rename to 'review'
                df = df.rename(columns={possible[0]: 'review'})
            else:
                raise HTTPException(400, "File harus memiliki kolom 'review' atau kolom yang mengandung 'review/text/comment'")

        reviews = df['review'].dropna().astype(str).tolist()

        if len(reviews) == 0:
            raise HTTPException(400, "Kolom 'review' kosong")

        analysis = analyze_reviews(reviews)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Error saat memproses file: {str(e)}")

    # Prepare sample html blocks
    sample_positive = "<br><br>".join([
        f'<div style="padding:10px; background:#e8f5e9; border-radius:5px; margin:5px 0">"{r}"</div>'
        for r in analysis["sample_positive"]
    ]) if analysis["sample_positive"] else "<em>Tidak ada review positif</em>"

    sample_negative = "<br><br>".join([
        f'<div style="padding:10px; background:#ffebee; border-radius:5px; margin:5px 0">"{r}"</div>'
        for r in analysis["sample_negative"]
    ]) if analysis["sample_negative"] else "<em>Tidak ada review negatif</em>"

    pos_width = analysis['positive_percent']
    neg_width = analysis['negative_percent']

    html = f"""
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8" />
        <title>Hasil Analisis - Sentiment</title>
        <style>
          * {{ margin:0; padding:0; box-sizing:border-box }}
          body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial; background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); min-height:100vh; padding:20px }}
          .container {{ max-width:900px; margin:40px auto; background:white; border-radius:20px; padding:30px }}
          h2 {{ color:#333 }}
          .stats {{ display:grid; grid-template-columns:repeat(3,1fr); gap:20px; margin-bottom:20px }}
          .card {{ padding:18px; border-radius:12px; text-align:center }}
          .card.total {{ background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white }}
          .card.pos {{ background:linear-gradient(135deg,#11998e 0%,#38ef7d 100%); color:white }}
          .card.neg {{ background:linear-gradient(135deg,#ee0979 0%,#ff6a00 100%); color:white }}
          .stat-number {{ font-size:32px; font-weight:700 }}
          .sentiment-bar {{ height:28px; border-radius:14px; overflow:hidden; display:flex; box-shadow:inset 0 2px 4px rgba(0,0,0,0.08); margin-bottom:20px }}
          .bar-pos {{ height:100%; background:#38ef7d }}
          .bar-neg {{ height:100%; background:#ff6a00 }}
          .section {{ margin-top:18px }}
          .back {{ display:inline-block; margin-top:20px; padding:10px 16px; background:#ee4d2d; color:white; border-radius:8px; text-decoration:none }}
        </style>
    </head>
    <body>
      <div class="container">
        <h2>📊 Hasil Analisis Sentimen</h2>

        <div class="stats">
          <div class="card total">
            <div class="stat-number">{analysis['total']}</div>
            <div>Total Reviews</div>
          </div>
          <div class="card pos">
            <div class="stat-number">{analysis['positive']}</div>
            <div>Positive ({analysis['positive_percent']}%)</div>
          </div>
          <div class="card neg">
            <div class="stat-number">{analysis['negative']}</div>
            <div>Negative ({analysis['negative_percent']}%)</div>
          </div>
        </div>

        <div class="sentiment-bar">
          <div class="bar-pos" style="width: {pos_width}%"></div>
          <div class="bar-neg" style="width: {neg_width}%"></div>
        </div>

        <div class="section">
          <h3>✅ Sample Positive Reviews</h3>
          {sample_positive}
        </div>

        <div class="section">
          <h3>❌ Sample Negative Reviews</h3>
          {sample_negative}
        </div>

        <a class="back" href="/">← Upload File Lain</a>
      </div>
    </body>
    </html>
    """

    return HTMLResponse(content=html)


# ============================================================================
# RUN SERVER
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    print("Starting Sentiment Analyzer (upload version) on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
