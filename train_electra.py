import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# 1. IMPORT LIBRARY
# ================================================================
import torch
import pandas as pd
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# ================================================================
# 2. KONFIGURASI MODEL
# ================================================================
MODEL_NAME = "ChristopherA08/IndoELECTRA"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device yang digunakan: {device}")


# ================================================================
# 3. LOAD DATASET LOKAL (VS CODE)
# ================================================================
# --- GANTI PATH LOKAL MU DI SINI ---
FILE_PATH = "train.csv"     # contoh: "D:/Project/dataset/train.csv"

file_extension = os.path.splitext(FILE_PATH)[1].lower()

if file_extension == '.csv':
    df = pd.read_csv(FILE_PATH)
elif file_extension == '.json':
    df = pd.read_json(FILE_PATH)
elif file_extension in ['.xlsx', '.xls']:
    df = pd.read_excel(FILE_PATH)
else:
    raise ValueError(f"Format file tidak didukung: {file_extension}")

print("Dataset Loaded.")
print(df.head())


# ================================================================
# 4. PREPROCESS DATA
# ================================================================
df = df.drop(columns=['header_review'])
df = df.dropna(subset=['review_sangat_singkat', 'label'])

if df['label'].dtype == 'object':
    encoder = LabelEncoder()
    df['label'] = encoder.fit_transform(df['label'])

df['label'] = df['label'].astype(int)

# TAMBAHKAN INI - Pastikan label mulai dari 0 dan berurutan
print(f"Label unik sebelum mapping: {sorted(df['label'].unique())}")
print(f"Jumlah label unik: {df['label'].nunique()}")

# Mapping ulang label ke 0, 1, 2, ... N-1
unique_labels = sorted(df['label'].unique())
label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
df['label'] = df['label'].map(label_mapping)

print(f"Label unik setelah mapping: {sorted(df['label'].unique())}")
print(f"Distribusi label:\n{df['label'].value_counts().sort_index()}")

# Validasi: pastikan tidak ada label >= num_labels
num_labels = int(df['label'].nunique())
assert df['label'].max() == num_labels - 1, f"Label max ({df['label'].max()}) harus sama dengan num_labels-1 ({num_labels-1})"
assert df['label'].min() == 0, f"Label min ({df['label'].min()}) harus 0"

dataset = Dataset.from_pandas(df[['review_sangat_singkat', 'label']])


# ================================================================
# 5. SPLIT DATA
# ================================================================
train_test = dataset.train_test_split(test_size=0.3, seed=42)
train_dataset = train_test['train']
test_dataset = train_test['test']


# ================================================================
# 6. LOAD TOKENIZER & MODEL
# ================================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
num_labels = int(df['label'].nunique())

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)

# CHECK & FIX VOCAB SIZE
print("\n=== VOCAB SIZE CHECK ===")
tokenizer_vocab_size = len(tokenizer)
model_vocab_size = model.config.vocab_size

print(f"Tokenizer vocab: {tokenizer_vocab_size}")
print(f"Model vocab: {model_vocab_size}")

if tokenizer_vocab_size != model_vocab_size:
    print(f"⚠️  MISMATCH! Resizing embeddings...")
    model.resize_token_embeddings(tokenizer_vocab_size)
    print(f"✓ Fixed! New size: {model.get_input_embeddings().weight.shape[0]}")
else:
    print(f"✓ Vocab sizes match")

model.to(device)


# ================================================================
# 7. TOKENISASI
# ================================================================
def preprocess(batch):
    return tokenizer(
        batch["review_sangat_singkat"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

tokenized_train = train_dataset.map(preprocess, batched=True)
tokenized_test = test_dataset.map(preprocess, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ================================================================
# 8. METRICS
# ================================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# ================================================================
# 9. TRAINING ARGUMENTS
# ================================================================
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_steps=50,
    bf16=True,  # ✅ Kalau RTX 3050/3060/3070/4060/4070
    fp16=False,  # Matikan FP16
    report_to="none"
)


# ================================================================
# 10. TRAINER
# ================================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# ================================================================
# 11. TRAIN MODEL
# ================================================================
print("\nTraining model...")
trainer.train()


# ================================================================
# 12. EVALUATION
# ================================================================
print("\nEvaluating model...")
results = trainer.evaluate(tokenized_test)
print(results)


# ================================================================
# 13. CONFUSION MATRIX
# ================================================================
pred_output = trainer.predict(tokenized_test)

y_pred = np.argmax(pred_output.predictions, axis=1)
y_true = pred_output.label_ids

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - ELECTRA")
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred))


# ================================================================
# 14. SAVE MODEL
# ================================================================
model.save_pretrained("./electra_model_local")
tokenizer.save_pretrained("./electra_model_local")

print("Model saved to ./electra_model_local")


# ================================================================
# 15. PREDIKSI MANUAL
# ================================================================
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        label_id = torch.argmax(probs).item()
        conf = probs[0][label_id].item()

    return label_id, conf


print("\nContoh prediksi:")
print(predict("Produk ini sangat bagus sekali!"))