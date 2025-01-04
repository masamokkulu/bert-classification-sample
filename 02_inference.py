import os, random, sys, japanize_matplotlib, torch, tqdm, ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from transformers.trainer_utils import set_seed

# ENVIRONMENT
MODEL_INPUT = "./model"
SAMPLE_FILE = "./sample_review.txt"

# サンプル読み込み
with open(SAMPLE_FILE, 'r', encoding='utf-8') as file:
    data = file.read()
sample_review = ast.literal_eval(data)

# 利用するデバイスの確認（GPU or CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "gpu")

# モデルの読み込み
model = (AutoModelForSequenceClassification.from_pretrained(MODEL_INPUT, ignore_mismatched_sizes=True).to(device))

# tokenizerの読み込み
tokenizer = AutoTokenizer.from_pretrained(MODEL_INPUT)

# Pipelinesの定義
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

# 推論実行
sample_results = pipe(sample_review)

# 結果の表示
for i in range(len(sample_review)):
    print(f"REVIEW: {sample_review[i]}")
    print(f"INFERENCE: {sample_results[i]['label']}")
    print(f"SCORE: {sample_results[i]['score']:.4f}")
    print()
