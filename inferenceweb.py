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

def classification_review(input_text):
    # サンプル読み込み
    sample_review = [ input_text ]

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

    # 結果を app に返却する
    return sample_review[0], sample_results[0]['label']
