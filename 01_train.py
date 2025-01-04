#!/usr/bin/python3
import os, random, sys, japanize_matplotlib, torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed

# ENVIRONMENT
DATASET_PATH = "shunk031/JGLUE"
MODEL_NAME = "tohoku-nlp/bert-base-japanese-v3"
# MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
TRAIN_LOG_OUTPUT = "./train-log"
MODEL_OUTPUT = "./model"
training_args = TrainingArguments(
    output_dir=TRAIN_LOG_OUTPUT,                # モデルの保存先
    num_train_epochs=3,                         # エポック数
    learning_rate=2e-5,                         # 学習率
    per_device_train_batch_size=32,             # 学習時のバッチサイズ
    per_device_eval_batch_size=32,              # 評価時のバッチサイズ
    save_strategy="epoch",                      # モデルの保存タイミング
    logging_strategy="epoch",                   # ログの出力タイミング
    evaluation_strategy="epoch",                # 評価のタイミング
    optim="adafactor",                          # 最適化手法
    gradient_accumulation_steps=4,              # 勾配蓄積のステップ数
    load_best_model_at_end=True,                # 最良のモデルを最後に読み込むかどうか
    metric_for_best_model="balanced_accuracy",  # 最良のモデルを判断する指標
    fp16=True,                                  # 16bit精度を利用するかどうか
    overwrite_output_dir=True,                  # 出力先のディレクトリを上書きするかどうか
)

# データセットのトークナイズ
def tokenize_dataset(example) -> dict:
    example_output = tokenizer(example["sentence"], truncation=True)
    example_output["label"] = example["label"]
    return example_output

# データの確認
def show_label_count(dataset: dict) -> None:
    dataset.set_format(type="pandas")
    df = dataset[:]
    print(df.value_counts(["label"]))
    dataset.reset_format()

# 成果指標確認
def compute_metrics(eval_pred) -> dict:
    labels = eval_pred.label_ids
    predicts = eval_pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicts, average="weighted")
    acc = accuracy_score(labels, predicts)
    balanced_acc = balanced_accuracy_score(labels, predicts)
    return {"accuracy": acc, "balanced_accuracy": balanced_acc, "f1": f1, "precision": precision, "recall": recall}

# ディレクトリ確認
if not os.path.exists(TRAIN_LOG_OUTPUT):
     os.makedirs(TRAIN_LOG_OUTPUT)
     print(f"[INFO] Created {TRAIN_LOG_OUTPUT} directory.")

if not os.path.exists(MODEL_OUTPUT):
     os.makedirs(MODEL_OUTPUT)
     print(f"[INFO] Created {MODEL_OUTPUT} directory.")

print("[INFO] Loading dataset ...")
train_dataset = load_dataset(
    DATASET_PATH,
    name="MARC-ja",
    split="train",
    trust_remote_code=True
)

valid_dataset = load_dataset(
    DATASET_PATH,
    name="MARC-ja",
    split="validation",
    trust_remote_code=True
)

classes = train_dataset.features["label"]
id2label = {}
label2id = {}
for i in range(train_dataset.features["label"].num_classes):
    id2label[i] = train_dataset.features["label"].int2str(i)
    label2id[train_dataset.features["label"].int2str(i)] = i

# モデルの読み込み
print("[INFO] Loading model ...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=classes.num_classes,
    label2id=label2id,
    id2label=id2label
)

# トークナイザの準備
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# トークン化の実行
tokenized_train_datasets = train_dataset.map(tokenize_dataset, batched=True)
tokenized_valid_datasets = valid_dataset.map(tokenize_dataset, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Fine Turning 実行
trainer = Trainer(
    model=model,                             # 利用するモデル
    args=training_args,                      # 学習時の設定
    train_dataset=tokenized_train_datasets,  # 訓練データ
    eval_dataset=tokenized_valid_datasets,   # 評価データ
    tokenizer=tokenizer,                     # トークナイザ
    data_collator=data_collator,             # データの前処理
    compute_metrics=compute_metrics,         # 評価指標の計算
)

print("[INFO] Train Start ...")
trainer.train()

# 評価指標の確認
eval_metrics = trainer.evaluate(tokenized_valid_datasets)
eval_metrics
# モデルの保存
trainer.save_model(MODEL_OUTPUT)
print("[INFO] SUCCESS!")
