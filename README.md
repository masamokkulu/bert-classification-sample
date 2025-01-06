# bert-classification-sample
BERTを用いた日本語の二項分類学習・推論を行うサンプル
## Requirements
* python == `3.11.10`
* DGX H100 にて検証
* そのほかの Python パッケージは `pip` からインストールしてください。
```bash
$ git clone git@github.com:masamokkulu/bert-classification-sample.git && cd bert-classification-sample
$ pip install -r requirements.txt
...
```
## Dataset & Model
* 東北大学 NLP チームが作成、公開している日本語で事前学習された BERT モデル `bert-base-japanese-v3` を使用
  * https://huggingface.co/tohoku-nlp/bert-base-japanese-v3
* ヤフーが公開している日本語分類のデータセット `MARC-ja` を使用（中身は Amazon などのレビューを収集したもの）
  * https://github.com/yahoojapan/JGLUE
## Quick Start
1. `01_train.py` スクリプトを実行することでモデルとデータセットの読み込みとファインチューニングを行います。ファインチューニングしたモデルは `./mode;` に保存されます。
```bash
$ python 01_train.py
...

{'eval_loss': 0.10683585703372955, 'eval_accuracy': 0.9593208348072161, 'eval_balanced_accuracy': 0.8752411941476934, 'eval_f1': 0.957326630970198, 'eval_precision': 0.9591395817470807, 'eval_recall': 0.9593208348072161, 'eval_runtime': 3.9352, 'eval_samples_per_second': 1436.77, 'eval_steps_per_second': 5.845, 'epoch': 3.0}
{'train_runtime': 662.0125, 'train_samples_per_second': 849.809, 'train_steps_per_second': 0.829, 'train_loss': 0.12223608854255606, 'epoch': 3.0}
100%|██████████| 23/23 [00:03<00:00,  6.21it/s]
[INFO] SUCCESS!
```
2. `02_inference.py` スクリプトを実行することで例文の二項分類を実行します。例文は `sample_review.txt` を編集することで変更可能です。
```bash
$ vim sample_review.txt
[
"シンプルながら最低限の機能は搭載しており、非常にコスパが良い製品だと思います。",
"購入して一週間で壊れてしまった。一応交換はしてもらえたが、リピートはナシかな。"
 <<< Add some reviews here!
] 

/* Start Inferenece */
$ python 02_inference.py
REVIEW: シンプルながら最低限の機能は搭載しており、非常にコスパが良い製品だと思います。
INFERENCE: positive
SCORE: 0.9988

REVIEW: 購入して一週間で壊れてしまった。一応交換はしてもらえたが、リピートはナシかな。
INFERENCE: negative
SCORE: 0.9787
```
## Web UI
参考程度にウェブから UI を起動し、推論を実行することが可能です。Flask == `3.1.0` を使用しています(`pip install ...` で自動的に導入されているはずです)。  
`app.py` を起動してください。その後、記載のアドレス・ポートでアクセスできるはずです。
```bash
$ python app.py
 * Serving Flask app 'app'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8000
 * Running on http://xxx.xxx.xxx.xxx:8000
...
```
