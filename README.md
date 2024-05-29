# Sentiment Analysis
Simple sentiment analysis model using BERT

## Run this in a python interpreter
```
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')
```

## Dataset
Download the [Amazon Customer Reviews Dataset](https://www.kaggle.com/bittlingmayer/amazonreviews) from Kaggle.

## Config
Choose your desired model config in `configuration` part of the `train.py`.

## Train
Run this command in a terminal
```
tensorboard --logdir=runs --host 0.0.0.0
```
to view training logs in `http://0.0.0.0:6006`

Then start training
```
python3 train.py
```