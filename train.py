import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import os
import collections
import torch
import gc
from transformers import BertConfig, BertTokenizer
from transformers import BertModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
import shutil
from sklearn.metrics import classification_report
import json
import bz2
import nltk
import string
import re
import time

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", default=None, required=False)
args = parser.parse_args()
RUN_NAME = args.run_name
if RUN_NAME is None:
    RUN_NAME = datetime.now().strftime("%Y%m%d_%H%M%S")

LABELS_MAP = {"amazon": {0: "positive", 1: "negative"}}
NUM_LABELS = 2


stopwords = nltk.corpus.stopwords.words("english")
url_pattern = r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
html_pattern = r"<[^>]+>"


def preprocess_text(text: str):
    text = text.strip()
    text = " ".join(text.split())
    text = re.sub(url_pattern, "", text)
    text = re.sub(html_pattern, "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [token for token in tokens if token.lower() not in stopwords]
    text = " ".join(tokens)
    return text


def prepare_data(dirpath: str, filename_train: str, filename_test: str):
    for idx, filename in enumerate([filename_train, filename_test]):
        filepath_extracted = os.path.join(dirpath, filename.replace(".bz2", ""))
        if os.path.exists(filepath_extracted + ".pkl"):
            df = pd.read_pickle(filepath_extracted + ".pkl")
        else:
            if not os.path.exists(filename.replace(".bz2", "")):
                filepath = os.path.join(dirpath, filename)
                with open(filepath_extracted, "wb") as new_file, bz2.BZ2File(
                    filepath, "rb"
                ) as file:
                    for data in iter(lambda: file.read(100 * 1024), b""):
                        new_file.write(data)
            df = pd.read_fwf(filepath_extracted, header=None, widths=[10, 2048])
            df.columns = ["label", "text"]
            df.label = df.label.apply(lambda item: 1 if item == "__label__1" else 0)
            df.text = df.text.apply(lambda item: preprocess_text(item))
            df = df.drop_duplicates(ignore_index=True)
            df.to_pickle(filepath_extracted + ".pkl")
        print(df.head())
        df.label = df.label.apply(lambda item: LABELS_MAP["amazon"][item])
        if idx == 0:
            train = df
        else:
            test, valid = train_test_split(
                df, test_size=0.5, random_state=1, stratify=df["label"]
            )
    return train, test, valid


train, test, valid = prepare_data(
    dirpath="./data", filename_train="train.ft.txt.bz2", filename_test="test.ft.txt.bz2"
)

train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)
test = test.reset_index(drop=True)

train = train[:50000]
test = test[:500]
valid = valid[:1000]


#################################### Configuration ####################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print("CUDA is not available.  Training on CPU ...")
else:
    print("CUDA is available!  Training on GPU ...")

# general config
MAX_LEN = 512
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8

EPOCHS = 200
EEVERY_EPOCH = 10 # train.shape[0] // TRAIN_BATCH_SIZE // 100  # 1000
LEARNING_RATE = 1e-3
CLIP = 0.0
WEIGHT_DECAY = 1e-4  # Regularization
DROPOUT_PROB = 0.3
OPTIMIZER = "SGD"

MODEL_NAME_OR_PATH = "models/bert-base-uncased"
OUTPUT_DIR = f"outputs/{RUN_NAME}"
RUN_PATH = os.path.join("runs", RUN_NAME)
CHECKPOINT_PATH = os.path.join("checkpoints", RUN_NAME, "checkpoint.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RUN_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

# Writer will output to ./runs/ directory by default
writer = SummaryWriter(RUN_PATH, flush_secs=60)

run_info = {
    "runname": RUN_NAME,
    "maxlen": MAX_LEN,
    "bsize": TRAIN_BATCH_SIZE,
    "epochs": EPOCHS,
    "lr": LEARNING_RATE,
    "clip": CLIP,
    "weightdecay": WEIGHT_DECAY,
    "dropoutprob": DROPOUT_PROB,
    "optim": OPTIMIZER,
    "num_labels": NUM_LABELS,
    "scheduler": "linear_with_warmup",
    "trainlen": train.shape[0],
    "testlen": test.shape[0],
    "validlen": valid.shape[0],
}

id2label = {k: v for k, v in LABELS_MAP["amazon"].items()}
label2id = {v: k for k, v in id2label.items()}

# setup the tokenizer and configuration
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
config = BertConfig.from_pretrained(
    MODEL_NAME_OR_PATH,
    **{
        "label2id": label2id,
        "id2label": id2label,
    },
)


#################################### Dataset ####################################
class Dataset(torch.utils.data.Dataset):
    """Create a PyTorch dataset"""

    def __init__(self, tokenizer, texts, targets=None, label_list=None, max_len=128):
        self.texts = texts
        self.targets = targets
        self.has_target = isinstance(targets, list) or isinstance(targets, np.ndarray)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = (
            {label: i for i, label in enumerate(label_list)}
            if isinstance(label_list, list)
            else {}
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])

        if self.has_target:
            target = self.label_map.get(
                str(self.targets[item]), str(self.targets[item])
            )

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        inputs = {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "token_type_ids": encoding["token_type_ids"].flatten(),
        }

        if self.has_target:
            inputs["targets"] = torch.tensor(target, dtype=torch.long)

        return inputs


def create_data_loader(x, y, tokenizer, max_len, batch_size, label_list):
    dataset = Dataset(
        texts=x, targets=y, tokenizer=tokenizer, max_len=max_len, label_list=label_list
    )

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


#################################### ####################################
label_list = [k for k, v in label2id.items()]
train_data_loader = create_data_loader(
    train["text"].to_numpy(),
    train["label"].to_numpy(),
    tokenizer,
    MAX_LEN,
    TRAIN_BATCH_SIZE,
    label_list,
)
valid_data_loader = create_data_loader(
    valid["text"].to_numpy(),
    valid["label"].to_numpy(),
    tokenizer,
    MAX_LEN,
    VALID_BATCH_SIZE,
    label_list,
)
test_data_loader = create_data_loader(
    test["text"].to_numpy(), None, tokenizer, MAX_LEN, TEST_BATCH_SIZE, label_list
)


#################################### Model ####################################
class SentimentModel(nn.Module):
    def __init__(self, config):
        super(SentimentModel, self).__init__()

        self.bert = BertModel.from_pretrained(MODEL_NAME_OR_PATH)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(DROPOUT_PROB)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def state_dict(self):
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "counter": self.counter,
            "min_validation_loss": self.min_validation_loss,
        }

    def load_state_dict(self, state_dict: dict):
        self.patience = state_dict["patience"]
        self.min_delta = state_dict["min_delta"]
        self.counter = state_dict["counter"]
        self.min_validation_loss = state_dict["min_validation_loss"]

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


#################################### Training Prepare ####################################
gc.collect()
torch.cuda.empty_cache()
pt_model = None
pt_model = SentimentModel(config=config)
pt_model = pt_model.to(device)


def simple_accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def score(y_true, y_pred, average="weighted"):
    acc = simple_accuracy(y_true, y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average=average)
    precision = precision_score(y_true=y_true, y_pred=y_pred, average=average)
    recall = recall_score(y_true=y_true, y_pred=y_pred, average=average)
    return {"acc": acc, "f1": f1, "prec": precision, "recall": recall}


def y_loss(y_true, y_pred, losses):
    y_true = torch.stack(y_true).cpu().detach().numpy()
    y_pred = torch.stack(y_pred).cpu().detach().numpy()
    y = [y_true, y_pred]
    loss = np.mean(losses)

    return y, loss


def eval_op(model, data_loader, loss_fn):
    model.eval()

    losses = []
    y_pred = []
    y_true = []

    with torch.no_grad():
        # for dl in tqdm(data_loader, total=len(data_loader), desc="Evaluation"):
        for dl in data_loader:
            input_ids = dl["input_ids"]
            attention_mask = dl["attention_mask"]
            token_type_ids = dl["token_type_ids"]
            targets = dl["targets"]

            # move tensors to GPU if CUDA is available
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            targets = targets.to(device)

            # compute predicted outputs by passing inputs to the model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            # convert output probabilities to predicted class
            _, preds = torch.max(outputs, dim=1)

            # calculate the batch loss
            loss = loss_fn(outputs, targets)

            # accumulate all the losses
            losses.append(loss.item())

            y_pred.extend(preds)
            y_true.extend(targets)

    eval_y, eval_loss = y_loss(y_true, y_pred, losses)
    return eval_y, eval_loss


def eval_callback(epoch, epochs, output_dir):
    def eval_cb(
        model, step, train_score, train_loss, eval_score, eval_loss, eval_loss_min
    ):
        writer.add_scalars("loss", {"train": train_loss, "test": eval_loss}, step)
        writer.add_scalars(
            "acc", {"train": train_score["acc"], "test": eval_score["acc"]}, step
        )
        # writer.add_scalars(
        #     "f1", {"train": train_score["f1"], "test": eval_score["f1"]}, epoch
        # )

        if eval_loss <= eval_loss_min:
            torch.save(
                model.state_dict(), os.path.join(output_dir, "pytorch_model.bin")
            )
            run_info["tloss"] = train_loss
            run_info["vloss"] = eval_loss
            run_info["tacc"] = train_score["acc"]
            run_info["vacc"] = eval_score["acc"]
            run_info["tf1"] = train_score["f1"]
            run_info["vf1"] = eval_score["f1"]
            run_info["epoch"] = epoch
            run_info["step"] = step
            run_info["itime"] = datetime.now()
            for k, v in run_info.items():
                if type(v) == np.float64:
                    run_info[k] = round(v, 6)
            if os.path.exists(os.path.join("outputs", "run_info.csv")):
                df_run_info = pd.read_csv(os.path.join("outputs", "run_info.csv"))
                if (df_run_info["runname"] == RUN_NAME).any():
                    df_run_info.loc[df_run_info["runname"] == RUN_NAME, "tloss"] = (
                        run_info["tloss"]
                    )
                    df_run_info.loc[df_run_info["runname"] == RUN_NAME, "vloss"] = (
                        run_info["vloss"]
                    )
                    df_run_info.loc[df_run_info["runname"] == RUN_NAME, "tacc"] = (
                        run_info["tacc"]
                    )
                    df_run_info.loc[df_run_info["runname"] == RUN_NAME, "vacc"] = (
                        run_info["vacc"]
                    )
                    df_run_info.loc[df_run_info["runname"] == RUN_NAME, "tf1"] = (
                        run_info["tf1"]
                    )
                    df_run_info.loc[df_run_info["runname"] == RUN_NAME, "vf1"] = (
                        run_info["vf1"]
                    )
                    df_run_info.loc[df_run_info["runname"] == RUN_NAME, "epoch"] = (
                        run_info["epoch"]
                    )
                    df_run_info.loc[df_run_info["runname"] == RUN_NAME, "step"] = (
                        run_info["step"]
                    )
                    df_run_info.loc[df_run_info["runname"] == RUN_NAME, "itime"] = (
                        run_info["itime"]
                    )
                else:
                    df_run_info = pd.concat([df_run_info, pd.DataFrame([run_info])])
            else:
                df_run_info = pd.DataFrame([run_info])
            df_run_info.to_csv(os.path.join("outputs", "run_info.csv"), index=False)
            eval_loss_min = eval_loss
        return eval_loss_min

    return eval_cb


def train_op(
    model,
    data_loader,
    loss_fn,
    optimizer,
    scheduler,
    step=0,
    print_every_step=100,
    eval=False,
    eval_cb=None,
    eval_loss_min=np.Inf,
    eval_data_loader=None,
    clip=0.0,
):
    model.train()

    losses = []
    y_pred = []
    y_true = []

    for dl in tqdm(data_loader, total=len(data_loader), desc="Batch"):
    # for dl in data_loader:
        step += 1

        input_ids = dl["input_ids"]
        attention_mask = dl["attention_mask"]
        token_type_ids = dl["token_type_ids"]
        targets = dl["targets"]

        # move tensors to GPU if CUDA is available
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        targets = targets.to(device)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # compute predicted outputs by passing inputs to the model
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # convert output probabilities to predicted class
        _, preds = torch.max(outputs, dim=1)

        # calculate the batch loss
        loss = loss_fn(outputs, targets)

        # accumulate all the losses
        losses.append(loss.item())

        # compute gradient of the loss with respect to model parameters
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

        # perform optimization step
        optimizer.step()

        # perform scheduler step
        scheduler.step()

        y_pred.extend(preds)
        y_true.extend(targets)

        if eval:
            train_y, train_loss = y_loss(y_true, y_pred, losses)
            train_score = score(train_y[0], train_y[1], average="weighted")

            if step % print_every_step == 0:
                eval_y, eval_loss = eval_op(model, eval_data_loader, loss_fn)
                eval_score = score(eval_y[0], eval_y[1], average="weighted")

                if hasattr(eval_cb, "__call__"):
                    eval_loss_min = eval_cb(
                        model,
                        step,
                        train_score,
                        train_loss,
                        eval_score,
                        eval_loss,
                        eval_loss_min,
                    )
                time.sleep(300)
        time.sleep(30)

    train_y, train_loss = y_loss(y_true, y_pred, losses)

    return train_y, train_loss, step, eval_loss_min


#################################### Training ####################################
# optimizer = AdamW(
#     pt_model.parameters(),
#     lr=LEARNING_RATE,
#     correct_bias=False,
#     weight_decay=WEIGHT_DECAY,
# )
optimizer = SGD(
    pt_model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=WEIGHT_DECAY,
)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=len(train_data_loader) * 1,
    num_training_steps=total_steps,
)
early_stopper = EarlyStopper(patience=2, min_delta=0.01)

loss_fn = nn.CrossEntropyLoss()

step = 0
eval_loss_min = np.Inf
history = collections.defaultdict(list)

start_epoch = 1
if os.path.exists(CHECKPOINT_PATH):
    checkpoint_args = torch.load(CHECKPOINT_PATH)
    start_epoch = checkpoint_args["epoch"]
    step = checkpoint_args["step"]
    pt_model.load_state_dict(checkpoint_args["model_state_dict"])
    optimizer.load_state_dict(checkpoint_args["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint_args["scheduler"])
    early_stopper.load_state_dict(checkpoint_args["early_stopper"])
    eval_loss_min = checkpoint_args["eval_loss_min"]
    history = checkpoint_args["history"]
    print(f"Model loaded from {start_epoch=}, {step=}")


for epoch in tqdm(range(start_epoch, start_epoch + EPOCHS), desc="Epoch"):
    train_y, train_loss, step, eval_loss_min = train_op(
        model=pt_model,
        data_loader=train_data_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        print_every_step=EEVERY_EPOCH,
        eval=True,
        eval_cb=eval_callback(epoch, EPOCHS, OUTPUT_DIR),
        eval_loss_min=eval_loss_min,
        eval_data_loader=valid_data_loader,
        clip=CLIP,
    )

    train_score = score(train_y[0], train_y[1], average="weighted")
    eval_y, eval_loss = eval_op(
        model=pt_model, data_loader=valid_data_loader, loss_fn=loss_fn
    )
    eval_score = score(eval_y[0], eval_y[1], average="weighted")

    history["epoch"].append(epoch)
    history["tloss"].append(train_loss)
    history["vloss"].append(eval_loss)
    history["tacc"].append(train_score["acc"])
    history["vacc"].append(eval_score["acc"])
    history["tf1"].append(train_score["f1"])
    history["vf1"].append(eval_score["f1"])
    history["tprec"].append(train_score["prec"])
    history["vprec"].append(eval_score["prec"])
    history["trecall"].append(train_score["recall"])
    history["vrecall"].append(eval_score["recall"])
    history["itime"].append(datetime.now())

    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": pt_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "early_stopper": early_stopper.state_dict(),
            "eval_loss_min": eval_loss_min,
            "history": history,
        },
        CHECKPOINT_PATH,
    )
    pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "history.csv"), index=False)

    if early_stopper.early_stop(eval_loss):
        print("EARLY STOPPED")
        break

writer.flush()
writer.close()
pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "history.csv"), index=False)
shutil.rmtree(os.path.dirname(CHECKPOINT_PATH), ignore_errors=True)


#################################### Testing ####################################
def predict(model, texts, tokenizer, max_len=128, batch_size=32):
    data_loader = create_data_loader(texts, None, tokenizer, max_len, batch_size, None)
    predictions = []
    prediction_probs = []
    model.eval()
    with torch.no_grad():
        for dl in tqdm(data_loader, position=0):
            input_ids = dl["input_ids"]
            attention_mask = dl["attention_mask"]
            token_type_ids = dl["token_type_ids"]
            # move tensors to GPU if CUDA is available
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            # compute predicted outputs by passing inputs to the model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            # convert output probabilities to predicted class
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds)
            prediction_probs.extend(F.softmax(outputs, dim=1))
    predictions = torch.stack(predictions).cpu().detach().numpy()
    prediction_probs = torch.stack(prediction_probs).cpu().detach().numpy()
    return predictions, prediction_probs


# gc.collect()
# torch.cuda.empty_cache()
# pt_model = None
# pt_model = SentimentModel(config=config)
# pt_model = pt_model.to(device)
# pt_model.load_state_dict(os.path.join(RUN_NAME, ))

test_texts = test["text"].to_numpy()
preds, probs = predict(
    pt_model, test_texts, tokenizer, max_len=MAX_LEN, batch_size=TEST_BATCH_SIZE
)
# print(preds.shape, probs.shape)
y_test, y_pred = [label_list.index(label) for label in test["label"].values], preds
test_result = {
    "test_acc": simple_accuracy(y_test, y_pred),
    "test_f1": f1_score(y_test, y_pred, average="weighted"),
    "classification_report": classification_report(
        y_test, y_pred, target_names=label_list, output_dict=True
    ),
}
with open(os.path.join(OUTPUT_DIR, "test_result.json"), "w") as fp:
    json.dump(test_result, fp)
