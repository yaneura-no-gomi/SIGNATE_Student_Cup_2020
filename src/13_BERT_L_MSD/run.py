import collections
import os
import random
import sys

import matplotlib.pyplot as plt
import nlp
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from transformers import AdamW, AutoModel, AutoTokenizer

sys.path.append("../utils")
from utils import check_submit_distribution, hack


# seeds
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    current_device = torch.cuda.current_device()
    print("Device:", torch.cuda.get_device_name(current_device))


# config
data_dir = os.path.join(
    os.environ["HOME"], "Workspace/learning/signate/SIGNATE_Student_Cup_2020/data"
)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAIN_FILE = os.path.join(data_dir, "train.csv")
TEST_FILE = os.path.join(data_dir, "test.csv")
MODELS_DIR = "./models/"
MODEL_NAME = "bert-large-uncased"
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 128
NUM_CLASSES = 4
EPOCHS = 10
NUM_SPLITS = 5


# dataset
def make_folded_df(csv_file, num_splits=5):
    df = pd.read_csv(csv_file)
    df["jobflag"] = df["jobflag"] - 1
    df["kfold"] = np.nan
    df = df.rename(columns={"jobflag": "labels"})
    label = df["labels"].tolist()

    skfold = StratifiedKFold(num_splits, shuffle=True, random_state=SEED)
    for fold, (_, valid_indexes) in enumerate(skfold.split(range(len(label)), label)):
        for i in valid_indexes:
            df.iat[i, 3] = fold
    return df


def make_dataset(df, tokenizer, device):
    dataset = nlp.Dataset.from_pandas(df)
    dataset = dataset.map(
        lambda example: tokenizer(
            example["description"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
    )
    dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
        device=device,
    )
    return dataset


# sampler for dataloader
class WeightedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, dataset, batch_size, n_classes, sampling_weight):
        loader = torch.utils.data.DataLoader(dataset)
        self.labels_list = []
        for i, batch in enumerate(loader):
            attention_mask, input_ids, labels, token_type_ids = batch.values()
            labels = labels.to("cpu").detach().numpy().copy()[0]
            self.labels_list.append(int(labels))

        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {
            label: np.where(self.labels.numpy() == label)[0]
            for label in self.labels_set
        }
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.sampling_weight = sampling_weight
        self.dataset = dataset
        self.batch_size = batch_size

        self.n_samples = {}
        count = 0
        for label, weight in zip(self.labels_set, self.sampling_weight):
            if label != 3:
                self.n_samples[label] = int(self.batch_size * weight)
                count += int(self.batch_size * weight)
            else:
                # label=3のデータ数でバッチサイズピッタリになるように帳尻合わせ
                self.n_samples[label] = self.batch_size - count

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            # classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in self.labels_set:
                indices.extend(
                    self.label_to_indices[class_][
                        self.used_label_indices_count[
                            class_
                        ] : self.used_label_indices_count[class_]
                        + self.n_samples[class_]
                    ]
                )
                self.used_label_indices_count[class_] += self.n_samples[class_]
                if self.used_label_indices_count[class_] + self.n_samples[class_] > len(
                    self.label_to_indices[class_]
                ):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            indices = [int(i) for i in indices]
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return len(self.dataset) // self.batch_size


# model with Multi-Sample Dropout(num=2)
class Classifier(nn.Module):
    def __init__(self, model_name, num_classes=4):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.msd = nn.ModuleList([nn.Dropout(0.5) for _ in range(8)])
        self.linear = nn.Linear(768, num_classes)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self, input_ids, attention_mask, token_type_ids, loss_func=None, labels=None
    ):
        output, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        output = output[:, 0, :]

        msd_ouputs = torch.tensor
        for i, dropout in enumerate(self.msd):
            if i == 0:
                out = dropout(output)
                out = self.linear(out)
                if loss_func is not None:
                    loss = loss_func(out, labels)
            else:
                tmp_out = dropout(output)
                tmp_out = self.linear(tmp_out)
                out = out + tmp_out
                if loss_func is not None:
                    loss = loss + loss_func(tmp_out, labels)

        if loss_func is not None:
            return out / len(self.msd), loss / len(self.msd)

        return out / len(self.msd), None


# training function
def train_fn(dataloader, model, criterion, optimizer, scheduler, device, epoch):

    model.train()
    total_loss = 0
    total_corrects = 0
    all_labels = []
    all_preds = []

    progress = tqdm(dataloader, total=len(dataloader))

    for i, batch in enumerate(progress):
        progress.set_description(f"<Train> Epoch{epoch+1}")

        attention_mask, input_ids, labels, token_type_ids = batch.values()
        del batch

        optimizer.zero_grad()

        outputs, loss = model(
            input_ids,
            attention_mask,
            token_type_ids,
            loss_func=criterion,
            labels=labels,
        )
        del input_ids, attention_mask, token_type_ids
        # loss = criterion(outputs, labels)  # 損失を計算
        _, preds = torch.max(outputs, 1)  # ラベルを予測
        del outputs

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        del loss
        total_corrects += torch.sum(preds == labels)

        all_labels += labels.tolist()
        all_preds += preds.tolist()
        del labels, preds

        progress.set_postfix(
            loss=total_loss / (i + 1),
            f1=f1_score(all_labels, all_preds, average="macro"),
        )

    train_loss = total_loss / len(dataloader)
    train_acc = total_corrects.double().cpu().detach().numpy() / len(dataloader.dataset)
    train_f1 = f1_score(all_labels, all_preds, average="macro")

    return train_loss, train_acc, train_f1


def eval_fn(dataloader, model, criterion, device, epoch):
    model.eval()
    total_loss = 0
    total_corrects = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        progress = tqdm(dataloader, total=len(dataloader))

        for i, batch in enumerate(progress):
            progress.set_description(f"<Valid> Epoch{epoch+1}")

            attention_mask, input_ids, labels, token_type_ids = batch.values()
            del batch

            outputs, loss = model(
                input_ids,
                attention_mask,
                token_type_ids,
                loss_func=criterion,
                labels=labels,
            )
            del input_ids, attention_mask, token_type_ids
            # loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            del outputs

            total_loss += loss.item()
            del loss
            total_corrects += torch.sum(preds == labels)

            all_labels += labels.tolist()
            all_preds += preds.tolist()
            del labels, preds

            progress.set_postfix(
                loss=total_loss / (i + 1),
                f1=f1_score(all_labels, all_preds, average="macro"),
            )

    valid_loss = total_loss / len(dataloader)
    valid_acc = total_corrects.double().cpu().detach().numpy() / len(dataloader.dataset)

    valid_f1 = f1_score(all_labels, all_preds, average="macro")

    return valid_loss, valid_acc, valid_f1


def plot_training(
    train_losses,
    train_accs,
    train_f1s,
    valid_losses,
    valid_accs,
    valid_f1s,
    epoch,
    fold,
):

    loss_df = pd.DataFrame(
        {"Train": train_losses, "Valid": valid_losses}, index=range(1, epoch + 2)
    )
    loss_ax = sns.lineplot(data=loss_df).get_figure()
    loss_ax.savefig(f"./figures/loss_plot_fold={fold}.png", dpi=300)
    loss_ax.clf()

    acc_df = pd.DataFrame(
        {"Train": train_accs, "Valid": valid_accs}, index=range(1, epoch + 2)
    )
    acc_ax = sns.lineplot(data=acc_df).get_figure()
    acc_ax.savefig(f"./figures/acc_plot_fold={fold}.png", dpi=300)
    acc_ax.clf()

    f1_df = pd.DataFrame(
        {"Train": train_f1s, "Valid": valid_f1s}, index=range(1, epoch + 2)
    )
    f1_ax = sns.lineplot(data=f1_df).get_figure()
    f1_ax.savefig(f"./figures/f1_plot_fold={fold}.png", dpi=300)
    f1_ax.clf()


def trainer(fold, df):

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    # ----- Valid Under Sampling ----- #
    # label = 2を100件まで減らす
    us_valid = pd.DataFrame()
    for i in range(4):
        if i != 2:
            us_valid = pd.concat([us_valid, valid_df[valid_df["labels"] == i]])
        else:
            random_indices = np.random.choice(
                valid_df[valid_df["labels"] == i].index, 100
            )
            us_valid = pd.concat(
                [us_valid, valid_df[valid_df["labels"] == i].loc[random_indices, :]]
            )
    valid_df = us_valid

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = make_dataset(train_df, tokenizer, DEVICE)
    valid_dataset = make_dataset(valid_df, tokenizer, DEVICE)

    train_weighted_bs = WeightedBatchSampler(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        n_classes=NUM_CLASSES,
        sampling_weight=[0.23, 0.18, 0.20, 0.39],
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_weighted_bs
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False
    )

    model = Classifier(MODEL_NAME, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    # BERTの重みを固定
    model_params = list(model.named_parameters())
    bert_params = [p for n, p in model_params if "bert" in n]
    other_params = [p for n, p in model_params if not "bert" in n]

    train_losses = []
    train_accs = []
    train_f1s = []
    valid_losses = []
    valid_accs = []
    valid_f1s = []

    best_loss = np.inf
    best_acc = 0
    best_f1 = 0

    for epoch in range(EPOCHS):
        if epoch == 0:
            #  最初のepochはBERT部分の重みを固定
            optimizer = AdamW(other_params, lr=2e-5)
        else:
            optimizer = AdamW(model.parameters(), lr=2e-5)

        # TODO: weightを調整
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 10, 1, 1], dtype=torch.float).to(DEVICE))
        # ダミーのスケジューラー
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=1.0)

        train_loss, train_acc, train_f1 = train_fn(
            train_dataloader, model, criterion, optimizer, scheduler, DEVICE, epoch
        )
        valid_loss, valid_acc, valid_f1 = eval_fn(
            valid_dataloader, model, criterion, DEVICE, epoch
        )
        print(f"Loss: {valid_loss}  Acc: {valid_acc}  f1: {valid_f1}  ", end="")

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        valid_f1s.append(valid_f1)

        plot_training(
            train_losses,
            train_accs,
            train_f1s,
            valid_losses,
            valid_accs,
            valid_f1s,
            epoch,
            fold,
        )

        best_loss = valid_loss if valid_loss < best_loss else best_loss
        besl_acc = valid_acc if valid_acc > best_acc else best_acc
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            print("model saving!", end="")
            torch.save(model.state_dict(), MODELS_DIR + f"best_{MODEL_NAME}_{fold}.pth")
        print("\n")

    return best_f1


# training
df = make_folded_df(TRAIN_FILE, NUM_SPLITS)
f1_scores = []
for fold in range(NUM_SPLITS):
    print(f"fold {fold}", "=" * 80)
    f1 = trainer(fold, df)
    f1_scores.append(f1)
    print(f"<fold={fold}> best score: {f1}\n")

cv = sum(f1_scores) / len(f1_scores)
print(f"CV: {cv}")

lines = ""
for i, f1 in enumerate(f1_scores):
    line = f"fold={i}: {f1}\n"
    lines += line
lines += f"CV    : {cv}"
with open(f"./result/{MODEL_NAME}_result.txt", mode="w") as f:
    f.write(lines)


# inference
models = []
for fold in range(NUM_SPLITS):
    model = Classifier(MODEL_NAME)
    model.load_state_dict(torch.load(MODELS_DIR + f"best_{MODEL_NAME}_{fold}.pth"))
    model.to(DEVICE)
    model.eval()
    models.append(model)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
test_df = pd.read_csv(TEST_FILE)
test_df["labels"] = -1
test_dataset = make_dataset(test_df, tokenizer, DEVICE)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False
)

with torch.no_grad():
    progress = tqdm(test_dataloader, total=len(test_dataloader))
    final_output = []

    for batch in progress:
        progress.set_description("<Test>")

        attention_mask, input_ids, labels, token_type_ids = batch.values()

        outputs = []
        for model in models:
            output, _ = model(input_ids, attention_mask, token_type_ids)
            outputs.append(output)

        outputs = sum(outputs) / len(outputs)
        outputs = torch.softmax(outputs, dim=1).cpu().detach().tolist()
        # outputs = np.argmax(outputs, axis=1)

        final_output.extend(outputs)

final_output = hack(np.array(final_output))

submit = pd.read_csv(
    os.path.join(data_dir, "submit_sample.csv"), names=["id", "labels"]
)
submit["labels"] = final_output
submit["labels"] = submit["labels"] + 1
check_submit_distribution(submit)
try:
    submit.to_csv(
        "./output/submission_cv{}.csv".format(str(cv).replace(".", "")[:10]),
        index=False,
        header=False,
    )
except NameError:
    submit.to_csv("./output/submission.csv", index=False, header=False)
