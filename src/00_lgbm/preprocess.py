import os
import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from tqdm import tqdm

root = os.path.join(
    os.environ["HOME"], "Workspace/learning/signate/SIGNATE_Student_Cup_2020"
)


def main():
    """
    processed_data/feature_words.txtを読み込んで、その単語の有無でOne-hotなベクトルを構成
    """

    # load feature words
    with open(os.path.join(root, "processed_data", "feature_words.txt"), mode="r") as f:
        feature_words = f.readlines()

    feature_words = [w.replace("\n", "") for w in feature_words]

    train = pd.read_csv(os.path.join(root, "data", "train.csv"))
    test = pd.read_csv(os.path.join(root, "data", "test.csv"))

    p_train = preprocess(train, feature_words)
    p_train["id"] = train["id"]
    p_train["jobflag"] = train["jobflag"] - 1
    p_train.to_csv(os.path.join(root, "processed_data", "00_train.csv"), index=False)

    p_test = preprocess(test, feature_words)
    p_test["id"] = test["id"]
    p_test.to_csv(os.path.join(root, "processed_data", "00_test.csv"), index=False)


def text_preprocess(text):
    text = re.sub("<br />", "", text)

    for p in string.punctuation:
        if (p == ".") or (p == ","):
            continue
        else:
            text = text.replace(p, " ")

    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    text = text.lower()
    text = text.split(" ")
    stop_words = stopwords.words("english") + ["", ",", "."]
    text = [w for w in text if w not in stop_words]

    return text


def preprocess(df, feature_words):
    """
    元データをfeature_wordsをそれぞれカラムにもつOne-hotベクトルに変換し
    データフレームを返す

    Args:
        df (pd.DataFrame): 元データのデータフレーム
        feature_words (list): feature_words
    """

    res = {}
    for fw in tqdm(feature_words):
        l = []
        for d in df["description"]:
            d = text_preprocess(d)
            if fw in d:
                l.append(1)
            else:
                l.append(0)
        res[fw] = l

    return pd.DataFrame(res)


if __name__ == "__main__":
    main()
