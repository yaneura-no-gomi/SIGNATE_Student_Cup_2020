import os
import random
import re
import string
from random import shuffle

import nltk
import numpy as np
import pandas as pd
import pulp
from googletrans import Translator
from nltk.corpus import stopwords, wordnet
from tqdm import tqdm

nltk.download('wordnet')



# 制約付き対数尤度最大化問題を解く
def hack(prob):
    N_CLASSES = [404, 320, 345, 674]  # @yCarbonによる推定（過去フォーラム参照）
    logp = np.log(prob + 1e-16)
    N = prob.shape[0]
    K = prob.shape[1]

    m = pulp.LpProblem('Problem', pulp.LpMaximize)  # 最大化問題

    # 最適化する変数(= 提出ラベル)
    x = pulp.LpVariable.dicts('x', [(i, j) for i in range(N) for j in range(K)], 0, 1, pulp.LpBinary)
    
    # log likelihood(目的関数)
    log_likelihood = pulp.lpSum([x[(i, j)] * logp[i, j] for i in range(N) for j in range(K)])
    m += log_likelihood
    
    # 各データについて，1クラスだけを予測ラベルとする制約
    for i in range(N):
        m += pulp.lpSum([x[(i, k)] for k in range(K)]) == 1  # i.e., SOS1
    
    # 各クラスについて，推定個数の合計に関する制約
    for k in range(K):
        m += pulp.lpSum([x[(i, k)] for i in range(N)]) == N_CLASSES[k]
        
    m.solve()  # 解く

    assert m.status == 1  # assert 最適 <=>（実行可能解が見つからないとエラー）

    x_ast = np.array([[int(x[(i, j)].value()) for j in range(K)] for i in range(N)])  # 結果の取得
    return x_ast.argmax(axis=1) # 結果をonehotから -> {0, 1, 2, 3}のラベルに変換

def check_submit_distribution(df: pd.DataFrame):
    df.columns = ["id", "pred"]
    print(df.groupby(by="pred").count())
    print(df.groupby(by="pred").count() / len(df))
    
    true_dis = pd.DataFrame({
        'num': [404, 320, 345, 674],
        'ratio' : [404 / len(df), 320 / len(df), 345 / len(df), 674 / len(df)]
    })
    
    print("\n推定分布")
    print(true_dis)


# Augmentation by synonym
def class_augmentation_synonym(df, target_job, replace_num, aug_num):
    df = df[df["labels"] == target_job]
    augmented_df = df.loc[:, ["description", "labels", "kfold"]].copy()

    while len(augmented_df) < aug_num:
        for d, j, k in zip(df["description"], df["labels"], df["kfold"]):
            if len(augmented_df) < aug_num:
                replaced_dict = synonym_replace(d, n=replace_num)
                new_d = replace_word(d, replaced_dict)
                tmp = pd.DataFrame({
                    "description": [new_d],
                    "labels": [j],
                    "kfold": [k]
                })
                augmented_df = pd.concat([augmented_df, tmp])
            else:
                break
    
    return augmented_df

def synonym_replace(line, n):
    
    random.seed(42)
    clean_line = ""

    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
            'ours', 'ourselves', 'you', 'your', 'yours', 
            'yourself', 'yourselves', 'he', 'him', 'his', 
            'himself', 'she', 'her', 'hers', 'herself', 
            'it', 'its', 'itself', 'they', 'them', 'their', 
            'theirs', 'themselves', 'what', 'which', 'who', 
            'whom', 'this', 'that', 'these', 'those', 'am', 
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 
            'have', 'has', 'had', 'having', 'do', 'does', 'did',
            'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
            'because', 'as', 'until', 'while', 'of', 'at', 
            'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 
            'above', 'below', 'to', 'from', 'up', 'down', 'in',
            'out', 'on', 'off', 'over', 'under', 'again', 
            'further', 'then', 'once', 'here', 'there', 'when', 
            'where', 'why', 'how', 'all', 'any', 'both', 'each', 
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
            'very', 's', 't', 'can', 'will', 'just', 'don', 
            'should', 'now', '']

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]

    words = clean_line.split(' ')
    
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    replaced_dict = {}
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            replaced_dict[random_word] = synonym
            num_replaced += 1
        if num_replaced >= n: #only replace up to n words
            break
    return replaced_dict


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def replace_word(text, replaced_dict):
    text = text.lower()
    words = text.split(' ')
    new_words = words.copy()
    
    for target_w, replace_w in replaced_dict.items():
        for i, w in enumerate(words):
            if target_w == w:
                new_words[i] = replace_w
    new_text = ''
    for nw in new_words:
        new_text += nw + ' '
    return new_text[:-1]


# Augmentation by retransration
def class_augmentation_retranslation(df, target_job, aug_num):
    """Augmentation by retransration

    Args:
        df (pd.DataFrame): train data
        target_job (int): labels [0, 3]
        aug_num (int): the number to increase

    Returns:
        [pd.DataFrame]: Augmented DataFrame
    """

    df = df[df["labels"] == target_job]

    # 'es', 'fr', 'de', 'ja'で対応できる数のaug_num
    assert aug_num < len(df) * 4

    random.seed(42)
    translator = Translator()
    languages = ['es', 'fr', 'de', 'ja']

    augmented_df = df.loc[:, ["description", "labels", "kfold"]].copy()
    trans_target = list(df.index)
    random.shuffle(trans_target)
    idx = 0
    for i in tqdm(range(aug_num+1)):
        if idx < len(df):
            # 翻訳
            translated = translator.translate(df.loc[trans_target[idx], "description"], dest=languages[i // len(df)]).text
            re_translated = translator.translate(translated, dest="en").text
            tmp = pd.DataFrame({
                "description": [re_translated],
                "labels": [df.loc[trans_target[idx], "labels"]],
                "kfold": [df.loc[trans_target[idx], "kfold"]]
            })
            augmented_df = pd.concat([augmented_df, tmp])
            idx += 1

        else:
            random.shuffle(trans_target)
            idx = 0

    return augmented_df

if __name__ == "__main__":
    check_submit_distribution(pd.read_csv("../../src/08_BERT_MSD/output/submission_cv0614341115.csv"))
    check_submit_distribution(pd.read_csv("../../src/09_BERT_MSD/output/1_10_1_1_submission_cv0587540339.csv"))
    check_submit_distribution(pd.read_csv("../../src/10_BERT_MSD/output/submission_cv0601808290.csv"))
