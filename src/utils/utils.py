import os
import random
import re
import string
from random import shuffle

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords, wordnet
from tqdm import tqdm

nltk.download('wordnet')


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


def class_augmentation(df, target_job, replace_num, aug_num):
    df = df[df["labels"] == target_job]
    augmented_df = df.loc[:, ["description", "labels", "kfold"]].copy()

    while len(augmented_df) < aug_num:
        for d, j, k in zip(df["description"], df["labels"], df["kfold"]):
            if len(augmented_df) < aug_num:
                replaced_dict = synonym_replace(d, n=replace_num)
                new_d = text_augmentation(d, replaced_dict)
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


def text_augmentation(text, replaced_dict):
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


if __name__ == "__main__":
    check_submit_distribution(pd.read_csv("../../src/07_BERT_MSD/output/submission_cv0619546447.csv"))
    check_submit_distribution(pd.read_csv("../../src/08_BERT_MSD/output/submission_cv0578829134.csv"))
