import glob
import os
import sys

import pandas as pd
import numpy as np
import pulp

root = os.path.join(os.environ["HOME"], "Workspace/learning/signate/SIGNATE_Student_Cup_2020")

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


def main():
    # Loading data
    ens1 = os.path.join(root, "src", "14_BERT_MSD_ENS")
    ens2 = os.path.join(root, "src", "15_BERT_L_MSD_ENS")
    ens3 = os.path.join(root, "src", "16_XLNet_ENS")
    ens4 = os.path.join(root, "src", "17_ALBERT_ENS")
    
    dfs = []
    for ens in [ens1, ens2, ens3, ens4]:
        ens = glob.glob(os.path.join(ens, 'output', '*'))[0]
        df = pd.read_csv(ens)
        df.columns = ["id", "labels", 1, 2, 3, 4]
        dfs.append(df)

    # Ensamble
    for i, df in enumerate(dfs):
        probs = df.iloc[:, 2:6]
        if i == 0:
            ensambled = probs
        else:
            ensambled = ensambled + probs

    ensambled = ensambled / len(dfs)

    # use hack
    tmp = hack(ensambled.to_numpy())
    output_hack = pd.DataFrame({
        'id': dfs[0]["id"],
        'labels': tmp + 1
    })
    print(output_hack)

    # dont use
    tmp = ensambled.idxmax(axis=1)
    output = pd.DataFrame({
        'id':dfs[0]["id"],
        'labels': tmp
    })
    print(output)

    output_hack.to_csv(os.path.join(root, "src/ensamble_output", "output_hack.csv"), header=False, index=False)
    output.to_csv(os.path.join(root, "src/ensamble_output", "output.csv"), header=False, index=False)


if __name__ == "__main__":
    main()
