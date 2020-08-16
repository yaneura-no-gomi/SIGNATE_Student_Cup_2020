import pandas as pd

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


if __name__ == "__main__":
    check_submit_distribution(pd.read_csv("../../src/03_BERT_MSD/output/submission_cv0626074897.csv"))
    print()
    print(pd.read_csv("../../src/04_BERT_MSD/output/submission_cv0623254261.csv").iloc[:, :2])
    check_submit_distribution(pd.read_csv("../../src/04_BERT_MSD/output/submission_cv0623254261.csv").iloc[:, [0, 2]])
    check_submit_distribution(pd.read_csv("../../src/04_BERT_MSD/output/submission_cv0623254261.csv").iloc[:, [0, 1]])
