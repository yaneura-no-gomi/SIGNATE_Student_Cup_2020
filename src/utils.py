import pandas as pd

def check_submit_distribution(df: pd.DataFrame):
    df.columns = ["id", "pred"]
    print(df.groupby(by="pred").count())
    print(df.groupby(by="pred").count() / len(df))
    print('推定分布: 404, 320, 345, 674')


if __name__ == "__main__":
    check_submit_distribution(pd.read_csv("./04_BERT_MSD/output/submission_cv0550392698.csv"))