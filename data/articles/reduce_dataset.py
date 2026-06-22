import pandas as pd

def clean():
    df = pd.read_csv("data/articles/urdu-news-dataset-1M.csv", encoding="cp1256", on_bad_lines="skip")
    
    # Drop rows with Category other than "Sports"
    df = df[df["Category"] == "Sports"]
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df[df["Date"] < pd.Timestamp("2015-01-01")]
    df.to_csv("data/articles/urdu-news-dataset-sports.csv", index=False, encoding="cp1256")

if __name__ == "__main__":
    clean()