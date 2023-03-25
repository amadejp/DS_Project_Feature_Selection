from numbers_parser import Document
import pandas as pd

from fuji import fuji


def get_ground_truth():
    doc = Document("data/FRI_automl_first_generation.numbers")
    sheets = doc.sheets
    tables = sheets[0].tables
    data = tables[0].rows(values_only=True)
    df = pd.DataFrame(data[1:], columns=data[0])

    df = df[['change', 'rig']]  # only keep relevant columns
    df = df.reset_index(drop=True)

    # regex to extract feature name
    # e.g. "{'add':['feature1']}" -> '1'
    df['change'] = df['change'].str.extract(r"(\d+)", expand=False)

    df = df.dropna()
    df['change'] = df['change'].astype(int)

    df = df.sort_values(by=['change'])

    return df


def jaccard(a, b):
    return fuji.compute_similarity(a, b, "jaccard")


def fuzzy_jaccard(a, b):
    return fuji.compute_similarity(a, b, "fuzzy_jaccard")


if __name__ == "__main__":
    print(get_ground_truth())
