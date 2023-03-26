from numbers_parser import Document
import pandas as pd

from fuji import fuji


def ground_truth_from_numbers(filename,
                              manual_fix=False, ordered=False):
    doc = Document(filename)
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

    if not ordered:
        df = df.sort_values(by=['change'])

    # add 98 and 99 to the end of the list with score 0.06 for FRI_singles_generation
    # and invert scores (1-score)
    if manual_fix:
        # use concat instead of append to avoid warning
        if not ordered:
            df = pd.concat([df, pd.DataFrame({'change': [98, 99], 'rig': [0.06, 0.06]})])
        else:
            df = pd.concat([pd.DataFrame({'change': [98, 99], 'rig': [0.06, 0.06]}), df])

        df['rig'] = df['rig'].astype(float)
        df['rig'] = 1 - df['rig']

    return df


def get_ground_truths():
    return (ground_truth_from_numbers("data/FRI_singles_generations.numbers", manual_fix=True),
            ground_truth_from_numbers("data/FRI_automl_first_generation.numbers"))


def get_ground_truths_ordered():
    return (ground_truth_from_numbers("data/FRI_singles_generations.numbers", manual_fix=True, ordered=True),
            ground_truth_from_numbers("data/FRI_automl_first_generation.numbers", ordered=True))

def jaccard(a, b):
    return fuji.compute_similarity(a, b, "jaccard")


def fuzzy_jaccard(a, b):
    return fuji.compute_similarity(a, b, "fuzzy_jaccard")


if __name__ == "__main__":
    #print(get_ground_truths())
    print(get_ground_truths_ordered())

    gd_singles, gd_first_gen = get_ground_truths_ordered()

    print(gd_singles)


