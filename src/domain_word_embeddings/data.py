import re
import spacy
import pandas as pd
from datasets import load_dataset
from .configuration import data_cache_path, DEBUG
from . import co_occurences as co
import torch
import numpy as np
from tqdm import tqdm

from collections import Counter

seed = 11586
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

rs = RandomState(MT19937(SeedSequence(seed)))


def get_cached_data():
    rd = pd.read_pickle
    p = data_cache_path + "/%s.pt"
    return rd(p % "Y"), rd(p % "vocab"), rd(p % "grouping_intervals")


def get_data():

    labels = [
        "Natural_sciences",
        "Chemistry",
        "Computer_science",
        "Biology",
        "Engineering&Technology",
        "Civil_engineering",
        "Electrical_engineering&Electronic_engineering",
        "Mechanical_engineering",
        "Social_sciences",
        "Business&Economics",
        "Law",
        "Psychology",
        "Humanities",
        "Literature&Languages",
        "History&Archaeology",
        "Religion&Philosophy&Ethics",
    ]

    data_set_groupby = "category"

    vocabulary_size = 20000

    df, nlp = load_raw_files()

    nlp.max_length = max(df.text.map(len)) + 1000

    grouper, vocab, df = parse_raw_text_df(
        df, vocabulary_size, nlp, data_set_groupby
    )

    df = split_dataset(df, labels)

    grouper = df.groupby(["split", "category"])
    window_size = 5
    
    i_s, v_s, sizes = [], [], []
    
    grouping_intervals = []

    for (group, subdf) in tqdm(grouper, total=len(grouper)):
        
        ci, cv, csize = co.get_cooccurrences(subdf.lemmatized, window_size, vocab)
        i_s.append(ci)
        v_s.append(cv)
        sizes.append(csize)
        grouping_intervals.append(group)


    Y = [torch.sparse.FloatTensor(i, v, s) for i, v, s in zip(i_s, v_s, sizes)]

    return Y, vocab, grouping_intervals


def split_dataset(df, labels):
    subdf = df.loc[df["category"].isin(labels)]
    idx = rs.permutation(np.arange(len(subdf)))

    result = subdf.iloc[idx].drop_duplicates("id", keep="first")
    train_test_mask = np.zeros(len(result))
    train_test_mask[: len(result) // 3] = 1
    rs.shuffle(train_test_mask)
    result["split"] = ["train" if it == 1 else "test" for it in train_test_mask]
    return result


def extract_titles(lines):

    tuples = []
    allSpans = list(re.finditer("<doc [^>]+>", lines, re.IGNORECASE))

    i = 0

    while i < len(allSpans):
        span = allSpans[i]
        startingDocIndex = allSpans[i].end() + 1

        if i < len(allSpans) - 1:
            startOfNextDocIndex = allSpans[i + 1].start()
        else:
            startOfNextDocIndex = len(lines)

        if startOfNextDocIndex == -1:
            startOfNextDocIndex = len(lines)

        doc = lines[startingDocIndex:startOfNextDocIndex]

        header = [lines[m.start() : m.end()].lower() for m in [span]][0]

        match = re.search('id="[0-9]+"', header)
        id = header[match.start() + 4 : match.end() - 1]

        match = re.search('title="[^>]+">', header)
        title = header[match.start() + 7 : match.end() - 2]
        tuples.append((doc, title, id))
        i += 1

    return tuples


def load_raw_files():
    
    data = load_dataset(
        "millawell/wikipedia_field_of_science",
        split="train"
    )

    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner", "textcat"])

    return data.to_pandas(), nlp


def parse_raw_text_df(df, vocabulary_size, nlp, data_set_groupby):

    spacified = []

    df = pd.DataFrame(df.sample(100)) if DEBUG else df
    for itext, text in enumerate(co.iterate_texts(df)):
        print(itext)
        spacified.append(nlp(text))

    df["lemmatized"] = [[t.lemma_ for t in doc] for doc in spacified]

    grouper = df.groupby(data_set_groupby)

    redlist_size = 100
    noccurences = 3

    print("load vocab")
    vocab = load_vocab(df, grouper, vocabulary_size, redlist_size, noccurences)

    return grouper, vocab, df


def load_vocab(df, grouper, vocabulary_size, redlist_size, noccurences):

    cnts = Counter(l for doc in df.lemmatized for l in doc)
    global_vocab, _ = zip(*cnts.most_common(vocabulary_size + redlist_size))
    global_redlist, _ = zip(*cnts.most_common(redlist_size))
    global_vocab = sorted(global_vocab)
    global_redlist = sorted(global_redlist)
    global_vocab = [el for el in global_vocab if not co.is_in(el, global_redlist)]

    # grouped vocab
    cnts_all = Counter()

    for ival, subdf in grouper:
        cnts_ival = Counter(l for doc in subdf.lemmatized for l in doc)
        vocab_ival, _ = zip(*cnts_ival.most_common(vocabulary_size))
        vocab_ival = sorted(vocab_ival)
        cnts_all.update(Counter(set([el for el in vocab_ival])))

    mostn, _ = zip(
        *Counter(
            el for el in cnts_all.elements() if cnts_all[el] >= noccurences
        ).items()
    )
    vocab = sorted([el for el in set(global_vocab) & set(mostn)])

    return vocab
