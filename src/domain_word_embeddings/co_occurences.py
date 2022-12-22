from collections import Counter
import torch as tr
import pandas as pd
from bisect import bisect_left



# ### CO-OCURRENCES ### #
def is_in(x, a):
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return True
    return False


def index(x, a):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError


def it_context(input_, window_size):
    for doc in input_:
        for i in range(len(doc)):
            begin_slice = max(0, i - window_size)
            end_slice = min(len(doc), i + window_size + 1)

            for j in range(begin_slice, end_slice):
                if i != j:
                    yield doc[i], doc[j]


def get_cooccurrences(input_, window_size, vocab):

    Y = tr.zeros((len(vocab), len(vocab)))

    for t1, t2 in it_context(input_, window_size):
        try:
            Y[index(t1, vocab), index(t2, vocab)] += 1
        #in case word doesn't occur in vocab
        except ValueError:
            pass

    #counts all words
    cnt_corpus = sum(map(len, input_))

    #stores all words
    count_words = Counter([l for d in input_ for l in d])

    #sanity check, if word is in vocab, sorts as vocab
    word_counts = tr.FloatTensor([count_words[l] for l in vocab])

    i = tr.nonzero(Y).t()
    v = Y[i[0], i[1]]

    #PMI
    v = tr.log(
        tr.div(
            v * cnt_corpus,
            1 + tr.mul(
                word_counts[i[0]],
                word_counts[i[1]]
            )
        )
    )

    return i, v, Y.size()


def iterate_texts(df):
    for _, row in df.iterrows():
        result = []
        if 'title' in df and row.title is not None:
            result.append(row.title)
        if 'headline' in df and row.headline is not None:
            result.append(row.headline)
        if 'lead' in df and row.lead is not None:
            result.append(row.lead)
        if 'teaser_title' in df and row.teaser_title != row.title and row.teaser_title is not None:
            result.append(row.teaser_title)
        if 'teaser_text' in df and row.teaser_text is not None:
            result.append(row.teaser_text)
        if 'abstract' in df and row.abstract is not None:
            result.append(row.abstract)
        if 'paragraphs' in df and row.paragraphs is not None:
            if len(row.paragraphs) > 0:
                result.append(". ".join(row.paragraphs))
        if 'text' in df and row.text is not None:
            result.append(row.text)

        if len(result) > 0:
            yield ". ".join(result)