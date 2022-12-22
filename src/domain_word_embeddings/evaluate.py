from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import networkx as nx
from . import configuration  as config
import matplotlib.pyplot as plt


def get_wikipedia_fos_gt():
    nnodes = 4
    labels_fos = ['Natural_sciences', 'Chemistry', 'Computer_science', 'Biology',
                  'Engineering&Technology', 'Civil_engineering', 'Electrical_engineering&Electronic_engineering',
                  'Mechanical_engineering', 'Social_sciences', 'Business&Economics', 'Law', 'Psychology',
                  'Humanities', 'Literature&Languages', 'History&Archaeology', 'Religion&Philosophy&Ethics']

    wfos_tree = get_tree(labels_fos, nnodes, './')
    wfos_tree.remove_node('root')
    true_adjacency_fos = nx.algorithms.shortest_paths.dense.floyd_warshall_numpy(wfos_tree)
    true_adjacency_fos = np.array(true_adjacency_fos)

    true_adjacency_fos[true_adjacency_fos == np.inf] = true_adjacency_fos[true_adjacency_fos != np.inf].max()+1

    G = nx.from_numpy_matrix(np.array(true_adjacency_fos))
    nx.draw(G, with_labels=True)
    plt.savefig('./plots/out/tree_fos')

    return true_adjacency_fos


def tree_distance_matrix_to_w(D):
    w = D.copy()
    w /= w.max()
    w = 1 - w
    w -= np.eye(len(w))
    w /= w.max()
    return w


def neighborhood_accuracy(w, inverted=True):
    '''
    see neighborhood_index in dynamic_word_embeddings/dynamic_word_embeddings/evaluation/evaluate.py
    '''
    try:
        w = w.numpy()
    except:
        pass

    ni = 0

    for irow, row in enumerate(w):
        closest = row.argsort()

        len_closest = 3
        if inverted:
            closest = closest[::-1]
            len_closest = 2

        closest = closest[:len_closest]

        ni += int((irow - 1) in closest)
        ni += int((irow + 1) in closest)

    return ni / (2 * (w.shape[0] - 1))


def analogy_accuracy(embeddings, idx_analogy, vocab, neighbors, avg = False):
    '''
    see analogy4 in dynamic_word_embeddings/dynamic_word_embeddings/evaluation/analogies_WEN_glove.py
    '''

    U_normed = []
    for iemb in range(len(embeddings)):
        U_normed.append(embeddings[iemb] / embeddings[iemb].norm(p=2, dim=1)[:, None])
    U_normed = torch.stack(U_normed).detach().cpu().numpy()

    if avg:
        U_normed = np.mean(U_normed, 0)[None, :, :]

    df = pd.read_csv("/home/space/datasets/text/nyt/raw/tests/questions-words.txt", header=None, sep=' |\n',
                     engine='python', names=["1", "2", "3", "4"])
    df = df[~df['1'].str.startswith(':')]
    # if config.DEBUG:
    #     df = pd.DataFrame(df.iloc[:200])

    idx = [[0] * 6 for i in range(len(idx_analogy))]
    acc = np.zeros([U_normed.shape[0], len(neighbors)])
    try:
        vocab = vocab.tolist()
    except:
        pass

    vocab = [word.lower() for word in vocab]

    stats = []
    stats_wrong = []
    try:
        U_normed = U_normed.numpy()
    except AttributeError:
        pass

    for sl in range(U_normed.shape[0]):
        for i_n, nn in enumerate(neighbors):

            for j in (range(len(idx_analogy))):
                # get each index of analogy
                idx0 = vocab.index(df.values[idx_analogy[j], 0].lower())
                idx1 = vocab.index(df.values[idx_analogy[j], 1].lower())
                idx2 = vocab.index(df.values[idx_analogy[j], 2].lower())
                idx3 = vocab.index(df.values[idx_analogy[j], 3].lower())

                idx[j][:4] = (df.values[idx_analogy[j], :4])

                query = U_normed[sl, :, :].dot(
                    U_normed[sl, idx1, :]
                    - U_normed[sl, idx0, :]
                    + U_normed[sl, idx2, :]
                )

                found_words = np.array(vocab)[np.argsort(query)[::-1][:nn]]

                stats.append({
                    "q1": df.values[idx_analogy[j], 0].lower(),
                    "q2": df.values[idx_analogy[j], 1].lower(),
                    "q3": df.values[idx_analogy[j], 2].lower(),
                    "q4": df.values[idx_analogy[j], 3].lower(),
                    "found_words": found_words
                })

                if idx[j][3].lower() in found_words:

                    idx[j][5] = 1
                    acc[sl, i_n] += 1

        print(100 * acc[sl] / len(idx_analogy))


def load_questions(vocab):
    # load question words
    df = pd.read_csv("/home/space/datasets/text/nyt/raw/tests/questions-words.txt", header=None, sep=' |\n',
                     engine='python', names=["1", "2", "3", "4"])
    df = df[~df['1'].str.startswith(':')]
    if config.DEBUG:
        df = pd.DataFrame(df.iloc[:200])
    print("load question words")
    idx_analogy = []

    vocab = [word.lower() for word in vocab]

    for i in range(len(df)):
        if (np.any(np.array(vocab) == df.values[i, 0].lower())
                and np.any(np.array(vocab) == df.values[i, 1].lower())
                and np.any(np.array(vocab) == df.values[i, 1].lower())
                and np.any(np.array(vocab) == df.values[i, 2].lower())
                and np.any(np.array(vocab) == df.values[i, 3].lower())):
            idx_analogy.append(i)

    return idx_analogy


def compute_nacc(U):

    U_normed = np.zeros(U.shape)
    for slice_ in range(U.shape[0]):
        U_normed[slice_] = U[slice_] / np.linalg.norm(U[slice_], ord='fro')

    T = U_normed.shape[0]

    neighbor_distances = torch.zeros(T, T)
    try:
        U_normed = torch.from_numpy(U_normed)
    except TypeError:
        pass

    for t in range(0, T):
        neighbor_distances[t] = torch.sum(
            torch.sum(torch.abs(U_normed[t] - U_normed) ** 2, 1), 1
        )

    true_adjacency = get_wikipedia_fos_gt()
    _, nacc = get_scores(true_adjacency, neighbor_distances, K=3, k=3)
    nacc = np.around(100 * nacc, decimals=2)

    return nacc


def get_wikipedia_fos_gt():
    nnodes = 4
    labels_fos = ['Natural_sciences', 'Chemistry', 'Computer_science', 'Biology',
                  'Engineering&Technology', 'Civil_engineering', 'Electrical_engineering&Electronic_engineering',
                  'Mechanical_engineering', 'Social_sciences', 'Business&Economics', 'Law', 'Psychology',
                  'Humanities', 'Literature&Languages', 'History&Archaeology', 'Religion&Philosophy&Ethics']

    wfos_tree = get_tree(labels_fos, nnodes, './')
    wfos_tree.remove_node('root')
    true_adjacency_fos = nx.algorithms.shortest_paths.dense.floyd_warshall_numpy(wfos_tree)
    true_adjacency_fos = np.array(true_adjacency_fos)

    true_adjacency_fos[true_adjacency_fos == np.inf] = true_adjacency_fos[true_adjacency_fos != np.inf].max()+1

    # G = nx.from_numpy_matrix(np.array(true_adjacency_fos))
    # nx.draw(G, with_labels=True)

    return true_adjacency_fos

def get_scores(true, pred, K, k):
    # warnings.warn("Make sure to use the correct order of input arguments: true, pred, K, k")
    try:
        pred = torch.from_numpy(pred)
    except TypeError:
        pass

    precs, recs = [], []
    for row_true, row_pred in zip(true, pred):
        prec, rec = score_at_K_k(row_true, row_pred, K, k)
        precs.append(prec)
        recs.append(rec)

    prec = np.array(precs).mean()
    rec = np.array(recs).mean()
    return prec, rec


def get_tree(labels, nnodes, path):

    categories = {labels[nnodes*(i-1)]: [labels[nnodes*(i-1)+j] for j in range(1,nnodes)]
                  for i in range(len(labels)//3)
                  }

    tree = nx.Graph()

    add_subtree("root", categories, tree)
    return tree


def add_subtree(name, children, tree):
    if isinstance(children, list):
        for child_name in children:
            tree.add_edge(name, child_name)
    else:
        for child_name,child_subtree in children.items():
            tree.add_edge(name, child_name)
            add_subtree(child_name, child_subtree, tree)


def score_at_K_k(true, pred, K, k):
    relevant = true.argsort()[1:K + 1]

    true_ = np.zeros(true.shape, dtype=bool)
    true_[relevant] = 1

    chosen = pred.argsort()[1:k + 1]

    # if inverted:
    #     chosen = pred.argsort(descending=True)[:k]

    pred_ = np.zeros(pred.shape, dtype=bool)
    pred_[chosen] = 1

    true_positive = (true_ & pred_).sum()
    false_positive = (~true_ & pred_).sum()
    false_negative = (true_ & ~pred_).sum()
    true_negative = (~true_ & ~pred_).sum()

    precision_at_k = true_positive / (true_positive + false_positive)
    recall_at_k = true_positive / (true_positive + false_negative)
    return precision_at_k, recall_at_k