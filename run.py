from domain_word_embeddings import data
from domain_word_embeddings.trainer import Trainer
from domain_word_embeddings.model import W2VPred, W2VConstr
from domain_word_embeddings.evaluate import load_questions, tree_distance_matrix_to_w, get_wikipedia_fos_gt
from domain_word_embeddings.utils import create_results_dir
from domain_word_embeddings.configuration import results_path
import numpy as np

def main():
    '''
    this is the main script to execute data loading, experiment, visualization
    data loading: see /src/data.py
    experiment: initialize model from src/model.py
    training: src/trainer.py
    calculate neighborhood and analogy accuracy only after the experiment and print it out
    visualization: not yet
    there should be variable parameters such as lambda, tau, d
    '''

    try:
        Y, vocab, grouping_intervals = data.get_cached_data()
    except FileNotFoundError:
        print('data has not been found and will be loaded from scratch, this can take up to several hours')
        Y, vocab, grouping_intervals = data.get_data()

    out_dir = create_results_dir(results_path)
    print(f'results will be stored in {out_dir}')

    labels = ['Natural_sciences', 'Chemistry', 'Computer_science', 'Biology',
              'Engineering&Technology', 'Civil_engineering', 'Electrical_engineering&Electronic_engineering',
              'Mechanical_engineering', 'Social_sciences', 'Business&Economics', 'Law', 'Psychology',
              'Humanities', 'Literature&Languages', 'History&Archaeology', 'Religion&Philosophy&Ethics']

    data_partition = "test"

    grouping_index = [i for i in range(len(grouping_intervals)) if data_partition in grouping_intervals[i][0]]
    grouping_intervals = [grouping_intervals[i][1] for i in range(len(grouping_intervals)) if i in grouping_index]
    true_index = [grouping_intervals.index(label) for label in labels]

    Y = [Y[i] for i in np.array(grouping_index)[true_index]]
    slices = grouping_intervals

    idx_analogy = load_questions(vocab)
    # idx_analogy = []
    train = Trainer(Y=Y)

    d = 50
    model = W2VPred(tau=1024, lam=512, V=len(vocab), T=len(slices), d=d)
    # w = tree_distance_matrix_to_w(get_wikipedia_fos_gt())
    # model = W2VConstr(tau=256, lam=512, V=len(vocab), T=len(slices), d=d, w=w)
    train.fit(model, idx_analogy, vocab, out_dir, n_steps=1000)

if __name__ == "__main__":
    main()
