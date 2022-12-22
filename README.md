## Domain-Specific Word Embeddings with Structure Prediction  
This repository contains the code for `Word2Vec with Structure Prediction` and `Word2Vec with Structure Constraint` 
from the paper [Domain-Specific Word Embeddings with Structure Prediction](https://arxiv.org/pdf/2210.04962.pdf) (to appear in TACL).

### Citation Informaation  

```
@article{brandl2022domain,
  title={Domain-Specific Word Embeddings with Structure Prediction},
  author={Brandl, Stephanie and Lassner, David and Baillot, Anne and Nakajima, Shinichi},
  journal={arXiv preprint arXiv:2210.04962},
  year={2022}
}
```

### Installation Requirements
After cloning the repository, we recommend to set up a virtual environment where you can install the 
code via `pip install -e .`, then all required packages will be downloaded.

### Running the code
With `python run.py` you should be able to download the dataset `Wikipedia Field of Science` and compute domain-specific 
word embeddings, you can choose between:  
```
model = W2VPred(tau=1024, lam=512, V=len(vocab), T=len(slices), d=d)
``` 
and
```
w = tree_distance_matrix_to_w(get_wikipedia_fos_gt())  
model = W2VConstr(tau=256, lam=512, V=len(vocab), T=len(slices), d=d, w=w)
```
uncomment accordingly in `run.py` and set your paths in `src.domain_word_mbeddings.configuration.py`

### Contact us
In case of questions or concerns, you can reach us via brandl[at]di.ku.dk and lassner[at]tu-berlin.de.
