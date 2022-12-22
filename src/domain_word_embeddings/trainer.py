from tqdm import tqdm
import numpy as np
import torch
from torch.optim import Adam
from .evaluate import compute_nacc, analogy_accuracy
from .utils import save_var

def lr_scheduler(step):
    if step < 100:
        return 0.1
    elif step < 500:
        return 0.05
    else:
        return 0.01

class Trainer:
    def __init__(self, Y):
        self.lr = None
        self.Y = Y
        self.T = len(Y)
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.it_T = lambda: np.random.permutation(range(self.T))

        print(self.device)

        for t in range(self.T):
            self.Y[t] = self.Y[t].to(self.device)

    def step(self, model):

        for t in self.it_T():
            optimizer = self.optimizers[t]

            optimizer.zero_grad(set_to_none=True)
            optimizer.zero_grad()

            _ = model.forward(self.Y, t)

            model.embedding.backward()

            embedding_current_grad = model.U[t].grad.clone().to(self.device)
            model.embedding_grad[t].append(torch.norm(model.U[t].grad.clone().cpu()).numpy())

            if hasattr(model, "neighborhood"):
                model.neighborhood.backward(retain_graph=True)

            if hasattr(model, "regularizer"):
                model.regularizer.backward()
                model.regularizer_grad[t].append(torch.norm(model.U[t].grad.clone().cpu() - embedding_current_grad.cpu()).numpy())

            model.U[t].grad[model.U[t].grad != model.U[t].grad] = 0

            optimizer.step()

    def fit(self, model, idx_analogy, vocab, out_dir, n_steps=100):

        neighboring_acc = []
        acc4 = []

        for step in tqdm(range(n_steps)):

            if self.lr is None or self.lr != lr_scheduler(step):
                self.lr = lr_scheduler(step)
                self.optimizers = [
                    Adam([model.U[t]], lr=lr_scheduler(step)) for t in range(self.T)
                ]
            
            self.step(model)

            if (step > 0 and (step % 50 == 0)) or step==n_steps-1:
                print(f"learning after {step} rounds is {self.lr}")

                neighboring_acc.append(compute_nacc(torch.stack(model.U).detach().cpu()))
                print(
                    "neighborhood accuracy after {} rounds: {}".format(
                        step, neighboring_acc[-1]
                    )
                )

                if step % 200 == 0 or step == n_steps - 1:
                    acc4.append(analogy_accuracy(
                        model.U,
                        idx_analogy, vocab, [1, 5, 10],
                        avg = True
                    ))

        accs = {
            'neighboring_acc': neighboring_acc,
            'acc4': acc4
        }

        save_var(accs, "accuracies", out_dir)
        save_var(torch.stack(model.U).detach().cpu(), "predicted_U", out_dir)
        print(f'model finished training, results can be found in {out_dir}')