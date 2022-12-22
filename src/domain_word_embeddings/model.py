import torch

class W2VPred():

    def __init__(self, tau, lam, V, T, d):
        
        self.tau = tau
        self.lam = lam
        self.T = T
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.D = torch.zeros(self.T, self.T).to(self.device)
        self.U = [torch.rand(V, d, device=self.device, requires_grad=True) for _ in range(T)]

        self.embedding_grad = [[] for t in range(self.T)]
        self.regularizer_grad = [[] for t in range(self.T)]

    def forward(self, Y, t):

        self.embedding = torch.norm(
            Y[t].to_dense() - self.U[t] @ self.U[t].t()
        )**2

        v = torch.stack(self.U).view(self.T, -1)
        dists = torch.nn.functional.pdist(v) ** 2
        D = torch.zeros(self.T, self.T).to(self.device)
        indices = torch.triu_indices(self.T, self.T, offset=1)
        D[indices[0], indices[1]] = dists
        self.D = D + D.T

        indices = torch.triu_indices(*(self.D.shape), offset=1)
        w = torch.zeros(self.D.shape, device=self.D.device)
        w[indices[0], indices[1]] = 1 / self.D[indices[0], indices[1]]
        w = w+w.t()

        N = 1 / (torch.sum(w, 0)[:, None] + torch.sum(w, 1)[None, :])
        N * w

        self.regularizer = self.tau * (w[t] * self.D[t]).sum() + self.lam * (torch.norm(self.D))

        return self.embedding + self.regularizer


class W2VConstr():

    def __init__(self, tau, lam, V, T, d, w):
        self.tau = tau
        self.lam = lam
        self.T = T
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.w = torch.from_numpy(w).to(self.device)
        self.D = torch.zeros(self.T, self.T).to(self.device)
        torch.manual_seed(0)
        self.U = [torch.rand(V, d, device=self.device, requires_grad=True) for _ in range(T)]

        self.embedding_grad = [[] for _ in range(self.T)]
        self.regularizer_grad = [[] for _ in range(self.T)]

    def forward(self, Y, t):

        self.embedding = torch.norm(
            Y[t].to_dense() - self.U[t] @ self.U[t].t()
        )**2

        v = torch.stack(self.U).view(self.T, -1)
        dists = torch.nn.functional.pdist(v)
        D = torch.zeros(self.T, self.T).to(self.device)
        indices = torch.triu_indices(self.T, self.T, offset=1)
        D[indices[0], indices[1]] = dists
        self.D = (D + D.T) ** 2

        self.neighborhood = self.tau * (self.w[t] * self.D[t]).sum()
        self.regularizer = self.lam * torch.norm(self.D)

        return self.embedding + self.regularizer + self.neighborhood

