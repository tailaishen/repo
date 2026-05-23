class BayesFactor:
    def __init__(self, n, k):
        raise NotImplementedError

    def likelihood(self, theta):
        raise NotImplementedError

    def evidence_slab(self):
        raise NotImplementedError

    def evidence_spike(self):
        raise NotImplementedError

    def bayes_factor(self):
        raise NotImplementedError