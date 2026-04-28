import math
import scipy.integrate

class BayesFactor:
    def __init__(self, n, k):
        self.n = n
        self.k = k

    def likelihood(self, theta):
        llh = math.comb(self.n, self.k) * theta**self.k * (1 - theta)**(self.n - self.k)
        return llh
    
    def evidence_slab(self):
        evidence, _ = scipy.integrate.quad(self.likelihood, 0, 1)
        return evidence
    
    def evidence_spike(self, a, b):
        evidence, _ = scipy.integrate.quad(lambda theta: 1/(b-a)*self.likelihood(theta), a, b)
        return evidence
    
    def bayes_factor(self, a, b):
        bf = self.evidence_spike(a, b) / self.evidence_slab()
        return bf
    
    