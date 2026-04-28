import math
import scipy.integrate

class BayesFactor:
    def __init__(self, n, k):
        if not isinstance(n, int) or not isinstance(k, int):
            raise TypeError("Not integer value(s)")
        if n < 0 or k < 0:
            raise ValueError("Negative values(s)")
        if k > n:
            raise ValueError("More successes than trials")
        self.n = n
        self.k = k

    def likelihood(self, theta):
        if theta < 0 or theta > 1:
            raise ValueError("Theta should be between 0 and 1")
        llh = math.comb(self.n, self.k) * theta**self.k * (1 - theta)**(self.n - self.k)
        return llh
    
    def evidence_slab(self):
        evidence, _ = scipy.integrate.quad(self.likelihood, 0, 1)
        return evidence
    
    def evidence_spike(self, a, b):
        if a >= b:
            raise ValueError("b should be greater than a")
        evidence, _ = scipy.integrate.quad(lambda theta: 1/(b-a)*self.likelihood(theta), a, b)
        return evidence
    
    def bayes_factor(self, a, b):
        bf = self.evidence_spike(a, b) / self.evidence_slab()
        return bf
    
    