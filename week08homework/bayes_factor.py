import math

class BayesFactor:
    def __init__(self, n, k):
        if not isinstance(n, int) or not isinstance(k, int):
            raise TypeError("n and k must be integers")
        if n < 0 or k < 0 or k > n:
            raise ValueError("Invalid values for n and k: n must be non-negative and 0 <= k <= n")
        self.n = n
        self.k = k

    def likelihood(self, theta):
        """
        Computes the likelihood for binomial data: P(k|n, theta) = comb(n, k) * theta^k * (1-theta)^(n-k)
        """
        if not (0 <= theta <= 1):
            raise ValueError("theta must be between 0 and 1 inclusive")
        
        # Using math.comb for binomial coefficient
        comb = math.comb(self.n, self.k)
        return comb * (theta**self.k) * ((1 - theta)**(self.n - self.k))

    def evidence_slab(self):
        """
        Computes the marginal likelihood under the slab prior: theta ~ Uniform(0, 1).
        Marginal likelihood = integral from 0 to 1 of [comb(n, k) * theta^k * (1-theta)^(n-k)] d_theta
        This is a Beta function B(k+1, n-k+1) multiplied by comb(n, k).
        comb(n, k) * Beta(k+1, n-k+1) = comb(n, k) * (k! * (n-k)!) / (n+1)!
        = [n! / (k!(n-k)!)] * [k!(n-k)! / (n+1)!] = 1 / (n+1).
        """
        return 1.0 / (self.n + 1)

    def evidence_spike(self):
        """
        Computes the marginal likelihood under the spike prior: point mass at theta = 0.5.
        Marginal likelihood = P(k|n, theta=0.5).
        """
        return self.likelihood(0.5)

    def bayes_factor(self):
        """
        Computes the ratio of evidence: BF = Evidence_spike / Evidence_slab.
        """
        e_spike = self.evidence_spike()
        e_slab = self.evidence_slab()
        return e_spike / e_slab
