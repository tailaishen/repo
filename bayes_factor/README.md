This directory contains files for implementing and testing the computation of Bayes factor for binomial data. The following is a brief description of the files:
**bayes_factor.py** includes the class `BayesFactor` which takes in the number of trials (n) and the number of successes (k). It contains methods for computing the likelihood, the evidence towards model slab, the evidence towards model spike, and the bayes factor.
**test/test_bayes_factor.py** includes the test suite `TestBayesFactor` that conducts six tests for invalid inputs, types of data returned, likelihood computation, the computation of evidence towards slab, the computation of evidence towards spike, and bayes factor computation. Within each test, there are multiple assert statements for testing difference cases. It is intended that `BayesFactor` passes all tests except the last assert statement for bayes factor computation. To run the test suite, run
`cd bayes_factor`
`python3 -m unittest tests/test_signal_detection.py`
**Dockerfile** is used to create the container image that runs bayes factor computation and testing.

