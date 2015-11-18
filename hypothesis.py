"""
Inferring 3D Shape from 2D Images

This file contains the abstract Hypothesis class.

Created on Aug 27, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

class Hypothesis:
    """
    Hypothesis class is an abstract class that specifies the template
    for an MCMC hypothesis.
    """
    def __init__(self):
        """
        Hypothesis class constructor
        """
        # p: prior, ll: likelihood
        # we want to cache these values, therefore we initialize them to None
        # prior and ll methods should calculate these once and return p and ll
        self.p = None
        self.ll = None
        pass

    def prior(self):
        """
        Returns prior probability p(H) of the hypothesis
        """
        pass

    def likelihood(self, data):
        """
        Returns the likelihood of hypothesis given data, p(D|H)
        """
        pass

    def copy(self, data):
        """
        Returns a (deep) copy of the hypothesis. Used for generating
        new hypotheses based on itself.
        """
        pass

class Proposal:
    """
    Proposal class implements MCMC moves (i.e., proposals) on Hypothesis.
    This is an abstract class specifying the template for Proposal classes.
    Propose method is called by MCMCSampler to get the next proposed hypothesis.
    """
    def __init__(self):
        pass

    def propose(self, h, *args):
        """
        Proposes a new hypothesis based on h
        Returns (information string, new hypothesis, probability of move, probability of reverse move)
        args: optional additional parameters
        """
        pass

