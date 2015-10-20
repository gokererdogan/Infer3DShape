"""
Inferring 3D Shape from 2D Images

This file contains the MCMC samplers for sampling from the posterior
over hypothesis given data.
It contains 
    MHSampler: Metropolis-Hastings sampler

Created on Aug 28, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import pandasql as psql
import cPickle

BEST_SAMPLES_LIST_SIZE = 20

class MCMCRun:
    """
    MCMCRun class holds information, e.g., probability, acceptance rate,
     samples, and best samples, related to a run of a MCMC chain.
    """
    def __init__(self, info, iteration_count, best_sample_count=BEST_SAMPLES_LIST_SIZE):
        self.info = info
        self.start_time = time.strftime("%Y.%m.%d %H:%M:%S")
        self.end_time = ""
        self.samples = SampleSet()
        self.best_samples = BestSampleSet(best_sample_count)
        self.iteration_count = iteration_count
        self.iter_df = pd.DataFrame(index=np.arange(0, self.iteration_count), dtype=np.float,
                                    columns=['Iteration', 'IsAccepted', 'Probability', 'LogProbability', 'MoveType'])

    def record_iteration(self, i, is_accepted, prob, move_type):
        self.iter_df.loc[i] = [i, is_accepted, prob, np.log(prob), move_type]

    def add_sample(self, s, prob, iter, info):
        self.samples.add(s, prob, iter, info)

    def add_best_sample(self, s, prob, iter, info):
        self.best_samples.add(s, prob, iter, info)

    def finish(self):
        self.end_time = time.strftime("%Y.%m.%d %H:%M:%S")

    def plot_probs(self):
        self.iter_df.plot('Iteration', 'LogProbability')

    def plot_acceptance_rate(self, window_size=100):
        # calculate moving average
        pd.rolling_mean(self.iter_df.IsAccepted, window=window_size).plot()

    def acceptance_rate_by_move(self):
        df = self.iter_df
        return psql.sqldf("select MoveType, AVG(IsAccepted) as AcceptanceRate from df group by MoveType", env=locals())

    def save(self, filename):
        cPickle.dump(obj=self, file=open(filename, 'wb'), protocol=2)

    @staticmethod
    def load(filename):
        return cPickle.load(open(filename, 'r'))


class SampleSet:
    """
    SampleSet class implements a simple list of samples.
    Each sample consists of the hypothesis, its posterior
    probability and some info associated with it.
    """
    def __init__(self):
        self.samples = []
        self.probs = []
        self.infos = []
        self.iters = []

    def add(self, s, prob, iter, info):
        self.samples.append(s)
        self.probs.append(prob)
        self.infos.append(info)
        self.iters.append(iter)

    def pop(self, i):
        if i < len(self.samples):
            s = self.samples.pop(i)
            prob = self.probs.pop(i)
            iter = self.iters.pop(i)
            info = self.infos.pop(i)
            return s, prob, iter, info
        return None

    def __getitem__(self, item):
        if item < len(self.samples):
            s = self.samples[item]
            prob = self.probs[item]
            iter = self.iters[item]
            info = self.infos[item]
            return s, prob, iter, info
        return None

class BestSampleSet(SampleSet):
    """
    BestSampleSet class implements a list of samples intended to
    keep the best samples (in terms of probability) so far in a
    chain. We add a sample to the set if it has higher probability
    than at least one of the samples in the set.
    """
    def __init__(self, capacity):
        SampleSet.__init__(self)
        self.capacity = capacity

    def add(self, s, prob, iter, info):
        if len(self.samples) < self.capacity:
            if s not in self.samples:
                SampleSet.add(self, s, prob, iter, info)
        elif prob > np.min(self.probs):
            if s not in self.samples:
                min_i = np.argmin(self.probs)
                self.pop(min_i)
                SampleSet.add(self, s, prob, iter, info)


class MHSampler:
    """
    Metropolis-Hastings sampler class.
    """
    def __init__(self, initial_h, data, kernel, burn_in, sample_count, best_sample_count, thinning_period,
                 report_period=500):
        """
        Metropolis-Hastings sampler constructor

        initial_h: Initial hypothesis (Hypothesis instance)
        data: observed data. Passed to Hypothesis.likelihood function
        kernel: Proposal class (Proposal instance)
        burn_in: Number of burn-in iterations
        sample_count: Number of samples
        best_sample_count: Number of highest probability samples to keep
        thinning_period: Number of samples to discard before getting the next sample
        report_period: Number of iterations to report sampler status.
        """
        self.initial_h = initial_h
        self.data = data
        self.kernel = kernel
        self.burn_in = burn_in
        self.sample_count = sample_count
        self.best_sample_count = best_sample_count
        self.thinning_period = thinning_period
        self.report_period = report_period
        self.iter_count = burn_in + (sample_count * thinning_period) + 1

    def sample(self):
        """
        Sample from the posterior over hypotheses given data using MH algorithm
        """
        h = self.initial_h

        run = MCMCRun("", self.iter_count, best_sample_count=self.best_sample_count)
        accepted_count = 0
        print("MHSampler Start\n")
        print("Initial Hypothesis\n")
        print(h)
        for i in range(self.iter_count):

            # propose next state
            move_type, hp, q_hp_h, q_h_hp = self.kernel.propose(h)

            # calculate acceptance ratio
            p_h = h.prior() * h.likelihood(self.data)
            p_hp = hp.prior() * hp.likelihood(self.data)
            # a(h -> hp)
            a_hp_h = (p_hp * q_h_hp) / (p_h * q_hp_h)

            is_accepted = 0
            # accept/reject
            if np.random.rand() < a_hp_h:
                is_accepted = 1
                accepted_count += 1
                h = hp
                p_h = p_hp
                # print("Iteration {0:d}: Accepted sample with probability {1:f}. \n".format(i, p_hp))
                # print("Iteration {0:d}: p {1:f} {2:f} a {3:f}. Move: {4:s}\n".format(i, p_h, p_hp, a_hp_h, info))

            run.record_iteration(i, is_accepted, p_h, move_type)

            if i > self.burn_in:
                if (i % self.thinning_period) == 0:
                    run.add_sample(h, p_h, i, move_type)

                run.add_best_sample(h, p_h, i, move_type)

            # report sampler state
            if (i % self.report_period) == 0:
                print("Iteration {0:d}, current hypothesis".format(i))
                print("Posterior probability: {0:e}".format(p_h))
                print(h)

        run.finish()
        print("Sampling finished. Acceptance ratio: {0:f}\n".format(float(accepted_count) / self.iter_count))
        return run


if __name__ == "__main__":
    import vision_forward_model as vfm
    import hypothesis as hyp
    fwm = vfm.VisionForwardModel(render_size=(200, 200))
    kernel = hyp.ShapeProposal(allow_viewpoint_update=True)

    # generate initial hypothesis shape randomly
    # parts = [hyp.CuboidPrimitive(np.array([0, 0, 0]), np.array([1.0, 0.75, 0.75]))]
    # h = hyp.Shape(fwm, parts=parts)
    h = hyp.Shape(fwm, viewpoint=[(3.0, -3.0, 3.0)])

    # read data (i.e., observed image) from disk
    # data = np.load('./data/stimuli20150624_144833/o1.npy')
    data = np.load('./data/test1_single_view.npy')
    '''
    # ground truth
    parts = [hyp.CuboidPrimitive(np.array([0.0, 0.0, 0.0]), np.array([1.0, .75, .75])),
             hyp.CuboidPrimitive(np.array([.75, 0.0, 0.0]), np.array([.5, .5, .5]))]


    # test 2
    parts = [hyp.CuboidPrimitive(np.array([0.0, 0.0, 0.0]), np.array([1.0, .75, .75])),
             hyp.CuboidPrimitive(np.array([.9, 0.0, 0.0]), np.array([.8, .5, .5])),
             hyp.CuboidPrimitive(np.array([0.0, 0.0, 0.75]), np.array([0.25, 0.35, 0.75])),
             hyp.CuboidPrimitive(np.array([0.0, 0.4, 0.75]), np.array([.2, .45, .25]))]


    gt = hyp.Shape(fwm, parts)
    '''
    sampler = MHSampler(h, data, kernel, 0, 10, 20, 200, 400)
    run = sampler.sample()
    print(run.best_samples.samples)
    print()
    print(run.best_samples.probs)

    # run.save('results/shape/shape_test3.pkl')

    for i, sample in enumerate(run.samples.samples):
        fwm.save_render("results/shape/test1/s{0:d}.png".format(i), sample)
    for i, sample in enumerate(run.best_samples.samples):
        fwm.save_render("results/shape/test1/b{0:d}.png".format(i), sample)

