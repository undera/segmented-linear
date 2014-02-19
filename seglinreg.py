import logging
import numpy
import itertools
from scipy.stats import stats


class SegLinReg:
    def __init__(self, segment_count):
        self.segment_count = segment_count
        self.first_pass_breakpoints_ratio = 2

    def calculate(self, data):
        return self.__second_pass(self.__first_pass(numpy.array(data)))

    def __first_pass(self, data):
        logging.info("Initial data len: %s, segments: %s, first pass ratio: %s", len(data), self.segment_count,
                     self.first_pass_breakpoints_ratio)

        chunks = []
        bpt_count = self.first_pass_breakpoints_ratio * self.segment_count
        if bpt_count > len(data):
            bpt_count = len(data)

        step = len(data) / bpt_count
        breakpoints = []
        for n in range(0, len(data), step):
            breakpoints.append(n)

        logging.debug("Breakpoints: %s", breakpoints)

        chunkset_candidates = []
        combinations = [x for x in itertools.combinations(breakpoints, self.segment_count - 1)]
        logging.debug("Combinations: %s", combinations)
        for comb in combinations:
            if comb[0] and comb[-1] != (len(data) - 1):
                comb = [0] + [x for x in comb] + [len(data) - 1]
                chunkset_candidates.append(SegLinRegResult(data, comb))
        logging.debug("Chunkset candidates: %s", chunkset_candidates)

        best = max(chunkset_candidates, key=lambda chunkset: chunkset.r_2)
        logging.info("First pass best chunks: %s", best)

        return best

    def __second_pass(self, bp):
        return (bp, 0)


class SegLinRegResult:
    def __init__(self, data=None, bpts=None):
        self.ss_res = None
        self.ss_tot = None
        self.r_2 = None
        self.chunks = []
        self.data = data
        if bpts:
            self.__load_breakpoints(bpts)

    def __repr__(self):
        return "R^2=%s: %s" % (self.r_2, self.chunks)

    def __load_breakpoints(self, breakpoints):
        logging.debug("BP: %s", breakpoints)
        self.chunks = []
        for n, bp in enumerate(breakpoints):
            if n < len(breakpoints) - 1:
                self.chunks.append({"start": bp, "end": breakpoints[n + 1] - 1, "ss_res": None, "ss_tot": None})

        self.__recalculate()

    def __recalculate(self):
        for chunk in self.chunks:
            if chunk["ss_tot"] is None:
                subchunk = self.data[chunk["start"]:chunk["end"]]
                values = subchunk[:, 1]
                logging.debug("Regress for: %s", subchunk)
                slope, intercept, r_value, p_value, std_err = stats.linregress(subchunk)
                mean = numpy.array([sum(values) / len(subchunk)] * len(subchunk))
                regress = numpy.array([slope * x + intercept for x in subchunk[:, 0]])
                chunk["ss_tot"] = sum([x * x for x in values - mean])
                chunk["ss_res"] = sum([x * x for x in regress - mean])
                logging.debug("slope: %s, intercept: %s, r_value: %s, p_value: %s, std_err: %s",
                              slope, intercept, r_value, p_value, std_err)

        self.ss_res = 0
        self.ss_tot = 0
        for chunk in self.chunks:
            self.ss_res += chunk["ss_res"]
            self.ss_tot += chunk["ss_tot"]

        self.r_2 = 1 - (self.ss_res / self.ss_tot )
