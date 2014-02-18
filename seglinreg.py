import logging
import numpy
import itertools


class SegLinReg:
    def __init__(self, segment_count):
        self.segment_count = segment_count
        self.first_pass_breakpoints_ratio = 2

    def calculate(self, data):
        return self.__second_pass(self.__first_pass(numpy.array(data)))

    def __first_pass(self, data):
        logging.debug("Initial data len: %s, segments: %s, first pass ratio: %s", len(data), self.segment_count,
                      self.first_pass_breakpoints_ratio)

        chunks = []
        bpt_count = self.first_pass_breakpoints_ratio * self.segment_count
        step = len(data) / bpt_count
        breakpoints = []
        for n in range(0, len(data), step):
            breakpoints.append(n)

        logging.debug("Breakpoints: %s", breakpoints)

        chunkset_candidates = []
        combinations = itertools.combinations(breakpoints, self.segment_count - 1)
        for comb in combinations:
            if comb[0] and comb[-1] != (len(data) - 1):
                logging.debug("Comb: %s", comb)
                comb = [0] + [x for x in comb] + [len(data) - 1]
                chunkset_candidates.append(SegLinRegResult(data, comb))
        logging.debug("Chunkset candidates: %s", chunkset_candidates)

        return chunks

    def __second_pass(self, bp):
        return (bp, 0)


class SegLinRegResult:
    def __init__(self, data=None, bpts=None):
        self.ss_res = None
        self.ss_tot = None
        self.chunks = []
        self.data = data
        if bpts:
            self.load_breakpoints(bpts)

    def load_breakpoints(self, breakpoints):
        logging.debug("BP: %s", breakpoints)
        self.chunks = []
        for n, bp in enumerate(breakpoints):
            if n < len(breakpoints) - 1:
                self.chunks.append({"start": bp, "end": breakpoints[n + 1] - 1})

    def __repr__(self):
        return "%s" % self.chunks

