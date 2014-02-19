import logging
import numpy
import itertools
from scipy.stats import stats
import math


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
        if bpt_count > len(data) / 2:
            bpt_count = math.floor(len(data) / 2) + 1

        step = int(math.ceil(len(data) / bpt_count))
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
                try:
                    chunkset = SegLinRegResult(data, comb)
                    chunkset_candidates.append(chunkset)
                except ValueError, exc:
                    logging.exception(exc)
        logging.debug("Chunkset candidates: %s", chunkset_candidates)

        best = max(chunkset_candidates, key=lambda chunkset: chunkset.r_2)
        logging.info("First pass best chunks: %s", best)

        return best

    def move_breakpoint(self, chunkset, n, direction):
        chunks = chunkset.chunks
        while True:
            prev_r2 = chunkset.r_2
            chunks[n]["end"] += direction
            chunks[n]["ss_tot"] = None

            chunks[n + 1]["start"] += direction
            chunks[n + 1]["ss_tot"] = None

            chunkset.r_2 = None


            if chunks[n]["end"] < chunks[n + 1]["end"]:
                chunkset.recalculate()
                logging.debug("Moved %s %s: %s", n, direction, chunkset)
            else:
                logging.debug("Break on single point")
                break

            if chunkset.r_2 < prev_r2:
                logging.debug("Break on r2: %s<%s", chunkset.r_2, prev_r2)
                break

        chunks[n]["end"] -= direction
        chunks[n]["ss_tot"] = None

        chunks[n + 1]["start"] -= direction
        chunks[n + 1]["ss_tot"] = None

        chunkset.recalculate()
        logging.debug("Done moves %s %s: %s", n, direction, chunkset)

    def __second_pass(self, chunkset):
        """ the algo is: we move 'end' of the chunk left and right to find local optimum """
        for n, chunk in enumerate(chunkset.chunks):
            if n == len(chunkset.chunks) - 1:
                continue

            self.move_breakpoint(chunkset, n, 1)
            self.move_breakpoint(chunkset, n, -1)

        return chunkset


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
        return "R^2=%s: %s" % (self.r_2, ["%s-%s" % (x["start"], x["end"]) for x in self.chunks])

    def __load_breakpoints(self, breakpoints):
        logging.debug("BP: %s", breakpoints)
        self.chunks = []
        for n, bp in enumerate(breakpoints):
            if n < len(breakpoints) - 1:
                self.chunks.append({"start": bp, "end": breakpoints[n + 1] - 1, "ss_res": None, "ss_tot": None})

        self.recalculate()


    def recalculate(self):
        for chunk in self.chunks:
            if chunk["ss_tot"] is None:
                subchunk = self.data[chunk["start"]:chunk["end"]]
                if not len(subchunk):
                    raise ValueError("Empty chunk: %s" % self.chunks)

                values = subchunk[:, 1]
                logging.debug("Regress for: %s", subchunk)
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(subchunk)
                except ValueError, exc:
                    logging.exception("Problems calculated stats.linregress for %s" % subchunk, exc)
                    raise exc
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
