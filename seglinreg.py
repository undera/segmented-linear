import logging
import numpy
import itertools
import math
from scipy import stats
import traceback


class SegLinRegAuto:
    def __init__(self, max_chunks):
        self.max_segments = max_chunks
        self.r2_threshold = 0.001


    def calculate(self, data):
        logging.info("Searching auto segments up to %s", self.max_segments)
        iterations = []
        for segments in range(2, self.max_segments + 1):
            reg = SegLinReg(segments)
            iterations.append(reg.calculate(data))
            logging.info("%s segments: %s", segments, iterations[-1])
            if segments > 2:
                if iterations[-1].r_2 < iterations[-2].r_2:
                    logging.info("R2 declines")
                    return iterations[-2]

                if iterations[-1].r_2 - iterations[-2].r_2 < self.r2_threshold:
                    logging.info("R2 growth below threshold")
                    return iterations[-2]

        logging.info("Segments limit reached")
        return iterations[-1]


class SegLinReg:
    def __init__(self, segment_count=2):
        self.segment_count = segment_count if segment_count > 1 else 2
        self.first_pass_breakpoints_ratio = 10

    def calculate(self, data):
        logging.info("Initial data len: %s, segments: %s, first pass ratio: %s", len(data), self.segment_count,
                     self.first_pass_breakpoints_ratio)

        arr = numpy.array(data)
        no_nulls = arr[numpy.where(arr[:, 1] > None)]
        logging.debug("Lens: %s/%s", len(arr), len(no_nulls))
        chunks = self.__first_pass(no_nulls)

        return self.__second_pass_unlimited(chunks)

    def __first_pass(self, data):
        #return SegLinRegResult(data, [x for x in range(0, len(data), len(data) / self.segment_count)])

        bpt_count = self.first_pass_breakpoints_ratio
        if bpt_count > len(data) / 2:
            bpt_count = math.floor(len(data) / 2) + 1

        step = len(data) / bpt_count
        breakpoints = []
        for n in range(0, len(data) + 1, int(step)):
            breakpoints.append(n)

        logging.debug("Breakpoints: %s", breakpoints)

        chunkset_candidates = []
        combinations = [x for x in itertools.combinations(breakpoints, self.segment_count - 1)]
        logging.debug("Combinations: %s", combinations)
        logging.debug("Len: %s", len(data))
        for comb in combinations:
            if comb[0] and comb[-1] != (len(data) - 1):
                comb = [0] + [x for x in comb] + [len(data) - 1]
                if self.__no_zero_chunks(comb):
                    chunkset = SegLinRegResult(data, comb)
                    chunkset_candidates.append(chunkset)

        for cand in sorted(chunkset_candidates, key=lambda xx: xx.r_2):
            logging.debug("Chunkset candidate: %s", cand)

        best = max(chunkset_candidates, key=lambda xx: xx.r_2)
        logging.debug("First pass best chunks: %s", best)

        return best

    def __move_breakpoint(self, chunkset, n, direction):
        chunks = chunkset.chunks
        while True:
            prev_r2 = chunkset.r_2
            chunks[n]["end"] += direction
            chunks[n]["ss_tot"] = None

            chunks[n + 1]["start"] += direction
            chunks[n + 1]["ss_tot"] = None

            chunkset.r_2 = None

            logging.debug("Try chunk %s,  step %s: %s", n, direction, chunkset)
            if chunks[n]["start"] < chunks[n]["end"] and chunks[n + 1]["start"] < chunks[n + 1]["end"]:
                chunkset.recalculate()
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

        chunkset.r_2 = None

        chunkset.recalculate()
        logging.debug("Done moves %s %s: %s", n, direction, chunkset)


    def __second_pass_limited(self, chunkset, step):
        """ the algo is: we move 'end' of the chunk left and right to find local optimum """
        for n, chunk in enumerate(chunkset.chunks):
            if n == len(chunkset.chunks) - 1:
                continue

            self.__move_breakpoint(chunkset, n, step)
            self.__move_breakpoint(chunkset, n, -step)

        return chunkset

    def __second_pass_unlimited(self, chunkset):
        logging.debug("Starting second pass: %s", chunkset)
        step = max([x['end'] - x['start'] for x in chunkset.chunks]) / 2
        while math.floor(step) > 0:
            pos_hash = ""
            while pos_hash != chunkset.get_position_hash():
                pos_hash = chunkset.get_position_hash()
                self.__second_pass_limited(chunkset, math.floor(step))
                logging.debug("Unlimited moved with step %s: %s", step, chunkset)
            step /= 2
        return chunkset

    def __no_zero_chunks(self, comb):
        prev = -2
        for bpt in comb:
            if prev >= bpt - 1:
                return False
            prev = bpt
        return True


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
        return "R^2=%s: %s" % (self.r_2, self.get_position_hash())

    def __load_breakpoints(self, breakpoints):
        logging.debug("BP: %s", breakpoints)
        self.chunks = []
        for n, bp in enumerate(breakpoints):
            if n < len(breakpoints) - 1:
                self.chunks.append(
                    {"start": bp, "end": breakpoints[n + 1] - 1, "ss_res": None, "ss_tot": None, "regress": []})

        self.recalculate()


    def recalculate(self):
        logging.debug("Calculating regression")
        for chunk in self.chunks:
            if chunk["ss_tot"] is None:
                subchunk = self.data[chunk["start"]:chunk["end"]]
                if not len(subchunk):
                    raise ValueError("Empty chunk: %s" % self.chunks)

                if len(subchunk) != chunk["end"] - chunk["start"]:
                    raise RuntimeError("Not similar!")

                logging.debug("Regress for %s: %s", chunk, subchunk)

                values = subchunk[:, 1]
                mean = numpy.array([sum(values) / len(subchunk)] * len(subchunk))

                try:
                    (slope, intercept, r, tt, stderr) = stats.linregress(subchunk)
                    #logging.info("a_s=%s, b_s=%s, r=%s, tt=%s, stderr=%s", slope, intercept, r, tt, stderr)
                except Exception, exc:
                    logging.debug("Exception in linregress: %s", traceback.format_exc(exc))
                    slope = 0
                    intercept = 0

                if math.isnan(slope):
                    slope = 0

                if math.isnan(intercept):
                    intercept = mean

                chunk["regress"] = numpy.array([(x, slope * x + intercept) for x in subchunk[:, 0]])

                regress = chunk["regress"][:, 1]
                chunk["ss_tot"] = sum([x * x for x in (values - mean)])
                chunk["ss_res"] = sum([x * x for x in (values - regress)])
                logging.debug("slope: %s, intercept: %s", slope, intercept)

        self.ss_res = 0
        self.ss_tot = 0
        for chunk in self.chunks:
            self.ss_res += chunk["ss_res"]
            self.ss_tot += chunk["ss_tot"]

        self.r_2 = 1 - (self.ss_res / self.ss_tot)

    def get_position_hash(self):
        return ' '.join(["%s:%s" % (x["start"], x["end"]) for x in self.chunks])

    def get_regression_data(self):
        for chunk in self.chunks:
            for val in chunk["regress"]:
                yield val
