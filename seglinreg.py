import logging
import numpy
import itertools
import math
from scipy import stats
import traceback


class SegLinReg:
    def __init__(self, segment_count=2):
        self.segment_count = segment_count if segment_count > 1 else 2
        self.first_pass_breakpoints_ratio = 2

    def calculate(self, data):
        logging.info("Initial data len: %s, segments: %s, first pass ratio: %s", len(data), self.segment_count,
                     self.first_pass_breakpoints_ratio)

        arr = numpy.array(data)
        chunks = self.__first_pass(arr[numpy.where(arr[:, 1] > None)])

        '''
        chunks.chunks[0]['end'] = 230
        chunks.chunks[1]['start'] = chunks.chunks[0]['end'] + 1
        chunks.chunks[0]["ss_tot"] = None
        chunks.chunks[1]["ss_tot"] = None
        chunks.recalculate()
        return chunks
        '''

        return self.__second_pass_unlimited(chunks)

    def __first_pass(self, data):
        #return SegLinRegResult(data, [x for x in range(0, len(data), len(data) / self.segment_count)])

        bpt_count = self.first_pass_breakpoints_ratio * self.segment_count
        if bpt_count > len(data) / 2:
            bpt_count = math.floor(len(data) / 2) + 1

        step = len(data) / bpt_count
        breakpoints = []
        for n in range(0, len(data), int(step)):
            breakpoints.append(n)

        logging.debug("Breakpoints: %s", breakpoints)

        chunkset_candidates = []
        combinations = [x for x in itertools.combinations(breakpoints, self.segment_count - 1)]
        logging.debug("Combinations: %s", combinations)
        for comb in combinations:
            if comb[0] and comb[-1] != (len(data) - 1) and self.__no_zero_chunks(comb):
                comb = [0] + [x for x in comb] + [len(data) - 1]
                chunkset = SegLinRegResult(data, comb)
                chunkset_candidates.append(chunkset)

        for cand in sorted(chunkset_candidates, key=lambda x: x.r_2):
            logging.debug("Chunkset candidate: %s", cand)

        best = max(chunkset_candidates, key=lambda x: x.r_2)
        logging.info("First pass best chunks: %s", best)

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
                logging.info("Unlimited moved with step %s: %s", step, chunkset)
            step /= 2
        return chunkset

    def __no_zero_chunks(self, comb):
        prev = -1
        for bpt in comb:
            if prev == bpt:
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
                slope, intercept, r = linreg(subchunk)
                mean = numpy.array([sum(values) / len(subchunk)] * len(subchunk))
                chunk["regress"] = numpy.array([(x, slope * x + intercept) for x in subchunk[:, 0]])
                regress = chunk["regress"][:, 1]
                chunk["ss_tot"] = sum([x * x for x in values - mean])
                chunk["ss_res"] = sum([x * x for x in regress - mean])
                logging.debug("slope: %s, intercept: %s", slope, intercept)

        self.ss_res = 0
        self.ss_tot = 0
        for chunk in self.chunks:
            self.ss_res += chunk["ss_res"]
            self.ss_tot += chunk["ss_tot"]

        self.r_2 =(self.ss_res / self.ss_tot)

    def get_position_hash(self):
        return ' '.join(["%s:%s" % (x["start"], x["end"]) for x in self.chunks])

    def get_regression_data(self):
        for chunk in self.chunks:
            for val in chunk["regress"]:
                yield val


def linreg(data):
    try:
        (a_s, b_s, r, tt, stderr) = stats.linregress(data)
        logging.debug("a_s=%s, b_s=%s, r=%s, tt=%s, stderr=%s", a_s, b_s, r, tt, stderr)
    except Exception, exc:
        logging.debug("Exception in linregress: %s", traceback.format_exc(exc))
        a_s = 0
        b_s = 0
        r = 0
    return a_s, b_s, r

    """
    Numpy's version of linregress caused div by zero errors
    http://www.answermysearches.com/how-to-do-a-simple-linear-regression-in-python/124/
    Summary
        Linear regression of y = ax + b
    Usage
        real, real, real = linreg(list, list)
    Returns coefficients to the regression line "y=ax+b" from x[] and y[], and R^2 Value
    """
    N = len(data)
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for x, y in data:
        Sx = Sx + x
        Sy = Sy + y
        Sxx = Sxx + x * x
        Syy = Syy + y * y
        Sxy = Sxy + x * y
    det = Sxx * N - Sx * Sx
    if det:
        a, b = (Sxy * N - Sy * Sx) / det, (Sxx * Sy - Sx * Sxy) / det
    else:
        a, b = (0, 0)
    return a, b