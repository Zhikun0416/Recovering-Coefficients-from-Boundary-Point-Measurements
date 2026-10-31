"""Shared full protocol for both geometries and all four experiments."""
from .backend import Problem

PROTOCOL = 'smooth-boundary-full-v2'
BASE_LEVEL = 32
INVERSE_LEVEL = 64
DATA_LEVEL = 96
SENSOR_STD = 1.0


class CorrectedProblem:
    """Second-order response transferred to the independent data mesh."""
    def __init__(self, case, model, level=INVERSE_LEVEL, data_level=DATA_LEVEL):
        self.fine = Problem(case, model, level)
        self.coarse = Problem(case, model, level//2)
        self.beta = (1-(level/data_level)**2)/3
        self.model, self.case = model, case
        self.grid = self.fine.grid

    def prepare(self, rows):
        return self.fine.prepare(rows), self.coarse.prepare(rows)

    def measure(self, p, inputs, gradient=False):
        a = self.fine.measure(p, inputs[0], gradient=gradient)
        b = self.coarse.measure(p, inputs[1], gradient=gradient)
        if gradient:
            return tuple((1+self.beta)*x-self.beta*y for x,y in zip(a,b))
        return (1+self.beta)*a-self.beta*b

    def field_errors(self, truth, recovered):
        return self.fine.field_errors(truth, recovered)
