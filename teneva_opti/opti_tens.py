import nevergrad as ng
import numpy as np
from teneva_opti import Opti


class OptiTens(Opti):
    def target(self, i):
        return self.target_tens(i)

    def _optimize_ng_helper(self, solver):
        if not self.is_n_equal:
            raise NotImplementedError

        optimizer = solver(
            parametrization=ng.p.TransitionChoice(range(self.n0),
            repetitions=self.d),
            budget=1.E+99,
            num_workers=1)

        recommendation = optimizer.provide_recommendation()

        while True:
            x = optimizer.ask()
            i = np.array(x.value, dtype=int)
            y = self.target(i)
            if y is None:
                break
            optimizer.tell(x, y)
