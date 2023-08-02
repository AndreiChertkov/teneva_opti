import nevergrad as ng
import numpy as np
from teneva_opti import Opti


class OptiTens(Opti):
    def target(self, i):
        return self.target_tens(i)

    def _optimize_ng_helper(self, solver):
        if not self.is_n_equal:
            raise NotImplementedError

        parametrization = ng.p.TransitionChoice(
            range(self.n0), repetitions=self.d)
        parametrization.random_state.seed(self.seed)

        optimizer = solver(parametrization=parametrization,
            budget=None, num_workers=1)

        while True:
            x = optimizer.ask()
            i = np.array(x.value, dtype=int)
            y = self.target(i)
            if y is None or self.bm.m == self.bm.budget_m-1:
                break
            optimizer.tell(x, -y if self.is_max else y)

        # We call for the final recommendation:
        x = optimizer.provide_recommendation()
        i = np.array(x.value, dtype=int)
        for _ in range(2):
            # We repeat it to stop the Bm
            y = self.bm.get(i, skip_cache=True)
