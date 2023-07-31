import nevergrad as ng
import numpy as np
from time import perf_counter as tpc


from .utils import Log
from .utils import path


class Opti:
    def __init__(self, name, bm, m, seed=42, fold='result', with_cache=True,
                 log=True, log_info=False, log_file=False):
        self.name = name
        self.seed = seed

        self.bm = bm
        self.bm.set_cache(with_cache)
        self.bm.set_budget(m, m_cache=m)
        self.bm.set_log(log, prefix=self.name,
            cond='max' if self.is_max else 'min',
            with_max=self.is_max, with_min=not self.is_max)

        self.log = Log(self.fpath(fold) if log_file else None, log, log_info)

        self.opts()

    @property
    def d(self):
        return self.bm.d

    @property
    def is_max(self):
        return self.bm.is_opti_max

    @property
    def is_n_equal(self):
        return self.bm.is_n_equal

    @property
    def n(self):
        return self.bm.n

    @property
    def n0(self):
        return self.bm.n0

    def fpath(self, fold, kind='log'):
        dir = [fold, self.bm.name, kind]

        for id in self.bm.identity:
            v = getattr(self.bm, id)

            if isinstance(v, (list, np.ndarray)):
                if isinstance(v, float):
                    self.log.err('Float bm identity is not supported')
                v = self.bm.list_convert(v, 'int')

            if isinstance(v, (list, np.ndarray)):
                self.log.err('List-like bm identity is not supported')

            if isinstance(v, float):
                self.log.err('Float bm identity is not supported')

            if isinstance(v, bool):
                if not v:
                    continue
                dir.append(f'{id}')
            else:
                dir.append(f'{id}-{v}')

        name = self.name + ('.txt' if kind == 'log' else '.npz')
        dir.append(name)

        return '/'.join(dir)

    def get_config(self):
        return {}

    def run(self):
        self.bm.prep()
        self.log.info(self.bm.info())
        self.log('Optimization process:\n')
        self._optimize()
        self.log(self.bm.info_current('<<< DONE\n'))
        self.log.info(self.bm.info_history())
        return self

    def opts(self):
        return self

    def target(self, i):
        return self.bm.get(i)

    def target_func(self, x):
        return self.bm.get_poi(x)

    def _optimize(self):
        raise NotImplementedError()

    def _optimize_ng_helper(self, solver):
        optimizer = solver(
            parametrization=ng.p.TransitionChoice(range(self.bm.n0),
            repetitions=self.bm.d),
            budget=self.m * 10000000,
            num_workers=1)

        recommendation = optimizer.provide_recommendation()

        while True:
            x = optimizer.ask()
            i = np.array(x.value, dtype=int)
            y = self.target(i)
            if y is None:
                break
            optimizer.tell(x, y)
