import nevergrad as ng
import numpy as np
import os
from time import perf_counter as tpc


from .utils import Log
from .utils import path
from teneva_opti import __version__


class Opti:
    def __init__(self, name, desc, bm, m, seed=42, fold='result',
                 with_cache=True, log=True, log_info=False, log_file=False):
        self.name = name
        self.desc = desc
        self.seed = seed
        self.fold = fold

        self.bm = bm

        if self.bm.is_prep:
            if self.bm.budget_m != m:
                raise ValueError
            if self.bm.budget_m_cache != m:
                raise ValueError
            if self.bm.with_cache != with_cache:
                raise ValueError
        else:
            self.bm.set_cache(with_cache)
            self.bm.set_budget(m, m_cache=m)
            self.bm.prep()

        self.log = Log(self.fpath() if log_file else None, log, log_info)
        self.bm.set_log(self.log, prefix=self.name,
            cond=('max' if self.is_max else 'min'),
            with_max=self.is_max, with_min=not self.is_max)

        self.opts()

    @property
    def d(self):
        return self.bm.d

    @property
    def identity(self):
        dir = []

        for id in self.bm.identity:
            v = getattr(self.bm, id)

            if isinstance(v, (list, np.ndarray)):
                if isinstance(v[0], float):
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

        return dir

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

    def fpath(self, kind='log'):
        fpath = [self.fold, self.bm.name, kind, *self.identity, self.name]
        return os.path.join(*fpath)

    def get_config(self):
        """Return a dict with configuration of the optimizer and benchmark."""
        conf = {}
        conf['name'] = self.name
        conf['seed'] = self.seed
        conf['bm'] = self.bm.get_config()
        return conf

    def get_history(self):
        """Return a dict with optimization results."""
        hist = {}
        hist['bm'] = self.bm.get_history()
        return hist

    def info(self, footer=''):
        """Returns a detailed description of the optimizer as text."""
        text = '*' * 78 + '\n' + 'OPTI: '
        text += self.name + ' ' * max(0, 34-len(self.name)) +  ' | '
        text += f'DIMS = {self.d:-4d} | '
        n = np.mean(self.n)
        text += '<MODE SIZE> = ' + (f'{n:-7.1f}' if n<9999 else f'{n:-7.1e}')
        text += '\n'
        text += '-' * 41 + '|             '
        text += '>           Description'
        text += '\n'

        text += '.' * 78 + '\n'
        desc = f'    {self.desc.strip()}'
        text += desc.replace('            ', '    ')
        text += '\n'
        text += '.' * 78 + '\n'

        text += '-' * 41 + '|            '
        text += '>          Configuration'
        text += '\n'

        text += 'Package version                          : '
        v = __version__
        text += f'{v}\n'

        text += 'Random seed                              : '
        v = self.seed
        text += f'{v}\n'

        text += 'Optimizer                                : '
        v = self.__class__.__name__
        text += f'{v}\n'

        if footer:
            text += '-' * 41 + '|             '
            text += '>               Options'
            text += '\n'
            text += footer

        text += '#' * 78 + '\n'
        return text

    def load(self, fpath=None):
        """Load configuration and optimization result from npz file."""
        fpath = path(fpath or self.fpath('data'), 'npz')
        data = np.load(fpath, allow_pickle=True).get('data').item()
        return data

    def run(self):
        self.bm.init()
        self.log.info(self.info() + '\n' + self.bm.info())
        self._optimize()
        self.log.info(self.bm.info_history())
        return self

    def opts(self):
        return self

    def render(self, fpath=None):
        if self.bm.with_render:
            return self.bm.render(fpath or self.fpath('render'))
        else:
            self.log.wrn(f'Render is not supported for BM "{self.bm.name}"')

    def save(self, fpath=None):
        """Save configuration and optimization result to npz file."""
        data = {'config': self.get_config(), 'history': self.get_history()}
        fpath = path(fpath or self.fpath('data'), 'npz')
        np.savez_compressed(fpath, data=data)

    def show(self, fpath=None):
        if self.bm.with_show:
            return self.bm.show(fpath or self.fpath('show'))
        else:
            self.log.wrn(f'Show is not supported for BM "{self.bm.name}"')

    def target(self, inp):
        raise NotImplementedError

    def target_func(self, x):
        return self.bm.get_poi(x)

    def target_tens(self, i):
        return self.bm.get(i)

    def _optimize(self):
        raise NotImplementedError()

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
