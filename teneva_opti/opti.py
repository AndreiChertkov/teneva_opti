import numpy as np
import os


from .utils import Log
from .utils import get_identity_str
from .utils import path
from teneva_opti import __version__


class Opti:
    def __init__(self, name, desc, bm, m=1.E+4, seed=0, fold='result_demo',
                 with_cache=True, log=True, log_info=False, log_file=False,
                 machine=''):
        self.machine = machine
        self.name = name
        self.desc = desc
        self.seed = seed
        self.fold = fold
        self.m = int(m)

        self.bm = bm

        if self.bm.is_prep:
            if self.bm.budget_m != m:
                raise ValueError('Invalid BM configuration ("m")')
            if self.bm.budget_m_cache != m:
                raise ValueError('Invalid BM configuration ("m_cache")')
            if self.bm.with_cache != with_cache:
                raise ValueError('Invalid BM configuration ("with_cache")')
        else:
            self.bm.set_cache(with_cache)
            self.bm.set_budget(m, m_cache=m)
            self.bm.prep()

        self.log = Log(self.fpath('log') if log_file else None, log, log_info)
        self.bm.set_log(self.log, prefix=self.name,
            cond=('max' if self.is_max else 'min'),
            with_max=self.is_max, with_min=not self.is_max)

        self.is_fail = False

        self.set_opts()

    @property
    def d(self):
        return self.bm.d

    @property
    def i_opt(self):
        return self.bm.i_max if self.is_max else self.bm.i_min

    @property
    def identity(self):
        return ['seed', 'm']

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

    @property
    def time_full(self):
        return self.bm.time_full

    @property
    def x_opt(self):
        return self.bm.x_max if self.is_max else self.bm.x_min

    @property
    def y_opt(self):
        return self.bm.y_max if self.is_max else self.bm.y_min

    def fpath(self, kind):
        id = get_identity_str(self)
        id_bm = get_identity_str(self.bm)
        return os.path.join(self.fold, self.bm.name, kind, id_bm, id, self.name)

    def get_config(self):
        """Return a dict with configuration of the optimizer and benchmark."""
        conf = {}
        conf['d'] = self.d
        conf['n'] = self.bm.list_convert(self.n, 'int')
        conf['m'] = self.m
        conf['seed'] = self.seed
        conf['name'] = self.name
        conf['machine'] = self.machine
        conf['opti'] = self.__class__.__name__
        conf['bm'] = self.bm.get_config()
        return conf

    def get_history(self):
        """Return a dict with optimization results."""
        hist = {}
        hist['err'] = 'Fail' if self.is_fail else ''
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

        if self.machine:
            text += 'Used machine                             : '
            v = self.machine
            text += f'{v}\n'

        text += 'Computation budget                       : '
        v = self.m
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

    def run(self, with_err=True):
        self.bm.init()
        self.log.info(self.info() + '\n' + self.bm.info())

        self.is_fail = False

        try:
            self._optimize()
        except Exception as e:
            self.is_fail = True
            msg = f'Optimization with "{self.name}" is failed [{e}]'
            self.log.err(msg) if with_err else self.log.wrn(msg)

        self.log.info(self.bm.info_history())

    def render(self, fpath=None, with_wrn=True):
        if self.bm.with_render:
            return self.bm.render(fpath or self.fpath('render'))
        elif with_wrn:
            self.log.wrn(f'Render is not supported for BM "{self.bm.name}"')

    def save(self, fpath=None):
        """Save configuration and optimization result to npz file."""
        data = {'config': self.get_config(), 'history': self.get_history()}
        fpath = path(fpath or self.fpath('data'), 'npz')
        np.savez_compressed(fpath, data=data)

    def set_opts(self):
        return

    def show(self, fpath=None, with_wrn=True):
        if self.bm.with_show:
            return self.bm.show(fpath or self.fpath('show'))
        elif with_wrn:
            self.log.wrn(f'Show is not supported for BM "{self.bm.name}"')

    def target(self, inp):
        raise NotImplementedError

    def target_func(self, x):
        return self.bm.get_poi(x)

    def target_tens(self, i):
        return self.bm.get(i)

    def _optimize(self):
        raise NotImplementedError()
