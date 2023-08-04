from copy import deepcopy as copy
import numpy as np


class BmView:
    def __init__(self, data=None):
        self.bms = []
        self.op_seed_list = []
        self.y_opt_list = []
        self.t_list = []
        self.is_init = False

        if data is not None:
            self.init_from_data(data)

    @property
    def is_group(self):
        return len(self.bms) > 0

    @property
    def t_best(self):
        return np.min(self.t_list)

    @property
    def t_mean(self):
        return np.mean(self.t_list)

    @property
    def t_wrst(self):
        return np.max(self.t_list)

    @property
    def y_opt_best(self):
        y = self.y_opt_list
        return np.max(y) if self.is_max else np.min(y)

    @property
    def y_opt_mean(self):
        return np.mean(self.y_opt_list)

    @property
    def y_opt_wrst(self):
        y = self.y_opt_list
        return np.min(y) if self.is_max else np.max(y)

    def add(self, bm):
        self.bms.append(bm)
        if len(self.bms) == 1:
            self.init_from_bm(bm)

        self.op_seed_list.append(bm.op_seed)
        self.y_opt_list.append(bm.y_opt)
        self.t_list.append(bm.t)

    def init_from_bm(self, bm):
        self.bm_config = bm.bm_config
        self.op_config = bm.op_config

        self.bm_history = bm.bm_history
        self.op_history = bm.op_history

        self.bm_opts = bm.bm_opts
        self.op_opts = bm.op_opts

        self.d = bm.d
        self.n = bm.n

        self.bm_name = bm.bm_name
        self.op_name = bm.op_name

        self.bm_seed = bm.bm_seed
        self.op_seed = None

        self.is_max = bm.is_max
        self.is_min = bm.is_min
        self.y_opt = bm.y_opt

        self.t = bm.t

        self.is_init = True

    def init_from_data(self, data):
        self.bm_config = copy(data['config']['bm'])
        self.op_config = copy(data['config'])
        del self.op_config['bm']

        self.bm_history = copy(data['history']['bm'])
        self.op_history = copy(data['history'])
        del self.op_history['bm']

        self.bm_opts = copy(data['bm_opts'])
        self.op_opts = copy(data['op_opts'])

        self.d = self.bm_config['d']
        self.n = self.bm_config['n']

        self.bm_name = self.bm_config['name']
        self.op_name = self.op_config['name']

        self.bm_seed = self.bm_config['seed']
        self.op_seed = self.op_config['seed']

        self.is_max = 'agent' in self.bm_name.lower() # TODO: fix it
        self.is_min = not self.is_max
        self.y_opt = self.bm_history['y_max' if self.is_max else 'y_min']

        self.t = self.bm_history['time_full']

        self.is_init = True

    def filter(self, d=None, n=None, bm_name=None, op_name=None, bm_seed=None):
        if d and self.d != d:
            return False
        if n and self.n != n:
            return False
        if bm_name and self.bm_name != bm_name:
            return False
        if op_name and self.op_name != op_name:
            return False
        if bm_seed and self.bm_seed != bm_seed:
            return False
        return True

    def info_table(self, prec=2, v_best=None):
        form = '{:-10.' + str(prec) + 'e}'

        v = self.y_opt_mean if self.is_group else self.y_opt
        v = form.format(v).strip()

        if v_best is not None:
            v_best = form.format(v_best).strip()

        text = '        & '
        if v == v_best:
            text += '\\fat{' + v + '}'
        else:
            text += v

        return text

    def info_text(self, len_max=21):
        text = ''

        name = self.bm_name[:(len_max-1)]
        pref = '- BM      > ' if self.is_group else '- BM   > '
        text += '\n'
        text += pref + name + ' '*(len_max-len(name))
        text += ' [' + self.get_opt_str(self.bm_opts, pretty=True) + ']'

        name = self.op_name[:(len_max-1)]
        pref = '- OPTI    > ' if self.is_group else '- OPTI > '
        opts = self.op_opts
        if self.is_group:
            opts['SEEDS'] = len(self.op_seed_list)
            del opts['seed']
        text += '\n'
        text += pref + name + ' '*(len_max-len(name))
        text += ' [' + self.get_opt_str(opts, pretty=True) + ']'

        task = 'max' if self.is_max else 'min'

        if self.is_group:
            text += '\n'
            text += '  > BEST >> '
            text += f'{task}: {self.y_opt_best:-14.5e}   '
            text += f'[time: {self.t_best:-8.1e}]'

            text += '\n'
            text += '  > MEAN >> '
            text += f'{task}: {self.y_opt_mean:-14.5e}   '
            text += f'[time: {self.t_mean:-8.1e}]'

            text += '\n'
            text += '  > WRST >> '
            text += f'{task}: {self.y_opt_wrst:-14.5e}   '
            text += f'[time: {self.t_wrst:-8.1e}]'

        else:
            text += '\n'
            text += '  >>>>>> '
            text += f'{task}: {self.y_opt:-14.5e}   '
            text += f'[time: {self.t:-8.1e}]'

        return text

    def is_same(self, bm, skip_op_seed=True):
        if not self.is_init:
            return True

        bm1 = self
        bm2 = bm

        if bm1.bm_name !=bm2.bm_name:
            return False

        if bm1.op_name !=bm2.op_name:
            return False

        for id in bm1.bm_opts.keys():
            if not id in bm2.bm_opts:
                return False
            if bm1.bm_opts[id] != bm2.bm_opts[id]:
                return False

        for id in bm2.bm_opts.keys():
            if not id in bm1.bm_opts:
                return False
            if bm1.bm_opts[id] != bm2.bm_opts[id]:
                return False

        for id in bm1.op_opts.keys():
            if id == 'seed' and skip_op_seed:
                continue
            if not id in bm2.op_opts:
                return False
            if bm1.op_opts[id] != bm2.op_opts[id]:
                return False

        for id in bm2.op_opts.keys():
            if id == 'seed' and skip_op_seed:
                continue
            if not id in bm1.op_opts:
                return False
            if bm1.op_opts[id] != bm2.op_opts[id]:
                return False

        return True

    def get_opt_str(self, opts, pretty=True):
        res = []
        for id, v in opts.items():
            if isinstance(v, bool):
                res.append(f'{id}')
            else:
                res.append(f'{id}: {v}' if pretty else f'{id}-{v}')
        return ('; ' if pretty else '__').join(res)
