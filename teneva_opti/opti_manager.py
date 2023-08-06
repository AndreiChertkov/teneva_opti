import numpy as np
import os


from .bm_view import BmView
from .utils import Log
from .utils import get_identity_str
from .utils import path


class OptiManager:
    def __init__(self, tasks=[], fold='result', machine='', is_show=False):
        self.tasks = tasks
        self.fold = fold
        self.machine = machine
        self.is_show = is_show

        fname = 'log_manager_show' if self.is_show else 'log_manager'
        fpath = os.path.join(self.fold, fname)
        self.log = Log(fpath)

        self.bms = []

    def build_args(self, args, bm):
        args['bm'] = args.get('bm', bm)
        args['fold'] = args.get('fold', self.fold)
        args['log'] = args.get('log', False)
        args['log_info'] = args.get('log_info', True)
        args['log_file'] = args.get('log_file', True)
        args['machine'] = args.get('machine', self.machine)
        return args

    def filter(self, d=None, n=None, bm_name=None, op_name=None, bm_seed=None):
        bms = []
        for bm in self.bms:
            if bm.filter(d, n, bm_name, op_name, bm_seed):
                bms.append(bm)
        self.bms = bms

    def info(self, bm, opti, len_max=21):
        text = ''

        name = bm.name[:(len_max-1)]
        text += '\n'
        text += '- BM   > ' + name + ' '*(len_max-len(name))
        text += ' [' + get_identity_str(bm, pretty=True) + ']'

        name = opti.name[:(len_max-1)]
        text += '\n'
        text += '- OPTI > ' + name + ' '*(len_max-len(name))
        text += ' [' + get_identity_str(opti, pretty=True) + ']'

        return text

    def info_history(self, opti, len_max=27):
        text = '  >>>>>> '

        if opti.is_fail:
            text += f'FAIL'
        else:
            task = 'max' if opti.is_max else 'min'
            text += f'{task}: {opti.y_opt:-14.5e}   '
            text += f'[time: {opti.time_full:-8.1e}]'

        return text

    def join_op_seed(self):
        bms = []
        for bm in self.bms:
            is_found = False
            for bmg in bms:
                if bmg.is_same(bm):
                    bmg.add(bm)
                    is_found = True
                    break
            if not is_found:
                bmg = BmView()
                bmg.add(bm)
                bms.append(bmg)

        self.bms = bms

    def load(self):
        self.bms = []

        def opts_str_to_dict(opts_str):
            opts = {}
            for opt_str in opts_str.split('__'):
                opt_list = opt_str.split('-')
                id = opt_list[0]
                if len(opt_list) > 1:
                    try:
                        v = int(opt_list[1])
                    except Exception as e:
                        v = opt_list[1]
                else:
                    v = True
                opts[id] = v
            return opts

        def check(data):
            if data['bm_name'] != data['config']['bm']['name']:
                raise ValueError('Invalid data')

            if data['op_name'] != data['config']['name']:
                raise ValueError('Invalid data')

            for id, v in data['bm_opts'].items():
                if not id in data['config']['bm']:
                    raise ValueError('Invalid data')
                if v != data['config']['bm'][id]:
                    raise ValueError('Invalid data')

            for id, v in data['op_opts'].items():
                if not id in data['config']:
                    raise ValueError('Invalid data')
                if v != data['config'][id]:
                    raise ValueError('Invalid data')

        fold1 = self.fold
        for bm_name in os.listdir(fold1):
            fold2 = os.path.join(fold1, bm_name)
            if os.path.isfile(fold2):
                continue
            fold3 = os.path.join(fold2, 'data')
            for bm_opts_str in os.listdir(fold3):
                fold4 = os.path.join(fold3, bm_opts_str)
                for op_opts_str in os.listdir(fold4):
                    fold5 = os.path.join(fold4, op_opts_str)
                    for op_file in os.listdir(fold5):
                        op_name = op_file.split('.npz')[0]
                        op_path = os.path.join(fold5, op_file)
                        data = np.load(op_path, allow_pickle=True)
                        data = data.get('data').item()
                        data['bm_opts'] = opts_str_to_dict(bm_opts_str)
                        data['op_opts'] = opts_str_to_dict(op_opts_str)
                        data['bm_name'] = bm_name
                        data['op_name'] = op_name
                        check(data)
                        self.bms.append(BmView(data))

    def run(self):
        for task in self.tasks:
            # Create Bm class instance:
            Bm = task['bm']
            args = task.get('bm_args', {})
            bm = Bm(**args)
            if 'bm_opts' in task:
                bm.set_opts(**task['bm_opts'])
            if 'bm_seed' in task:
                # TODO: seed should be the argument of bm.__init__
                bm.set_seed(task['bm_seed'])

            # Create Opti class instance:
            Opti = task['opti']
            args = task.get('opti_args', {})
            args = self.build_args(args, bm)
            opti = Opti(**args)
            if 'opti_opts' in task:
                opti.set_opts(**task['opti_opts'])

            # Run the optimization:
            self.log(self.info(bm, opti))
            opti.run(with_err=False)
            self.log(self.info_history(opti))

            # Save the results:
            opti.save()
            if not opti.is_fail:
                opti.render(with_wrn=False)
                opti.show(with_wrn=False)

    def show_table(self, prefix='', prec=2, kind='mean', is_time=False):
        bm_names = []
        for bm in self.bms:
            bm_names.append(bm.bm_name)
        bm_names = list(set(bm_names))
        if len(bm_names) > 1:
            raise ValueError(f'Invalid (more than 1 bm)')
        bm_name = bm_names[0]

        ops = {}
        for bm in self.bms:
            ops[bm.op_name] = ops.get(bm.op_name, 0) + 1
        for name, count in ops.items():
            if count > 1:
                raise ValueError(f'Invalid for opti "{name}" (repeated)')

        value_best = None
        for bm in self.bms:
            if bm.is_better(value_best, kind, is_time):
                value_best = bm.get(kind, is_time)

        if prefix:
            self.log(prefix)
        for bm in self.bms:
            self.log(bm.info_table(prec, value_best, kind, is_time))

    def show_text(self):
        for bm in self.bms:
            self.log(bm.info_text())

    def sort_by_bm(self, names):
        def ind(name):
            return names.index(name) if name in names else len(names)
        self.bms = sorted(self.bms, key=lambda bm: ind(bm.bm_name))

    def sort_by_op(self, names):
        def ind(name):
            return names.index(name) if name in names else len(names)
        self.bms = sorted(self.bms, key=lambda bm: ind(bm.op_name))
