import numpy as np
import os


from .utils import Log
from .utils import get_identity_str
from .utils import path


class OptiManager:
    def __init__(self, tasks, fold='result', machine=''):
        self.tasks = tasks
        self.fold = fold
        self.machine = machine

        fpath = os.path.join(self.fold, 'log_manager')
        self.log = Log(fpath)

    def build_args(self, args, bm):
        args['bm'] = args.get('bm', bm)
        args['fold'] = args.get('fold', self.fold)
        args['log'] = args.get('log', False)
        args['log_info'] = args.get('log_info', True)
        args['log_file'] = args.get('log_file', True)
        args['machine'] = args.get('machine', self.machine)
        return args

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

    def run(self):
        for task in self.tasks:
            # Create Bm class instance:
            Bm = task['bm']
            args = task.get('bm_args', {})
            bm = Bm(**args)
            if 'bm_opts' in task:
                bm.set_opts(task['bm_opts'])

            # Create Opti class instance:
            Opti = task['opti']
            args = self.build_args(task.get('opti_args', {}), bm)
            opti = Opti(**args)
            if 'opti_opts' in task:
                opti.set_opts(task['opti_opts'])

            # Run the optimization:
            self.log(self.info(bm, opti))
            opti.run(with_err=False)
            self.log(self.info_history(opti))

            # Save the results:
            opti.save()
            if not opti.is_fail:
                opti.render(with_wrn=False)
                opti.show(with_wrn=False)
