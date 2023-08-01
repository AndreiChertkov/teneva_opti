import numpy as np
import os


from .utils import Log
from .utils import path


class OptiManager:
    def __init__(self, bms, optis, m=1.E+4, fold='result'):
        self.bms = bms
        self.optis = optis
        self.m = int(m)
        self.fold = fold

        fpath = path(os.path.join(self.fold, 'log'))
        self.log = Log(fpath)

    def info(self, opti):
        name = opti.name[:26]
        task = 'max' if opti.is_max else 'min'

        text = name + ' '*(27-len(name)) + ' | '

        if opti.is_fail:
            text += f'FAIL'
        else:
            text += f'{task} {opti.y_opt:-10.3e} | '
            text += f'time {opti.time_full:-7.1e} | '

        return text

    def info_bm(self, opti):
        name = opti.bm.name[:19]

        text = '>>> BM ' + name + ' '*(20-len(name))
        if len(opti.identity):
            text += ' [' + ', '.join(opti.identity) + ']'

        return text

    def run(self):
        for i, bm in enumerate(self.bms, 1):
            for j, Opti in enumerate(self.optis, 1):
                opti = Opti(bm, self.m, fold=self.fold,
                    log=False, log_info=True, log_file=True)

                if j == 1:
                    self.log(self.info_bm(opti) + '\n')

                opti.run(with_raise=False)
                self.log(self.info(opti))

                if opti.is_fail:
                    continue

                opti.save()

                if opti.bm.with_render:
                    opti.render()
                if opti.bm.with_show:
                    opti.show()

            self.log('\n')
