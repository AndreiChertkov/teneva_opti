from ttopt import TTOpt


from teneva_opti import OptiTens


DESC = """
    The TTOpt optimizer. See the repo ttpot:
    https://github.com/AndreiChertkov/ttopt
    and the paper "TTOpt: A maximum volume quantized tensor train-based
    optimization and its application to reinforcement learning":
    https://openreview.net/forum?id=Kf8sfv0RckB
"""


class OptiTensTtopt(OptiTens):
    def __init__(self, *args, **kwargs):
        super().__init__('ttopt', DESC, *args, **kwargs)

    def get_config(self):
        conf = super().get_config()
        conf['rank'] = self.rank
        conf['fs_opt'] = self.fs_opt
        return conf

    def info(self, footer=''):
        text = ''

        text += 'rank (TT-rank)                           : '
        v = self.rank
        text += f'{v}\n'

        text += 'fs_opt (transformation option)           : '
        v = self.fs_opt
        text += f'{v}\n'

        return super().info(text + footer)

    def set_opts(self, rank=4, fs_opt=1.):
        self.rank = rank
        self.fs_opt = fs_opt

    def _optimize(self):
        tto = TTOpt(f=self.target, d=self.d, n=self.n,
            evals=1.E+99, is_func=False, is_vect=True)
        tto.optimize(rank=self.rank, seed=self.seed,
            fs_opt=self.fs_opt, is_max=self.is_max)
