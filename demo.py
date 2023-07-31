from teneva_bm import *
from teneva_opti import *


def demo(d=100, m=1.E+4):
    bm = BmQuboKnapDet(d)
    opti = OptiTensTtopt(bm, m, log_info=True, log_file=True)
    opti.opts(rank=4)
    opti.run()

    bm = BmQuboKnapDet(d)
    opti = OptiTensProtes(bm, m, log_info=True, log_file=True)
    opti.run()


if __name__ == '__main__':
    demo()
