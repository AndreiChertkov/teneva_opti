from teneva_bm import *
from teneva_opti import *


def demo(d=100, steps=250, m=1.E+2):
    bm = BmQuboKnapDet(d)

    opti = OptiTensTtopt(bm, m, log_info=True, log_file=True)
    opti.opts(rank=4)
    opti.run()


    print('\n\n')

    # opti.save()
    # data = opti.load()
    # print(data['history']['bm']['y_list'])
    # return

    opti = OptiTensProtes(bm, m, log_info=True, log_file=True)
    opti.run()

    print('\n\n')

    bm = BmAgentSwimmer(steps=steps)

    opti = OptiTensTtopt(bm, m, log_info=True, log_file=True)
    opti.opts(rank=4)
    opti.run()
    opti.render()
    opti.show()

    print('\n\n')

    opti = OptiTensProtes(bm, m, log_info=True, log_file=True)
    opti.run()
    opti.render()
    opti.show()


if __name__ == '__main__':
    demo()
