from teneva_bm import *
from teneva_opti import *


def demo(d=100, steps=250, m=1.E+3):
    bm = BmQuboKnapDet(d)

    opti = OptiTensTtopt(bm, m, log_info=True, log_file=True)
    opti.opts(rank=4)
    opti.run()
    opti.save()


    print('\n\n')

    opti = OptiTensProtes(bm, m, log_info=True, log_file=True)
    opti.run()
    opti.save()

    print('\n\n')

    bm = BmAgentSwimmer(steps=steps)

    opti = OptiTensTtopt(bm, m, log_info=True, log_file=True)
    opti.opts(rank=4)
    opti.run()
    opti.render()
    opti.show()
    opti.save()

    print('\n\n')

    opti = OptiTensProtes(bm, m, log_info=True, log_file=True)
    opti.run()
    opti.render()
    opti.show()
    opti.save()

    print('\n\n')

    print('Demo for result loading (we present y_list below):')
    data = opti.load()
    print(data['history']['bm']['y_list'][:25])
    print('...')
    print(data['history']['bm']['y_list'][-25:])


if __name__ == '__main__':
    demo()
