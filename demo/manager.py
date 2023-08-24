"""Basic examples for manager usage."""
from teneva_bm import *
from teneva_opti import *


OPTIS = [OptiTensProtes, OptiTensTtopt, OptiTensPortfolio]


BMS_AGENT = [
    BmAgentAnt,
    BmAgentHuman,
    BmAgentHumanStand,
    BmAgentPendInv,
    BmAgentPendInvDouble,
    BmAgentSwimmer,
]


TASKS = []


for Opti in OPTIS:
    for seed in [0, 1]:
        TASKS.append({
            'bm': BmQuboMvc,
            'bm_args': {'d': 55, 'pcon': 3, 'seed': 99},
            'opti': Opti,
            'opti_args': {'m': 1.E+2, 'seed': seed},
        })


for Bm in BMS_AGENT:
    for Opti in OPTIS:
        TASKS.append({
            'bm': Bm,
            'bm_args': {'steps': 250},
            'opti': Opti,
            'opti_args': {'m': 1.E+2, 'seed': 12345},
        })


def demo():
    oman = OptiManager(TASKS, fold='result_demo_baseline')
    oman.run()

    oman = OptiManager(fold='result_demo_baseline', is_show=True)
    oman.load()
    oman.filter(d=100, bm_name='QuboKnapMvc')
    oman.sort_by_op(['protes', 'ttopt', 'portfolio'])
    oman.join_op_seed()
    print('\n\nLoaded result for QuboKnapMvc:\n')
    oman.show_text()

    oman.show_table('\n\nTable for mean:')
    oman.show_table('\n\nTable for best:', kind='best')
    oman.show_table('\n\nTable for wrst:', kind='wrst')

    oman.show_table('\n\nTable for mean time:', prec=1, is_time=True)
    oman.show_table('\n\nTable for best time:', kind='best', is_time=True)
    oman.show_table('\n\nTable for wrst time:', kind='wrst', is_time=True)


if __name__ == '__main__':
    demo()
