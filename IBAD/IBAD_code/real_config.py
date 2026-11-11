"""Shared configuration for empirical preprocessing, calibration and simulation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
REAL_DIR = ROOT / 'real_data'
RESULT_DIR = ROOT / 'results_real'
NETWORKS = ['PPN', 'FW', 'EN']
USE_SEARCH_PARAMETERS = False
SELECTED_DIR = ROOT / 'm_search' / 'selected'

# Matrix rows define hyperedges; columns define nodes. Positive entries are binary connections.
DATASETS = {
    'PPN': dict(kind='matrix', file='M_PL_015..csv', nodes_axis='columns',
                drop_first_row=False, drop_isolated_nodes=False, threshold=0.0),
    'FW': dict(kind='matrix', file='FW_008..csv', nodes_axis='columns',
               drop_first_row=False, drop_isolated_nodes=False, threshold=0.0),
    'EN': dict(kind='email', sizes='email-Eu-full-nverts.txt',
               vertices='email-Eu-full-simplices.txt', fraction=0.002,
               edge_limit=470, deduplicate_hyperedges=False, keep_singletons=True),
}

# mu is a death rate; the mean lifetime is 1/mu. These are the manuscript settings.
PARAMETERS = {
    'PPN': dict(lambda_n=6.0, lambda_e=13.0, m_n=1, m_e=5,
                match_population=False, mu_n=1/111, mu_e=1/10),
    'FW': dict(lambda_n=4.0, lambda_e=13.0, m_n=1, m_e=3,
               match_population=False, mu_n=1/61, mu_e=13/248),
    'EN': dict(lambda_n=2.0, lambda_e=5.0, m_n=1, m_e=2,
               match_population=False, mu_n=1/187, mu_e=1/94),
}

INITIAL_SIZES = {'PPN': (10, 1), 'FW': (1, 10), 'EN': (10, 1)}
INITIAL_NETWORK = 'small'
SEEDS = [391176436, 391176437, 391176438]
DELETION_MODE = 'ibad_single'
SIMILARITY_RULE = 'history_rbf_raw'
SIMILARITY_SIGMA = 1.0
A_N = A_E = 1.0
MIN_HISTORY = 100  # Minimum common records; all common history records enter the similarity.
T_PRE = 100000
T = 20000
SAMPLE_EVERY = 400
TRACE_EVERY = 1000
COMPUTE_APL = True
SAVE_FINAL_NETWORK = True
