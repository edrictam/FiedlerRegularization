"""
This is the main loop that I use to run my experiments
"""

from experiments import run_experiment

for i in range(5):
    # run_experiment("mnist", 'l1')
    # run_experiment("mnist", 'dropout')
    # run_experiment("mnist", 'weight_decay')
    # run_experiment("cifar10", 'dropout')
    # run_experiment("cifar10", 'weight_decay')
    # run_experiment("cifar10", 'l1')
    # run_experiment("mnist", 'fiedler')
    # run_experiment("cifar10", 'fiedler')
    # run_experiment('tcga', 'weight_decay')
    # run_experiment('tcga', 'l1')
    # run_experiment('tcga', 'dropout')
    run_experiment('tcga', 'fiedler')
    # run_experiment('mnist', 'l1')