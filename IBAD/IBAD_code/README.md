# IBAD hypergraph experiments

Python code for **Inheritance of Connectivity: Modeling Higher Order Dynamic Networks with the Birth Death Process**.

The implementation uses node and hyperedge birth–death processes, preferential attachment, and single-proposal inheritance. The baseline similarity is a Gaussian kernel of the mean squared difference between raw degree histories. Direct deletion, uniform inheritance, Pearson similarities, log-history similarity, and feasible-target inheritance are also available.

## Installation

Use Python 3.10 or later and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run commands from this directory. Paths are resolved relative to the scripts. `environment\_tested.txt` records the environment used for integration checks. There are no absolute machine-specific data paths.

## Files

|File|Purpose|
|-|-|
|`hyper\_test1\_simple\_history.py`|Original procedural single-run simulation; saves ordinary lists in JSON|
|`run\_experiments.py`|Batch parameter sweeps using the single-run script|
|`plot\_population.py`|Population trajectories, stationary distributions and numerical summaries|
|`plot\_degree.py`|Node-degree and hyperedge-size distributions; descriptive slopes for rate/lifetime sweeps|
|`plot\_structure.py`|Clustering and largest-component statistics|
|`plot\_mechanism\_comparison.py`|Mechanism/similarity comparisons and metric tables|
|`results\_common.py`|Shared filtering, ordering and grouping|
|`real\_config.py`|Empirical data, simulation parameters, seeds and initial sizes|
|`prepare\_real.py`|Empirical conversion and reference statistics|
|`simulate\_real.py`|Simulation from small initial hypergraphs|
|`search\_m.py`|Attachment-parameter calibration and independent-seed evaluation|
|`plot\_real.py`|Empirical degree distributions, structural comparisons and divergences|
|`ibad\_functions.py`, `real\_utils.py`|Empirical simulation functions and shared metrics|
|`data/`|Supplied empirical input files|
|`tests/smoke\_test.py`|Temporary short simulations and end-to-end integration checks|



