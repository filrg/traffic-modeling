# seren-countfirst

A workload generator for LLM supervised fine-tuning (SFT) jobs, calibrated on the
Seren GPU-cluster trace of [AcmeTrace](https://github.com/InternLM/AcmeTrace). It
generates job arrivals and hardware-independent job sizes for any horizon and load,
and accompanies our AINTEC 2026 paper *Trace-Driven Stochastic Modeling for LLM
Supervised Fine-Tuning Workloads in Data Centers*.

The generator reproduces:

+ **Bursty** batch arrivals, with quiet and busy periods and dependence from 15
  minutes to a week (GLI-NB, a gated latent-intensity negative binomial model)
+ **Hardware-independent** job sizes in work units, whose distribution shifts with
  arrival volume
+ **Controllable load** through a single factor that scales the arrival intensity
+ **Controllable duration**
## Installation

Python 3.12 or higher.

```bash
git clone https://github.com/filrg/traffic-modeling.git
cd traffic-modeling
pip install -e ".[figures]"
```

## Usage

The fitted models are in `results/`.

```python
import numpy as np
from seren_countfirst.ensemble import state, winner

model, calibration = winner(state())  # the fitted GLI-NB model
rng = np.random.default_rng(0)

# 30 days of 15-minute batch counts at twice the trace's load
counts = model.simulate(96 * 30, rng, alpha=2.0 * calibration)
```

`results/workload_export.csv` is a generated job-level workload at the trace's
load: the arrival time (seconds) and size (work units) of each job over 138 days.

## Reproducing the paper

Download `data/job_trace/trace_seren.csv` from AcmeTrace into the repository root
(SHA-256 `747854c7...d14af9a1`, checked on load), then:

```bash
python reproduce.py --verify          # every stage (~45 min), diffed against results/
python figures/code/make_all.py       # every figure, into figures/{pdf,png,csv}
```

`python -m seren_countfirst.run_pipeline` runs single stages; `seren_countfirst/README.md`
describes the pipeline.

## Repository structure

```
seren_countfirst/   arrival and job-size models, evaluation, and the pipeline
traffic_modeling/   dataset, run identity and count-evaluation infrastructure
figures/code/       the paper's figures
results/            fitted models, evaluation tables and a generated workload
reproduce.py        rebuilds results/ from the trace
```

## Citation

The citation for our paper will be added on publication. Please also cite the
trace:

```bibtex
@inproceedings{hu2024characterization,
  author    = {Qinghao Hu and Zhisheng Ye and Zerui Wang and Guoteng Wang and Meng Zhang and
               Qiaoling Chen and Peng Sun and Dahua Lin and Xiaolin Wang and Yingwei Luo and
               Yonggang Wen and Tianwei Zhang},
  title     = {Characterization of Large Language Model Development in the Datacenter},
  booktitle = {21st USENIX Symposium on Networked Systems Design and Implementation (NSDI 24)},
  pages     = {709--729},
  year      = {2024}
}
```

## License

Code is released under the [MIT License](LICENSE). The contents of `results/` are
released under [CC BY 4.0](LICENSE-DATA); they derive from the AcmeTrace Seren
trace (CC BY 4.0), which is not redistributed here.
