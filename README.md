# GTMF
The GTMF (Ground Truth Maturity Framework) aims to assist teams in improving the quality of their ground truth (GT) data. To do this, it provides a range of methodologies, metrics, and tools that allow users to measure and understand their GT data more effectively. Please find more details in this [blog post](https://research.facebook.com/blog/2022/8/-introducing-the-ground-truth-maturity-framework-for-assessing-and-improving-ground-truth-data-quality/).

In this repository, we introduce the GTMF library, a toolkit we developed for the measurement and improvement of ground truth data. The library covers multiple dimensions including representativity, accuracy, reliability, metric variance, and efficiency. It provides APIs for general metrics in each dimension and allows for customized parameters and flexible measurement granularity. It also allows teams to build their own workflows wrapping up the GTMF metric APIs.

## Using GTMF

Available APIs of each dimensions are included in the following files:
* Representativity: [representativity.py](representativity.py). This dimension is also included in _[balance](https://import-balance.org/)_ (A python package for balancing biased data samples)
* Auccuracy: [accuracy.py](accuracy.py)
* Reliability: [reliability.py](reliability.py)
* Metric variance: [metric_variance.py](metric_variance.py)
* Efficiency: [efficiency.py](efficiency.py)

Jupyter Notebook Examples are under the folder [jupyter_notebook_example](jupyter_notebook_example)

## Installation

**Installation Requirements**
You may find out the requirements [here](requirements.txt).

## License

You may find out more about the license [here](LICENSE).
