# Offline Model-Based Reinforcement Learning

[![Pipeline: passing](https://github.com/Mr-Pepe/offline-model-based-rl/actions/workflows/pipeline.yml/badge.svg)](https://app.codecov.io/github/Mr-Pepe/offline-model-based-rl)
[![codecov](https://codecov.io/github/Mr-Pepe/offline-model-based-rl/branch/main/graph/badge.svg?token=E7JIWU9FS9)](https://codecov.io/github/Mr-Pepe/offline-model-based-rl)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![Type checks: mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=555555)](https://pycqa.github.io/isort/)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


This library provides a simple but high-quality baseline for playing around with model-free and model-based reinforcement learning approaches in both online and offline settings.
It is mostly tested, type-hinted, and documented. Read the detailed documentation [here](https://offline-model-based-rl.readthedocs.io/en/latest/#).

The code in this repository is based on the code written as part of my master's thesis on uncertainty
estimation in offline model-based reinforcement learning.
Please [cite](#citation) accordingly.



# Contribute

Clone the repo and install the package and all required development dependencies with:

```
pip install -e .[dev]
```

After making changes to the code, make sure that static checks and unit tests pass by running `tox`.
Tox only runs unit tests that are not marked as `slow`.
For faster feedback from unit tests, run `pytest -m fast`.
Please run the slow tests if you have a GPU available by executing `pytest -m slow`.

# Citation

Feel free to use the code but please cite the usage as:

```
@misc{peter2021ombrl,
    title={Investigating Uncertainty Estimation Methods for Offline Reinforcement Learning},
    author={Felipe Peter and Elie Aljalbout},
    year={2021}
}
```
