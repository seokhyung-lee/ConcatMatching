# ConcatMatching: Generalised Concatenated Matching Decoder

This Python package implements the **concatenated matching decoder** for decoding arbitrary stabiliser codes.

The concatenated matching decoder was originally proposed for 2D color codes in [Quantum 9, 1609 (2025)](https://doi.org/10.22331/q-2025-01-27-1609) (see [color-code-stim](https://github.com/seokhyung-lee/color-code-stim) for its initial implementation). This repository extends the decoder to support any stabiliser code with a user-friendly interface.

**Note:** *This package is under development and may fail to find a valid solution for codes that are not 2D color codes.*

For decoding 2D color codes, this package offers a more streamlined approach compared to [color-code-stim](https://github.com/seokhyung-lee/color-code-stim). Simply provide a check matrix and some additional data, and the decoder will automatically decompose the Tanner graph using a graph colouring algorithm and execute the matching algorithm on each decomposed graph. No manual configuration is required.

## Installation

Need Python >= 3.10

```
pip install git+https://github.com/seokhyung-lee/ConcatMatching.git
```

## Usage

See [example.ipynb](https://github.com/seokhyung-lee/ConcatMatching/blob/main/example.ipynb).

## Citation

If you want to cite this module in an academic work, please cite the [original paper](https://doi.org/10.22331/q-2025-01-27-1609):

```
@article{lee2025color,
  doi = {10.22331/q-2025-01-27-1609},
  url = {https://doi.org/10.22331/q-2025-01-27-1609},
  title = {Color code decoder with improved scaling for correcting circuit-level noise},
  author = {Lee, Seok-Hyung and Li, Andrew and Bartlett, Stephen D.},
  journal = {{Quantum}},
  issn = {2521-327X},
  publisher = {{Verein zur F{\"{o}}rderung des Open Access Publizierens in den Quantenwissenschaften}},
  volume = {9},
  pages = {1609},
  month = jan,
  year = {2025}
}
```

## License

This module is distributed under the MIT license. Please see the LICENSE file for more details.