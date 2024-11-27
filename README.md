# Generalised Concatenated Matching Decoder

This Python package implements the **concatenated matching decoder** for decoding arbitrary stabiliser codes.

The concatenated matching decoder was originally proposed for 2D color codes in [arXiv:2404.07482](https://arxiv.org/abs/2404.07482) (see [color-code-stim](https://github.com/seokhyung-lee/color-code-stim) for its initial implementation). This repository extends the decoder to support any stabiliser code with a user-friendly interface.

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

If you want to cite this package in an academic work, please cite the original arXiv preprint:

```
@misc{lee2024color,
      title={Color code decoder with improved scaling for correcting circuit-level noise}, 
      author={Seok-Hyung Lee and Andrew Li and Stephen D. Bartlett},
      year={2024},
      eprint={2404.07482},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2404.07482}
}
```

## License

This module is distributed under the MIT license. Please see the LICENSE file for more details.