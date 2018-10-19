# nonsmoothquasilinear

This repository contains a Python implementation accompanying the paper
[Optimal control of a non-smooth quasilinear elliptic equation](https://arxiv.org/abs/1810.08007)
by Christian Clason, Arnd Rösch, and Vũ Hữu Nhự.

The implementation uses Numpy (v1.15.2), SciPy (v1.1.0) and DOLFIN (v2018.1).
To run a representative example (`N=100`, `alpha = 1e-6`, `beta=0.8`), run `python3 nonsmoothquasilinear.py`.

If you find this code useful, you can cite the paper as

    @article{nonsmoothquasilinear,
        author = {Clason, Christian and Nhu, Vu Huu and Rösch, Arnd},
        title = {Optimal control of a non-smooth quasilinear elliptic equation},
        year = {2018},
        eprinttype = {arxiv},
        eprint = {1810.08007},
    }


