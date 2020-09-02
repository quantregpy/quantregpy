# quantregpy
A translation of the popular R library quantreg to python for quantile regression

Fortran code from the [quantreg R package](https://github.com/cran/quantreg) is modified to work with [f2py](https://numpy.org/doc/stable/f2py/)

R code from the [quantreg R package](https://github.com/cran/quantreg) is translated into python, making heavy use of [numpy](https://numpy.org).

The quantreg R package was accessed through [github](https://github.com/cran/quantreg) and distributed by GPL. I cannot overstate my gratitude for the original authors, who asked to be referenced as follows:

> Esmond G. Ng and Barry W. Peyton, "Block sparse Cholesky algorithms on advanced uniprocessor computers". SIAM J. Sci. Stat. Comput. 14  (1993), pp. 1034-1056.

> John R. Gilbert, Esmond G. Ng, and Barry W. Peyton, "An efficient algorithm to compute row and column counts for sparse Cholesky factorization". SIAM J. Matrix Anal. Appl. 15 (1994), pp. 1075-1091.

If this project matures, I will write better documentation. However, I am trying to mimic the R package, so [their documentation should be useful](https://cran.r-project.org/web/packages/quantreg/quantreg.pdf)


## Sample usage
    python setup.py install
    python
    >>> import numpy as np
    >>> from quantregpy.quantreg import rq
    >>> import pandas as pd
    >>> df = pd.DataFrame(dict(a=[1,2,3,4], y=[2,3,4,5]))
    >>> out = rq("y ~ a", np.array([.5]), df)
    >>> out
    {'coefficients': array([1., 1.]), ... etc
