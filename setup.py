import setuptools
import os
req = ['numpy']

from setuptools import find_packages
lapackDir = "/usr/lib/lapack"
#if (input(f"Use default lapack location({lapackDir}) [Y]/n?").lower() == 'n'):
#lapackDir = input("Enter Lapack path: ")

if __name__ == "__main__":
    from numpy.distutils.core import Extension
    from numpy.distutils.system_info import (get_info, system_info, lapack_opt_info, blas_opt_info)
    rqfName = 'rqf'
    rqfSources = ['./fortran/rqs.f', './fortran/rq0.f', './fortran/rq1.f', './fortran/rqbr.f', './fortran/rqfnb.f','./fortran/crq.f']
    lapack_info = get_info('lapack_opt', 0)
    src_dir = 'lapack_lite'
    blasFiles = ["blas_src/dcopy.f", "blas_src/dgemv.f", "blas_src/lsame.f", "blas_src/xerbla.f", "blas_src/dpotrs.f", "blas_src/dposv.f", "blas_src/dsyr.f", "blas_src/dtrsm.f", "blas_src/dpotrf.f", "blas_src/dpotrf2.f", "blas_src/dsyrk.f", "blas_src/disnan.f", "blas_src/ilaenv.f", "blas_src/dgemm.f", "blas_src/ieeeck.f", "blas_src/iparmq.f", "blas_src/dlaisnan.f"]#[os.path.join("blas_src",f) for f in os.listdir("blas_src") if os.path.isfile(os.path.join("blas_src", f))]
    if lapack_info:
        ext1 = Extension(name = rqfName,
                    sources = rqfSources,
                    library_dirs= lapack_info['library_dirs'],
                    libraries = lapack_info['libraries'],
                )
    else:
        ext1 = Extension(name = rqfName,
                    sources = rqfSources + blasFiles,
                )

    from numpy.distutils.core import setup
    setup(name = 'quantregpy',
          packages          = find_packages(),
          description       = "Translation of R quantreg for python",
          author            = "David Kaftan",
          author_email      = "kaftand@gmail.com",
          version='0.0.9',
          install_requires = ["patsy", "numpy", "scipy", "pandas", "scikit-learn"],
          data_files =blasFiles,
          ext_modules = [ext1]
          )