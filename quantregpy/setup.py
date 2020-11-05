import setuptools
import os
req = ['numpy']

from setuptools import find_packages
lapackDir = "/usr/lib/lapack"
#if (input(f"Use default lapack location({lapackDir}) [Y]/n?").lower() == 'n'):
#lapackDir = input("Enter Lapack path: ")
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.core import Extension
    from numpy.distutils.system_info import (get_info, system_info, lapack_opt_info, blas_opt_info)

    config = Configuration('quantregpy', parent_package, top_path)

    rqfName = '_fortran'
    rqfSources = ['./_fortran/rqs.f', './_fortran/rq0.f', './_fortran/rq1.f', './_fortran/rqbr.f', './_fortran/rqfnb.f','./_fortran/crq.f']
    lapack_info = get_info('lapack_opt', 0)
    blasFiles = ["./_fortran/blas_src/dcopy.f", "./_fortran/blas_src/dgemv.f", "./_fortran/blas_src/lsame.f", "./_fortran/blas_src/xerbla.f", "./_fortran/blas_src/dpotrs.f", "./_fortran/blas_src/dposv.f", "./_fortran/blas_src/dsyr.f", "./_fortran/blas_src/dtrsm.f", "./_fortran/blas_src/dpotrf.f", "./_fortran/blas_src/dpotrf2.f", "./_fortran/blas_src/dsyrk.f", "./_fortran/blas_src/disnan.f", "./_fortran/blas_src/ilaenv.f", "./_fortran/blas_src/dgemm.f", "./_fortran/blas_src/ieeeck.f", "./_fortran/blas_src/iparmq.f", "./_fortran/blas_src/dlaisnan.f"]#[os.path.join("blas_src",f) for f in os.listdir("blas_src") if os.path.isfile(os.path.join("blas_src", f))]
    if lapack_info:
        config.add_extension(name = rqfName,
                    sources = rqfSources,
                    library_dirs= lapack_info['library_dirs'],
                    libraries = lapack_info['libraries'],
                )
    else:
        config.add_extension(name = rqfName,
                    sources = rqfSources + blasFiles,
                )
    config.add_data_files(*blasFiles)
    return config

if __name__ == "__main__":
  from numpy.distutils.core import setup
  setup(**configuration(top_path='').todict())
