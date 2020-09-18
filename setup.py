from numpy.distutils.core import Extension
import os

lapackDir = "/usr/lib/lapack"
if (input(f"Use default lapack location({lapackDir}) [Y]/n?").lower() == 'n'):
    lapackDir = input("Enter Lapack path: ")

ext1 = Extension(name = 'rqf',
                 sources = ['./fortran/rqs.f', './fortran/rq0.f', './fortran/rq1.f', './fortran/rqbr.f', './fortran/rqfnb.f','./fortran/crq.f'],
                 library_dirs= [lapackDir],
                 libraries = ["lapack"])

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(name = 'quantregpy',
          packages          = ["quantregpy"],
          packagedir        = {"quantregpy":"./quantregpy/"},
          description       = "Translation of R quantreg for python",
          author            = "David Kaftan",
          author_email      = "kaftand@gmail.com",
          ext_modules = [ext1]
          )