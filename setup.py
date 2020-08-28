from numpy.distutils.core import Extension
import os

ext1 = Extension(name = 'rqf',
                 sources = ['./fortran/rqs.f', './fortran/rq0.f'])

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