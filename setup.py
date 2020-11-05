import setuptools
import os
import sys
req = ['numpy']

from setuptools import find_packages
lapackDir = "/usr/lib/lapack"
#if (input(f"Use default lapack location({lapackDir}) [Y]/n?").lower() == 'n'):
#lapackDir = input("Enter Lapack path: ")
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    config.add_subpackage('quantregpy')

    return config

metadata = dict(
    name='quantregpy',
    maintainer="David Kaftan and Paul Kaefer",
    maintainer_email="kaftand@gmail.com",
    description     = "Translation of R quantreg for python",
    url="https://github.com/quantregpy",
    license='GNU',
    python_requires='>=3.6',
    version="0.0.11",
    install_requires = ["patsy", "numpy", "scipy", "pandas", "scikit-learn"]
)
# Disable OSX Accelerate, it has too old LAPACK

# This import is here because it needs to be done before importing setup()
# from numpy.distutils, but after the MANIFEST removing and sdist import
# higher up in this file.
metadata['configuration'] = configuration
from distutils.command.sdist import sdist
#cmdclass={'sdist': sdist}
#metadata['cmdclass'] = cmdclass
if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**metadata)