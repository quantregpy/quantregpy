import os, sys

extra_dll_dir = os.path.join(os.path.dirname(__file__), '.libs')

if sys.platform == 'win32' and os.path.isdir(extra_dll_dir):
  os.environ.setdefault('PATH', '')
  os.environ['PATH'] += os.pathsep + extra_dll_dir

#import _fortran
__all__ = ["quantreg"]
