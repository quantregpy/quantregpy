import numpy as np
import pandas as pd
from .. import quantreg as qr

simple_data = pd.DataFrame(dict(x1 = [1.,1.,1.,1.,], x2 = [1.,2.,3.,4], y = [3.,4.,5.,6.]))

def test_rq_fit_fnb():
  mdl = qr.rq_fit_fnb(simple_data.loc[:,['x1','x2']].values, simple_data.y.values)
  assert np.all( mdl['coefficients'].shape == np.array([2]) )
  assert np.all(np.abs(mdl['coefficients'].flatten() - np.array([2.,1.])) < 1e-8)

def test_rq_fit_br():
  mdl = qr.rq_fit_br(simple_data.loc[:,['x1','x2']].values, simple_data.y.values)
  assert np.all( mdl['coefficients'].shape == np.array([2]) )
  assert np.all(np.abs(mdl['coefficients'].flatten() - np.array([2.,1.])) < 1e-8)

def test_rq_fit_fnc():
  mdl = qr.rq_fit_fnc(simple_data.loc[:,['x1','x2']].values, simple_data.y.values, np.zeros([2,2]), -np.ones(2))
  assert np.all( mdl['coefficients'].shape == np.array([2]) )
  assert np.all(np.abs(mdl['coefficients'].flatten() - np.array([2.,1.])) < 1e-8)

def test_rq_wfit():
  mdl = qr.rq_wfit(simple_data.loc[:,['x1','x2']].values, simple_data.y.values, 0.5, np.ones(4))
  assert np.all( mdl['coefficients'].shape == np.array([2]) )
  assert np.all(np.abs(mdl['coefficients'].flatten() - np.array([2.,1.])) < 1e-8)

def test_rqs_fit():
  multiple_ys = np.stack([simple_data.y.values, simple_data.y.values], axis=1)
  coef = qr.rqs_fit(simple_data.loc[:,['x1','x2']].values, multiple_ys, 0.5)
  assert np.all( coef.shape == np.array([2,2]) )
  assert np.all(np.abs(coef - np.array([[2.,1.],[2.,1.]])) < 1e-8)

def test_rq():
  mdl = qr.rq("y ~ x1 + x2 - 1", 0.5, simple_data)
  assert np.all( mdl['coefficients'].shape == np.array([2]) )
  assert np.all(np.abs(mdl['coefficients'].flatten() - np.array([2.,1.])) < 1e-8)

def test_rq_fit_pfnb():
  mdl = qr.rq_fit_pfnb(simple_data.loc[:,['x1']].values, simple_data.y.values, 0.5)
  assert np.all( mdl['coefficients'].shape == np.array([1,1]) )
