import numpy as np
import rqf

#def bandwidth_rq(p, n, hs = True, alpha = 0.05)
#{
#	# Bandwidth selection for sparsity estimation two flavors:
#	#	Hall and Sheather(1988, JRSS(B)) rate = O(n^{-1/3})
#	#	Bofinger (1975, Aus. J. Stat)  -- rate = O(n^{-1/5})
#	# Generally speaking, default method, hs=TRUE is preferred.
#
#	x0 <- qnorm(p)
#	f0 <- dnorm(x0)
#	if(hs)
#            n^(-1/3) * qnorm(1 - alpha/2)^(2/3) *
#                ((1.5 * f0^2)/(2 * x0^2 + 1))^(1/3)
#	else n^-0.2 * ((4.5 * f0^4)/(2 * x0^2 + 1)^2)^ 0.2
#}

def rqs_fit(x, y, tau = 0.5, tol = 0.0001):
  """ 
  function to compute rq fits for multiple y's
  """
  p = x.shape[1]
  n = x.shape[0]
  m = y.shape[1]

  flag, coef, e = rqf.rqs(
                          x,
                          y,
                          tau,
                          tol,
                          np.zeros([n]),
                          np.zeros([(n + 5) , (p + 2)]),
                          np.zeros(n))
  if(np.sum(flag)>0):
    if(np.any(flag==2)):
      print(f"{np.sum(flag==2)} out of {m} BS replications have near singular design")
    if(np.any(flag==1)):
      print(f"{np.sum(flag==1)} out of {m} may be nonunique")
      
  return(coef.T)