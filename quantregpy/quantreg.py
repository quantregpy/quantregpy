import numpy as np
import rqf
from scipy.stats import norm
from scipy.stats import t as studentT
from sklearn.linear_model import LinearRegression
from copy import deepcopy
import pandas as pd
import collections
from patsy import dmatrices, DesignMatrix

Fit = collections.namedtuple(
	'Fit', 
	'na_action formula terms x_design y_design call tau weights residuals rho method model coefficients fitted_values na_message Class')

def bandwidth_rq(p, n, hs = True, alpha = 0.05):
	# Bandwidth selection for sparsity estimation two flavors:
	#	Hall and Sheather(1988, JRSS(B)) rate = O(n^{-1/3})
	#	Bofinger (1975, Aus. J. Stat)  -- rate = O(n^{-1/5})
	# Generally speaking, default method, hs=TRUE is preferred.

	x0 = norm.ppf(p)#qnorm(p)
	f0 = norm.pdf(x0)#dnorm(x0)
	if(hs):
		bandwidth = ( n**(-1./3.) * norm.ppf(1. - alpha/2.)**(2./3.) * ((1.5 * f0**2.)/(2. * x0**2. + 1.))**(1./3.) )
	else:
		bandwidth = n**-0.2 * ((4.5 * f0**4.)/(2. * x0**2. + 1.)**2.)**0.2
	return bandwidth

def print_rq(x : Fit, *args):
	print("Call:")
	print(x.call)
	coef = x.coefficients
	print("\nCoefficients:\n")
	print(coef, args)
	rank = x.rank
	nobs = x.residuals.shape[0]
	p = coef.shape[1] if( len(coef.shape) > 1) else coef.shape[0]
	rdf = nobs - p
	print(f"\nDegrees of freedom:{nobs}total;{rdf}residual\n")
	print(f"{x.na_message}\n")

def print_summary_rq(x, digits = 5, *args):
	print("\nCall: ")
	print(x.call)
	coef = x.coef
	tau = x.tau
	print("\ntau: ")
	print([round(t,digits) for t in tau.tolist()], args)
	print("\nCoefficients:\n")
	print([round(c, digits) for c in coef.tolist()], args)

def failIfMissingData(df):
	if df.isnull().values.any():
		print("Missing data in data")
		raise ValueError

def rq(formula : str, tau : np.array, data : pd.DataFrame, subset = None, weights = None,
			 na_action = failIfMissingData,
			 method = "br", model = True, contrasts = None, *args):
	tau = np.array([tau]) if type(tau) is float else tau
	expandedArgs = ", ".join(str(arg) for arg in args)
	baseCall = f"rq(formula = {formula}, tau = {tau}, data = {data}, subset = {subset}, weights = {weights}, na_action = {na_action}, method = {method}, model = {model}, constrasts = {contrasts}"
	call = baseCall + ", " + expandedArgs
	mf = baseCall + f", args = {args}"
	m = ",".join([param for param in mf.split(",") if param in ("formula", "data", "subset", "weights", "na_action")])
	Y, X = dmatrices(formula, data)
	mt = X.design_info.term_names
	eps = np.finfo(float).eps**(2/3)
	Rho = lambda u, tau: u * (tau - (u < 0)) 
	if(tau.shape[0]>1):
		if(np.any(tau < 0) or np.any(tau > 1)):
			print("invalid tau:  taus should be >= 0 and <= 1")
			raise ValueError
		tau[tau == 0] = eps
		tau[tau == 1] = 1 - eps
		coef = np.zeros((X.shape[1],tau.shape[0]))
		rho = np.zeros(tau.shape[0])
		fitted = np.zeros((X.shape[0], tau.shape[0]))
		resid = np.zeros((X.shape[0], tau.shape[0]))
		for i in range(tau.shape[0]):
			z = rq_wfit(X, Y, tau[i], weights, method, *args) if not (weights is None) else rq_fit(X, Y, tau[i], method, *args)
			coef[:,i] = z.coefficients
			resid[:,i] = z.residuals
			rho[i] = np.sum(Rho(z.residuals,tau[i]))
			fitted[:,i] = Y - z.residuals
		taulabs = f"tau={np.around(tau,3)}"
		fit = deepcopy(z)
		fit.coefficients = coef
		fit.residuals = resid
		fit.fitted_values = fitted
		if(method == "lasso"): 
			fit.Class = ("lassorqs","rqs")
		elif(method == "scad"):
			fit.Class = ("scadrqs","rqs")
		else:
			fit.Class = "rqs"
	else:
		process = (tau < 0) or (tau > 1)
		if(tau == 0):
			tau = eps
		if(tau == 1):
			tau = 1 - eps
		fit = rq_wfit(X, Y, tau, weights, method, *args) if not (weights is None) else rq_fit(X, Y, tau, method, *args) 
		if(process):
			rho = [fit.sol[1,:],fit.sol[3,:]]
		else:
			rho = np.sum(Rho(fit['residuals'],tau))  
	if(method == "lasso"):
		fit['Class'] = ("lassorq","rq")
	elif(method == "scad"):
		fit['Class'] = ("scadrq","rq")
	else:
		fit['Class'] = "rq.process" if process else "rq"
	fit['na_action'] = na_action
	fit['formula'] = formula
	fit['terms'] = mt
	fit['x_design'] = X.design_info
	fit['y_design'] = Y.design_info
	fit['call'] = call
	fit['tau'] = tau
	fit['weights'] = weights
	fit['rho'] = rho
	fit['method'] = method
	fit['na_message'] = "" # unsure what to do here m.na_message
	if(model):
		fit['model'] = mf
	return fit

def rq_fit(x : np.array, y : np.array, tau = 0.5, method = "br", *args):
	#if (method == "fn"): 
	#	fit = dict() #fit = rq_fit_fnb(x, y, tau, *args)
	#elif (method == "fnb"):
	#	fit = rq_fit_fnb(x, y, tau, *args)
	#elif (method == "fnc"):
	#	fit = rq_fit_fnc(x, y, tau, *args)
	#elif (method == "pfn"):
	#	fit = rq_fit_pfn(x, y, tau, *args)
	if (method == "br"):
		fit = rq_fit_br(x, y, tau, *args)
	#elif (method == "lasso"):
	#	fit = rq_fit_lasso(x, y, tau, *args)
	#elif (method == "scad"):
	#	fit = rq_fit_scad(x, y, tau = tau, *args)
	else:
		print(f"rq.fit.{method} not yet implemented")
		raise ValueError

	fit['fitted_values'] = y - fit['residuals']
	return fit

def dropNpColumn(npmat, j):
	if j == 0:
		return npmat[:,1:]
	elif j == npmat.shape[1] - 1:
		return npmat[:,:j]
	else:
		return np.concatenate((npmat[:,:j],npmat[:,:j+1]), axis=1)

def rq_wfit(x, y, tau, weights, method = "br",  *args):
	if(any(weights < 0)):
		raise ValueError("negative weights not allowed")
#	contr <- attr(x, "contrasts")
	if len(weights.shape) != 2:
		weights = weights.reshape((weights.shape[0],1))
	wx = x * weights 
	wy = y * weights

	#if (method == "fn"): 
	#fit = rq_fit_fnb(x, y, tau, *args)
	#elif (method == "fnb"):
	#	fit = rq_fit_fnb(x, y, tau, *args)
	#elif (method == "fnc"):
	#	fit = rq_fit_fnc(x, y, tau, *args)
	#elif (method == "pfn"):
	#	fit = rq_fit_pfn(x, y, tau, *args)
	if (method == "br"):
		fit = rq_fit_br(wx, wy, tau, *args)
	else:
		print(f"rq.fit.{method} not yet implemented")
		raise ValueError
	if(len(fit.get('sol',[])) > 0):
		fit['fitted_values'] = np.matmul( x , fit['sol'][3:,:])
	else:
		yhat = np.matmul( x , fit['coefficients'])
		ny = 1 if len(y.shape) == 1 else y.shape[1]
		fit['fitted_values'] = yhat.reshape((yhat.shape[0], ny))
	fit['residuals'] = y - fit["fitted_values"]
	fit['weights'] = weights
	return fit
#	fit$contrasts <- attr(x, "contrasts")
#	
#	fit
#}
# Function to compute regression quantiles using original simplex approach
# of Barrodale-Roberts/Koenker-d'Orey.  There are several options.
# The options are somewhat different than those available for the Frisch-
# Newton version of the algorithm, reflecting the different natures of the
# problems typically solved.  Succintly BR for "small" problems, FN for
# "large" ones.  Obviously, these terms are conditioned by available hardware.
#
# Basically there are two modes of use:
# 1.  For Single Quantiles:
#
#       if tau is between 0 and 1 then only one quantile solution is computed.
#
#       if ci = FALSE  then just the point estimate and residuals are returned
#		If the column dimension of x is 1 then ci is set to FALSE since
#		since the rank inversion method has no proper null model.
#       if ci = TRUE  then there are two options for confidence intervals:
#
#               1.  if iid = TRUE we get the original version of the rank
#                       inversion intervals as in Koenker (1994)
#               2.  if iid = FALSE we get the new version of the rank inversion
#                       intervals which accounts for heterogeneity across
#                       observations in the conditional density of the response.
#                       The theory of this is described in Koenker-Machado(1999)
#               Both approaches involve solving a parametric linear programming
#               problem, the difference is only in the factor qn which
#               determines how far the PP goes.  In either case one can
#               specify two other options:
#                       1. interp = FALSE returns two intervals an upper and a
#                               lower corresponding to a level slightly
#                               above and slightly below the one specified
#                               by the parameter alpha and dictated by the
#                               essential discreteness in the test statistic.
#				interp = TRUE  returns a single interval based on
#                               linear interpolation of the two intervals
#                               returned:  c.values and p.values which give
#                               the critical values and p.values of the
#                               upper and lower intervals. Default: interp = TRUE.
#                       2.  tcrit = TRUE uses Student t critical values while
#                               tcrit = FALSE uses normal theory ones.
# 2. For Multiple Quantiles:
#
#       if tau < 0 or tau >1 then it is presumed that the user wants to find
#       all of the rq solutions in tau, and the program computes the whole
#	quantile regression solution as a process in tau, the resulting arrays
#	containing the primal and dual solutions, betahat(tau), ahat(tau)
#       are called sol and dsol.  These arrays aren't printed by the default
#       print function but they are available as attributes.
#       It should be emphasized that this form of the solution can be
#	both memory and cpu quite intensive.  On typical machines it is
#	not recommended for problems with n > 10,000.
#	In large problems a grid of solutions is probably sufficient.
#
def rq_fit_br(x, y, tau = 0.5, alpha = 0.1, ci = False, iid = True,
	interp = True, tcrit = True):
		tol = np.finfo(float).eps**(2/3)
		eps = tol
		big = np.finfo(float).max**(2/3)
		p = x.shape[1]
		n = x.shape[0]
		ny = y.shape[1]
		nsol = 2
		ndsol = 2
		# Check for Singularity of X since br fortran isn't very reliable about this
		#storage.mode(y) <- "double"
		if (np.linalg.matrix_rank(x) < p):
				raise ValueError("Singular design matrix")
		if (tau < 0) or (tau > 1):
			nsol = 3 * n
			ndsol = 3 * n
			lci1 = False
			qn = np.array([0] * p)
			cutoff = 0
			tau = -1
		else:
				if (p == 1):
						ci = False
				if (ci):
					lci1 = True
					if (tcrit):
						cutoff = studentT.ppf(1 - alpha/2, n - p)
					else: 
						cutoff = norm.ppf(1 - alpha/2.)
					if (not iid):
						h = bandwidth_rq(tau, n, hs = True)
						bhi = rq_fit_br(x, y, tau + h, ci = False)
						bhi = bhi['coefficients']
						blo = rq_fit_br(x, y, tau - h, ci = False)
						blo = blo['coefficients']
						dyhat = np.matmul(x, (bhi - blo))
						if (np.any(dyhat <= 0)):
							pfis = (100 * np.sum(dyhat <= 0))/n
							print(f"{pfis}percent fis <=0")
						f = np.maximum(eps, (2 * h)/(dyhat - eps))
						qn = np.array([0]*p)
						for j in range(p):
							tempX = dropNpColumn(x, j)
							tempY = x[:,j]
							lr = LinearRegression().fit(tempX, tempY, sample_weight=f)
							qnj = lr.predict(tempX) - tempY
							qn[j] <- np.sum(qnj * qnj)
					else:
						qn = 1./np.diagonal(np.linalg.inv(np.matmul(x.T,x)))
				else:
						lci1 = False
						qn = np.array([0]*p)
						cutoff = 0
		sFor,waFor,wbFor,nsolFor,ndsFor= np.zeros([n]), np.zeros([(n + 5), (p + 4)]), np.zeros(n), nsol,ndsol
		tnmat = np.zeros([4,p])
		flag,coef,resid,sol,dsol,lsol, h, qn, cutoff, ci, tnmat = rqf.rqbr(p+3,x,y,tau,tol,sFor,waFor,wbFor,nsolFor,ndsFor,tnmat, big, lci1)
		if (flag != 0):
				if flag == 1:
					print("Solution may be nonunique")
				else:
					print("Premature end - possible conditioning problem in x")
		if (tau < 0) or (tau > 1):
				sol = sol[1:((p + 3) * lsol)]
				dsol = dsol[1:(n * lsol)]
				return({"sol" : sol, "dsol" : dsol})
		if (not np.any(ci)):
				dual = dsol.T.flatten()[0:n]
				yhatCols = 1 if len(coef.shape) < 2 else coef.shape[1]
				yhat = np.matmul(x, coef).reshape((x.shape[0], yhatCols))
				return(dict(coefficients = coef, x = x, y = y, residuals = y - yhat, dual = dual))
		if (interp):
				Tn = tnmat
				Tci = ci
				Tci[3, :] = Tci[3, :] + (np.abs(Tci[4, :] - Tci[3, :]) * (cutoff -
						np.abs(Tn[3, :])))/np.abs(Tn[4,: ] - Tn[3, :])
				Tci[2, :] = Tci[2, :] - (np.abs(Tci[1,: ] - Tci[2, :]) * (cutoff -
						np.abs(Tn[2, :])))/np.abs(Tn[1, :] - Tn[2, :])
				Tci[2, np.isnan(Tci[2,:]) ] = -big
				Tci[3, np.isnan(Tci[3,:]) ] = big
				coefficients = np.concatinate((coef,Tci[2:4, : ].T), axis = 1)
				residuals = y - np.matmul(x, coef)
				return(dict(coefficients = coefficients, residuals = residuals))
		else:
				Tci = ci
				coefficients = np.concatenate([coef, Tci.T], axis=1)
				residuals = y - np.matmul(x , coef)
				c_values = tnmat.T
				c_values = np.fliplr(c_values) 
				p_values = studentT.cdf(c_values, n - p) if (tcrit) else norm.cdf(c.values)
				return dict(coefficients = coefficients, residuals = residuals,
						c_values = c_values, p_values = p_values)

def rq_fit_fnb (x, y, tau = 0.5, beta = 0.99995, eps = 1e-06):
	n = y.shape[0]
	p = 1 if len(x.shape) == 1 else x.shape[1]
	if(n != x.shape[0]):
		raise ValueError("x and y don't match n")
	if (tau < eps) or (tau > 1 - eps):
		raise ValueError("No parametric Frisch-Newton method.  Set tau in (0,1)")
	rhs = (1 - tau) * np.sum(x, axis = 0)
	d   = np.ones(n)
	u   = np.ones(n)
	wn = np.zeros(10*n)
	wn[0:n] = (1-tau) #initial value of dual solution
	a = x.T
	info, wp = rqf.rqfnb(a,-y,rhs,d,u,beta,eps,wn)
#    z <- .Fortran("rqfnb", as.integer(n), as.integer(p), a = as.double(t(as.matrix(x))),
#        c = as.double(-y), rhs = as.double(rhs), d = as.double(d),as.double(u),
#        beta = as.double(beta), eps = as.double(eps),
#        wn = as.double(wn), wp = double((p + 3) * p),
#        it.count = integer(3), info = integer(1),PACKAGE= "quantreg")
#    if (z$info != 0)
#        stop(paste("Error info = ", z$info, "in stepy: singular design"))
#    coefficients <- -z$wp[1:p]
#    names(coefficients) <- dimnames(x)[[2]]
#    residuals <- y - x %*% coefficients
#    list(coefficients=coefficients, tau=tau, residuals=residuals)
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