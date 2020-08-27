      subroutine rqs(m,n,k,m5,n2,a,b,tau,toler,ift,x,e,s,wa,wb)
	  integer s(m),m,n,k,m5,n2,ift(k)
      double precision a(m,n), b(m,k),x(n,k),e(m),wa(m5,n2),wb(m)
      double precision tau,toler
Cf2py intent(in) a,b,tau,toler,s,wa,wb
Cf2py intent(out) ift,x,e
      do23000 i=1,k
      call rq0(m,n,m5,n2,a,b(1,i),tau,toler,ift(i),x(1,i),e,s,wa,wb)
23000 continue
23001 continue
      return
      end
