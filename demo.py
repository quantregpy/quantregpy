import numpy as np
from quantregpy.quantreg import rq
import pandas as pd
df = pd.DataFrame(dict(a=[1,2,3,4], y=[2,3,4.5,5]))
out = rq("y ~ a", np.array([.5]), df, weights=np.array([1,1,0,1]))
print(out)