import numpy as np
from sklearn import mixture
np.random.seed(1)
g = mixture.GaussianMixture(n_components=2)
 # Generate random observations with two modes centered on 0
 # and 10 to use for training.
obs = np.concatenate((np.random.randn(100, 1), 10 + np.random.randn(300, 1)))
print(obs.shape)
print(g.fit(obs) )
# np.round(g.weights_, 2)
# array([ 0.75,  0.25])
# np.round(g.means_, 2)
# array([[ 10.05],
#        [  0.06]])
# np.round(g.covars_, 2) 
# array([[[ 1.02]],
#        [[ 0.96]]])
# g.predict([[0], [2], [9], [10]]) 

# np.round(g.score([[0], [2], [9], [10]]), 2)
# array([-2.19, -4.58, -1.75, -1.21])
# g.fit(20 * [[0]] +  20 * [[10]]) 
# GMM(covariance_type='diag', init_params='wmc', min_covar=0.001,
#         n_components=2, n_init=1, n_iter=100, params='wmc',
#         random_state=None, thresh=None, tol=0.001)
# np.round(g.weights_, 2)
# array([ 0.5,  0.5])