# %%
import numpy as np
from scipy import linalg as la
from sklearn.linear_model import MultiTaskElasticNetCV
from lib_using_einsum import compute_A_ElasticNet
from joblib import Parallel, delayed
import sys

def sqrt_mat_sym(M):
    # square root for symmetric matrix
    s, v = la.eigh(M)
    result = v @ np.diag(s**0.5) @ v.T
    # assert np.isclose(M, v @ np.diag(s) @ v.T).all()
    # assert np.isclose(result @ result, M).all()
    # assert result.shape == M.shape
    return result

# %%
#----------------------------------------------
# varying inputs: n; (different S_type)
#----------------------------------------------
rep = 100
n = 1000 # 1500, 2000
print("n=", n)

T = 20
S  = np.zeros((T, T)) 
S_type = "full_rank" # low_rank

if S_type == 'full_rank':
    for t in range(T):
        for tt in range(T):
            S[t, tt] = np.cos(t-tt) / (1+np.sqrt(abs(t-tt)))
if S_type == 'low_rank':
    rng = np.random.RandomState(1234)
    V = rng.randn(T, 10)/np.sqrt(10)
    S = V @ V.T 

p = int(1.5*n)
sparsity = int(p * 0.1)

# covariates covariance
Sigma = la.toeplitz(0.5 ** np.arange(p))
Sigma_square_root = sqrt_mat_sym(Sigma)


# fix regression coefficient
rng = np.random.RandomState(42)
B = np.zeros((p, T))
ind = rng.choice(p, sparsity)
B[ind, 0:T] = rng.normal(1, 1/p, size=(sparsity, T))
B =  B * np.sqrt(np.trace(S)/ np.trace(B.T@Sigma@B))

# %%
reg = MultiTaskElasticNetCV(fit_intercept=False, n_alphas=100,\
    l1_ratio=[.5, .7, .9, 1], cv=5)
# run 1 time
def one_run(seedid):
    # run simulation 1 time with a given seed
    rng = np.random.RandomState(seedid)
    Z = rng.randn(n, p)
    X = Z @ Sigma_square_root
    # generate noise, shape (n, T)
    E = rng.multivariate_normal(np.zeros(T), S, n)
    
    # generate coefficients matrix, shape (p, T)
    # B = np.zeros((p, T))
    # ind = rng.choice(p, sparsity)
    # B[ind, 0:T] = rng.normal(1, 1/p, size=(sparsity, T))
    # B =  B * np.sqrt(np.trace(S)/ np.trace(B.T@Sigma@B))
    # generate response, shape (n, T)
    Y = X @ B + E
    
    # oracle estimator 
    S_oracle = E.T @ E/n
    # method of moments estimator
    S_mm = (p + n + 1)/(n*(n+1)) * Y.T @ Y - 1/(n*(n+1)) * Y.T @ Z @ Z.T @ Y

    # Estimating coefficent by Multi-Task Elastic Net with Cross-validation
    regr = reg.fit(X, Y)
    A = compute_A_ElasticNet(X, regr.coef_, alpha=regr.alpha_, l1_ratio=regr.l1_ratio_)
    B_hat = regr.coef_.T
    
    F = Y - X @ B_hat # residuals, shape (n, T)
    # naive estimator
    S_naive = F.T @ F/n

    inv_fac = np.linalg.inv(np.eye(T) - A/n) # inverse factor
    temp = F.T @ ( (p + n)* np.eye(n) - Z @ Z.T ) @ F - A @ F.T @ F - F.T @ F @ A
    # proposed estimator
    S_hat = n**(-2) * inv_fac @ temp @ inv_fac
    
    est = np.array([S_naive, S_mm, S_hat, S_oracle])
    # pars = np.array([seedid, regr.alpha_, regr.l1_ratio_])
    # print(pars)
    return(est)

# %%
# run 100 repetitions using parallel computing
rep = 100
EST = Parallel(n_jobs=-1,verbose=10)(
    delayed(one_run)(seedid)
    for seedid in range(rep))
EST = np.array(EST) # shape (rep, 4, T, T)
np.save(f'EstimationResults/EST_n{n}_{S_type}.npy', EST)