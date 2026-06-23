import numpy as np
np.random.seed(0)

# Gated DeltaNet reference (matches ggml CPU ops.cpp gated_delta_net_one_chunk).
# State S is D x D, S[i,j] (i=key dim, j=value dim). Per token (vector gate / KDA):
#   S <- diag(a_t) @ S            (a_t[i] = exp(g_t[i]))   [decay rows by a]
#   u_t[j]   = sum_i S[i,j] k_t[i]                          [readout on DECAYED state]
#   delta_t[j] = beta_t (v_t[j] - u_t[j])
#   S[i,j]  += k_t[i] delta_t[j]                            [rank-1 update]
#   o_t[j]   = scale * sum_i S[i,j] q_t[i]                  [POST-update readout]
def sequential(q, k, v, g, beta, S0, scale):
    N, D = q.shape
    S = S0.astype(np.float64).copy()
    O = np.zeros((N, D))
    for t in range(N):
        a = np.exp(g[t])                       # (D,) decay per key-dim i
        S = (a[:, None]) * S                   # decay rows
        u = S.T @ k[t]                         # (D,) over j
        delta = beta[t] * (v[t] - u)           # (D,)
        S = S + np.outer(k[t], delta)          # rank-1
        O[t] = scale * (S.T @ q[t])            # post-update
    return O, S

# Chunked (single chunk = whole block), per agent B's design. Inclusive cumulative
# decay A_r[i] = prod_{s<=r} a_s[i]. Deflate by A to factor the decay out.
def chunked(q, k, v, g, beta, S0, scale):
    N, D = q.shape
    S0 = S0.astype(np.float64)
    a = np.exp(g.astype(np.float64))           # (N,D)
    A = np.cumprod(a, axis=0)                   # (N,D) inclusive cumulative decay
    Kbar = A * k                               # (N,D)  "later token" (carries A_r)
    Qbar = A * q
    Ktil = k / A                               # (N,D)  "earlier token" (carries 1/A_s)
    # carry from incoming state
    U_carry = Kbar @ S0                        # (N,D) over j
    O_carry = scale * (Qbar @ S0)
    # pairwise decay s->r is A_r/A_s  =>  Kbar_r . Ktil_s  (bounded for s<r)
    # strictly-lower T[r,s] = beta_r * (Kbar_r . Ktil_s) for s<r
    KK = Kbar @ Ktil.T                         # (N,N)  KK[r,s] = sum_i k_r k_s A_r/A_s
    T = np.tril(beta[:, None] * KK, k=-1)      # strict lower
    rhs = beta[:, None] * (v - U_carry)        # (N,D)
    # Dmat = (I+T)^-1 @ rhs  (unit lower-tri -> forward substitution)
    Dmat = np.linalg.solve(np.eye(N) + T, rhs)
    # intra-chunk output: lower-tri (incl diagonal) of (Qbar Ktil^T)
    QK = np.tril(Qbar @ Ktil.T, k=0)           # (N,N)
    O = O_carry + scale * (QK @ Dmat)
    # carry-out state: S_out = diag(A_{N-1}) S0 + Kw^T @ Dmat, Kw_r = (A_{N-1}/A_r) k_r
    Aend = A[-1]                                # (D,)
    Kw = (Aend[None, :] / A) * k               # (N,D)
    S_out = Aend[:, None] * S0 + Kw.T @ Dmat
    return O, S_out

for trial in range(5):
    N = np.random.randint(2, 17)   # block up to 16
    D = 64
    q = np.random.randn(N, D)*0.5
    k = np.random.randn(N, D)*0.5
    v = np.random.randn(N, D)*0.5
    g = -np.abs(np.random.randn(N, D))*0.1   # gates: log-decay <=0 (a<=1)
    beta = np.random.rand(N)
    S0 = np.random.randn(D, D)*0.3
    scale = 1.0/np.sqrt(D)
    Os, Ss = sequential(q,k,v,g,beta,S0,scale)
    Oc, Sc = chunked(q,k,v,g,beta,S0,scale)
    eO = np.abs(Os-Oc).max()
    eS = np.abs(Ss-Sc).max()
    print(f"trial {trial}: N={N} D={D} | max|dO|={eO:.2e} max|dS|={eS:.2e} | {'OK' if max(eO,eS)<1e-9 else 'MISMATCH'}")
