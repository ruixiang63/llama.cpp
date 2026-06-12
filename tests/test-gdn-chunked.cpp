// Standalone CPU validation: chunked Gated-DeltaNet (ggml-op decomposition) vs the reference
// sequential ggml_gated_delta_net. Build: see the compile cmd at the bottom of this file.
// Goal: bitwise-ish match (max|diff| < 1e-3) so the chunked path can replace the sequential
// GDN kernel on the DFlash verify (block of N tokens), affording larger blocks -> ~3x.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

// Leaf inputs are created in a no_alloc context (so the graph can run on CUDA); their host data is
// staged here and uploaded with ggml_backend_tensor_set after the backend allocates the graph.
struct Pending { ggml_tensor * t; std::vector<float> data; };
static std::vector<Pending> g_pending;

static ggml_tensor * rnd(ggml_context * c, int64_t a,int64_t b,int64_t d,int64_t e, std::mt19937 & g, float sc, float bias=0.f){
    ggml_tensor * t = ggml_new_tensor_4d(c, GGML_TYPE_F32, a,b,d,e);
    std::vector<float> h(ggml_nelements(t));
    std::normal_distribution<float> N(0,1);
    for (auto & x : h) x = N(g)*sc + bias;
    g_pending.push_back({t, std::move(h)});
    return t;
}

// L2-normalize each ne0 vector to unit norm (delta-net normalizes q/k). Without this, random keys
// have ||k||^2 ~ S_v*sc^2 >> 2, so beta*||k||^2 violates the delta-rule stability bound and the TRUE
// recurrence diverges -> ref and chunked blow up in the unstable directions for long sequences.
static void l2norm_rows(ggml_tensor * t){
    for (auto & pd : g_pending) if (pd.t == t) {
        const int64_t S = t->ne[0]; const int64_t rows = (int64_t)pd.data.size()/S;
        for (int64_t r=0;r<rows;++r){ float * v=pd.data.data()+r*S; double n=0; for(int64_t i=0;i<S;++i)n+=(double)v[i]*v[i];
            float inv = n>0 ? (float)(1.0/sqrt(n)) : 0.f; for(int64_t i=0;i<S;++i)v[i]*=inv; }
        return;
    }
}

// pick CUDA if GDN_BACKEND=CUDA and a matching device is registered, else CPU
static ggml_backend_t make_backend(){
    const char * want = getenv("GDN_BACKEND");
    if (want) {
        for (size_t i=0;i<ggml_backend_dev_count();++i){
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            const char * nm = ggml_backend_dev_name(d);
            const char * ds = ggml_backend_dev_description(d);
            if ((nm && strstr(nm, want)) || (ds && strstr(ds, want))) return ggml_backend_dev_init(d, nullptr);
        }
        fprintf(stderr, "GDN_BACKEND=%s not found, using CPU\n", want);
    }
    return ggml_backend_cpu_init();
}

// Chunked GDN as a ggml subgraph. Inputs match ggml_gated_delta_net:
//   q,k,v,g [S_v,H,N,1], beta [1,H,N,1], S0 [S_v,S_v,H,1]. Returns attn [S_v,N,H] and writes S_out.
// All math from the validated oracle (gdn_chunked_oracle.py). Single chunk (whole block).
static void build_chunked(ggml_context * c, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                          ggml_tensor * g, ggml_tensor * beta, ggml_tensor * S0,
                          ggml_tensor ** out_attn, ggml_tensor ** out_state) {
    const int64_t S_v = v->ne[0], H = v->ne[1], N = v->ne[2];
    const int64_t Hk = q->ne[1]; // GQA: q/k have Hk heads, broadcast (interleaved iv%Hk) to H v-heads
    const float scale = 1.0f/sqrtf((float)S_v);
    // reorg [S_v,Hx,N] -> [S_v,N,Hx]
    auto toDNH = [&](ggml_tensor * x){ return ggml_cont(c, ggml_permute(c, x, 0,2,1,3)); };
    ggml_tensor * qp = toDNH(q), * kp = toDNH(k), * vp = toDNH(v), * gp = toDNH(g); // q/k:[S_v,N,Hk] v/g:[S_v,N,H]
    if (Hk != H) { // expand q/k heads Hk->H, interleaved (ggml_repeat tiles ne2: h -> h%Hk)
        ggml_tensor * tgt = ggml_new_tensor_3d(c, GGML_TYPE_F32, S_v, N, H);
        qp = ggml_repeat(c, qp, tgt);
        kp = ggml_repeat(c, kp, tgt);
    }
    // A_r[i] = prod_{s<=r} exp(g): cumsum over tokens. cumsum is ne0-only -> put tokens on ne0.
    ggml_tensor * gN = ggml_cont(c, ggml_permute(c, gp, 1,0,2,3));     // [N,S_v,H]
    ggml_tensor * L  = ggml_cumsum(c, gN);                            // [N,S_v,H] cumulative log-decay
    ggml_tensor * Lp = ggml_cont(c, ggml_permute(c, L, 1,0,2,3));     // [S_v,N,H]
    ggml_tensor * A  = ggml_exp(c, Lp);                              // [S_v|1, N, H]
    ggml_tensor * Ainv = ggml_exp(c, ggml_neg(c, Lp));               // 1/A
    // full-size tensor first so a scalar gate's A=[1,N,H] broadcasts into kp/qp=[S_v,N,H]
    ggml_tensor * Kbar = ggml_mul(c, kp, A);
    ggml_tensor * Qbar = ggml_mul(c, qp, A);
    ggml_tensor * Ktil = ggml_mul(c, kp, Ainv);
    // beta -> [1,N,H] (broadcast over dim)
    ggml_tensor * betaNH = ggml_cont(c, ggml_permute(c, beta, 0,2,1,3)); // [1,N,H]
    // U_carry[j,r] = sum_i Kbar[i,r] S0[i,j] ; O_carry = scale Qbar . S0
    ggml_tensor * Ucar = ggml_mul_mat(c, S0, Kbar);   // [j, r, H]  (contract i=ne0)
    ggml_tensor * Ocar = ggml_scale(c, ggml_mul_mat(c, S0, Qbar), scale); // [j,r,H]
    // rhs[j,r] = beta_r (v[j,r] - Ucar[j,r])   (vp is [S_v(j),N(r),H])
    ggml_tensor * rhs = ggml_mul(c, ggml_sub(c, vp, Ucar), betaNH);  // [S_v(j),N(r),H]
    // KK[s,r] = sum_i Ktil[i,s] Kbar[i,r] = Kbar_r . Ktil_s. solve_tri wants A[ne0=s, ne1=r]
    // = (I+T)[r,s], so KK is already in the right orientation (no transpose). beta_r over ne1=r.
    ggml_tensor * KK   = ggml_mul_mat(c, Ktil, Kbar);             // [s,r,H]
    ggml_tensor * Tfull= ggml_mul(c, KK, betaNH);                 // [s,r,H] * beta_r(ne1)
    ggml_tensor * Tlo  = ggml_tri(c, Tfull, GGML_TRI_TYPE_LOWER); // keep s<r (strict)
    // (I+T): add identity. Build it the SAME way the model does (no host fill -> works with no_alloc
    // / CUDA): all-ones [N,1] via exp(0*beta) then ggml_diag -> [N,N], broadcast over H.
    ggml_tensor * b0   = ggml_cont(c, ggml_transpose(c, ggml_view_2d(c, betaNH, 1, N, betaNH->nb[1], 0)));
    ggml_tensor * ones = ggml_exp(c, ggml_scale(c, b0, 0.0f));    // [N,1] all-ones
    ggml_tensor * Imat = ggml_diag(c, ones);                      // [N,N]
    ggml_tensor * IT = ggml_add(c, Tlo, Imat);                    // [s,r,H] = (I+T) in solve orientation
    // Dmat = (I+T)^-1 rhs. b = rhs [S_v(j), N(r), H] (ne1=N matches A->ne1). result [S_v(j),N(r),H].
    ggml_tensor * Dmat = ggml_solve_tri(c, IT, rhs, true, true, false); // [S_v(j), N(token), H]
    Dmat = ggml_cont(c, ggml_transpose(c, Dmat));                 // -> [N(token), S_v(j), H]
    // O = O_carry + scale * tril(Qbar.Ktil^T incl-diag) @ Dmat. QK[s,r]=Qbar_r.Ktil_s.
    ggml_tensor * QK   = ggml_mul_mat(c, Ktil, Qbar);            // [s,r,H]
    ggml_tensor * QKlo = ggml_tri(c, QK, GGML_TRI_TYPE_LOWER_DIAG); // keep s<=r (incl diagonal)
    // intra[r,j] = sum_s QKlo[s,r] Dmat[s,j].  mul_mat(QKlo[s,r,H], Dmat[token=s,j,H]) -> [r,j,H]
    ggml_tensor * intra = ggml_mul_mat(c, QKlo, Dmat);          // contract s=ne0 -> [r,j,H]
    intra = ggml_scale(c, intra, scale);                        // [r,j,H]
    // O_carry is [j,r,H]; intra is [r,j,H] -> transpose O_carry
    ggml_tensor * OcarT = ggml_cont(c, ggml_transpose(c, Ocar)); // [r,j,H]
    ggml_tensor * O = ggml_add(c, OcarT, intra);                // [r(N),j(S_v),H]  attn per token
    *out_attn = O;
    // S_out[i,j] = A_end[i] S0[i,j] + sum_r Kw[i,r] Dmat[r,j],  Kw_r=(A_end/A_r) k_r
    ggml_tensor * Aend = ggml_view_3d(c, A, A->ne[0], 1, H, A->nb[1], A->nb[2], (N-1)*A->nb[1]); // [S_v|1,1,H]
    ggml_tensor * AendB = ggml_cont(c, Aend);
    ggml_tensor * S0dec = ggml_mul(c, S0, AendB);               // [i,j,H] * A_end[i](ne0,bcast over j)
    ggml_tensor * Kw = ggml_mul(c, kp, ggml_mul(c, Ainv, AendB)); // (A_end/A_r) k_r  [S_v(i),N(r),H]
    // Kw^T @ Dmat: sum_r Kw[i,r] Dmat[r,j]. mul_mat contracts ne0 -> need Kw[r,i] and Dmat[r,j]
    ggml_tensor * KwT = ggml_cont(c, ggml_transpose(c, Kw));    // [N(r),S_v(i),H]
    ggml_tensor * upd = ggml_mul_mat(c, KwT, Dmat);            // contract r -> [i,j,H]
    *out_state = ggml_add(c, S0dec, upd);                      // [i,j,H]
}

// Multi-chunk tiling: split the N tokens into blocks of C, run build_chunked per block carrying the
// recurrent state forward. Bounds Ainv=exp(-cumsum(g)) to <=C tokens -> numerically stable for long
// prefill (single-chunk overflows). ceil(N/C) chunks, unrolled at graph-build time (N is static).
static void build_chunked_tiled(ggml_context * c, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                ggml_tensor * g, ggml_tensor * beta, ggml_tensor * S0, int64_t C,
                                ggml_tensor ** out_attn, ggml_tensor ** out_state) {
    const int64_t N = v->ne[2];
    ggml_tensor * S = S0;
    ggml_tensor * attn_full = nullptr;
    for (int64_t start = 0; start < N; start += C) {
        const int64_t cn = std::min<int64_t>(C, N - start);
        auto slice = [&](ggml_tensor * x){
            return ggml_view_4d(c, x, x->ne[0], x->ne[1], cn, 1, x->nb[1], x->nb[2], x->nb[3], start*x->nb[2]);
        };
        ggml_tensor *ac=nullptr,*sc=nullptr;
        build_chunked(c, slice(q), slice(k), slice(v), slice(g), slice(beta), S, &ac, &sc);
        attn_full = attn_full ? ggml_concat(c, attn_full, ac, 0) : ac; // O is [token, S_v, H]; concat tokens on ne0
        S = ggml_reshape_4d(c, sc, S0->ne[0], S0->ne[0], S0->ne[2], 1); // match model: thread 4D state
    }
    *out_attn = attn_full;
    *out_state = S;
}

static int run_case(int64_t S_v, int64_t H, int64_t N, std::mt19937 & rng, int64_t Hk=-1, bool scalar_gate=false, int64_t chunk=0){
    if (Hk < 0) Hk = H;
    g_pending.clear();
    size_t mem = 64ull*1024*1024; // metadata only; tensor data lives in the backend buffer (no_alloc)
    ggml_init_params ip{mem, nullptr, true};
    ggml_context * c = ggml_init(ip);
    ggml_tensor * q = rnd(c,S_v,Hk,N,1,rng,0.5f);
    ggml_tensor * k = rnd(c,S_v,Hk,N,1,rng,0.5f);
    l2norm_rows(q); l2norm_rows(k); // delta-net normalizes q,k -> stable recurrence
    ggml_tensor * v = rnd(c,S_v,H,N,1,rng,0.5f);
    // gate: vector (KDA, [S_v,H,N]) or per-head scalar (Gated DeltaNet, [1,H,N])
    ggml_tensor * g = rnd(c, scalar_gate ? 1 : S_v, H, N, 1, rng, 0.1f, -0.2f); // log-decay <0
    g = ggml_neg(c, ggml_abs(c, g)); // ensure <=0 -> a<=1
    ggml_tensor * beta = rnd(c,1,H,N,1,rng,0.0f,0.5f);
    ggml_tensor * S0 = rnd(c,S_v,S_v,H,1,rng,0.3f);

    ggml_tensor * ref = ggml_gated_delta_net(c, q,k,v,g,beta,S0); // [S_v*H, N+S_v]
    ggml_tensor *ca=nullptr,*cs=nullptr;
    if (chunk > 0) build_chunked_tiled(c,q,k,v,g,beta,S0,chunk,&ca,&cs);
    else           build_chunked(c,q,k,v,g,beta,S0,&ca,&cs);
    // replicate the model's EXACT output op: permute concatenated O [t,j,H] -> [S_v,H,N,1] + cont
    ca = ggml_reshape_4d(c, ggml_cont(c, ggml_permute(c, ca, 2,0,1,3)), S_v, H, N, 1);
    cs = ggml_reshape_4d(c, cs, S_v, S_v, H, 1);

    ggml_cgraph * gf = ggml_new_graph_custom(c, 8192, false);
    ggml_build_forward_expand(gf, ref);
    ggml_build_forward_expand(gf, ca);
    ggml_build_forward_expand(gf, cs);
    ggml_backend_t be = make_backend();
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(galloc, gf);
    for (auto & pd : g_pending) ggml_backend_tensor_set(pd.t, pd.data.data(), 0, pd.data.size()*sizeof(float));
    ggml_backend_graph_compute(be, gf);

    // read outputs back to host (works for CPU and CUDA)
    std::vector<float> hr(ggml_nelements(ref)), hca(ggml_nelements(ca)), hcs(ggml_nelements(cs));
    ggml_backend_tensor_get(ref, hr.data(), 0, hr.size()*sizeof(float));
    ggml_backend_tensor_get(ca,  hca.data(),0, hca.size()*sizeof(float));
    ggml_backend_tensor_get(cs,  hcs.data(),0, hcs.size()*sizeof(float));
    const int64_t ca_n0=ca->ne[0], ca_n1=ca->ne[1], cs_n0=cs->ne[0], cs_n1=cs->ne[1];
    // reference attn: cols 0..N-1 of [S_v*H, N+S_v]; per (head h, token t): ref[ h*S_v + j , t ]
    auto refAttn = [&](int h,int t,int j){ return hr[(int64_t)t*(S_v*H) + h*S_v + j]; };
    auto refState= [&](int h,int i,int j){ return hr[(int64_t)S_v*H*N + (int64_t)h*S_v*S_v + (int64_t)j*S_v + i]; };
    // chunked attn O[r,j,H]; state [i,j,H]
    // ca is now model layout [S_v,H,N,1]: element[j,h,t] = j + h*S_v + t*S_v*H
    (void)ca_n0;(void)ca_n1;
    auto caV=[&](int h,int t,int j){ return hca[ (int64_t)t*S_v*H + (int64_t)h*S_v + j ]; };
    auto csV=[&](int h,int i,int j){ return hcs[ (int64_t)h*cs_n0*cs_n1 + (int64_t)j*cs_n0 + i ]; };
    double mA=0, mS=0;
    for(int h=0;h<H;h++)for(int t=0;t<N;t++)for(int j=0;j<S_v;j++) mA=std::max(mA,(double)fabs(refAttn(h,t,j)-caV(h,t,j)));
    for(int h=0;h<H;h++)for(int i=0;i<S_v;i++)for(int j=0;j<S_v;j++) mS=std::max(mS,(double)fabs(refState(h,i,j)-csV(h,i,j)));
    int ok = (mA<1e-3 && mS<1e-3);
    char ch[16]; if (chunk>0) snprintf(ch,sizeof ch,"C=%lld",(long long)chunk); else snprintf(ch,sizeof ch,"single");
    printf("S_v=%lld H=%lld N=%lld Hk=%lld gate=%-6s %-7s: max|dAttn|=%.2e max|dState|=%.2e -> %s\n",
           (long long)S_v,(long long)H,(long long)N,(long long)Hk, scalar_gate?"scalar":"vector", ch, mA, mS, ok?"PASS":"FAIL");
    ggml_gallocr_free(galloc);
    ggml_backend_free(be);
    ggml_free(c);
    return ok;
}

int main(){
    std::mt19937 rng(0);
    int all = 1;
    { ggml_backend_t b = make_backend(); printf("backend: %s\n", ggml_backend_name(b)); ggml_backend_free(b); }
    printf("== vector (KDA) gate ==\n");
    for (int64_t S_v : {64, 128}) for (int64_t N : {1, 2, 5, 8, 12, 16}) all &= run_case(S_v, 4, N, rng);
    // GQA: H_v=4, H_k=2 and H_k=1 (q/k broadcast interleaved)
    for (int64_t N : {1, 5, 16}) { all &= run_case(64, 4, N, rng, 2); all &= run_case(128, 4, N, rng, 1); }
    printf("== scalar (Gated DeltaNet) gate -- Qwen3.5 ==\n");
    for (int64_t S_v : {64, 128}) for (int64_t N : {1, 2, 5, 8, 12, 16}) all &= run_case(S_v, 4, N, rng, -1, true);
    // GQA + scalar gate (Qwen3.5 is GQA: H_v=32, H_k=16 -> ratio 2)
    for (int64_t N : {1, 5, 16}) { all &= run_case(64, 4, N, rng, 2, true); all &= run_case(128, 4, N, rng, 1, true); }
    printf("== multi-chunk tiling (long sequences; single-chunk overflows) ==\n");
    // N far beyond a verify block; C=8 chunks carry state. C must stay small: the deflation
    // A=exp(+/-cumsum(g)) has wide dynamic range, so strong-decay heads lose fp32 precision when a
    // chunk is too long (the model garbles at C>=16 on Qwen3.5; the deployed default is C=8).
    for (int64_t N : {32, 64, 128, 200}) {
        all &= run_case(128, 4, N, rng, -1, false, 8); // vector
        all &= run_case(128, 4, N, rng, -1, true,  8); // scalar (Qwen3.5)
    }
    all &= run_case(128, 4, 128, rng, 1, true, 8);  // GQA + scalar + tiled
    all &= run_case(64,  4, 96,  rng, 2, false, 8); // GQA + vector + tiled
    // sanity: tiling with C>=N must equal the single-chunk path
    all &= run_case(128, 4, 12, rng, -1, true, 64);
    printf("%s\n", all ? "ALL PASS" : "SOME FAIL");
    return all ? 0 : 1;
}
