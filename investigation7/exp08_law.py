"""
exp08: pinning down the collapse law, and testing the mechanism.

exp07's `var_ratio * d` column is very nearly a function of K alone, and the values it takes --
3.912 at K = 50, 4.605 at K = 100, 5.98 at K = 400 -- are ln 50 = 3.912, ln 100 = 4.605,
ln 400 = 5.991, to three decimals. That is not a coincidence waiting to happen: the
median-heuristic bandwidth in `MSVGD.pairwise_distance` is h = median(||x-y||^2) / ln K, so ln K
is already in the algorithm.

  Conjecture A (the law):        Var_SVGD / Var_target  =  ln(K) / d        for K <~ d.

  Conjecture B (the mechanism):  A is a restatement of h -> 2. For an ensemble with per-coordinate
    variance v in d dimensions, median ||x - y||^2 ~= 2 d v, so the median heuristic gives
    h = 2 d v / ln K; substituting A gives h = 2 exactly, independent of d and K. That is, the
    ensemble contracts until the adaptive bandwidth equals twice the target's own variance scale,
    and the median heuristic then RE-TIGHTENS the bandwidth around the contracted ensemble, which
    is the positive feedback.

Part 1 sweeps K over three orders of magnitude at fixed d to test A.
Part 2 tests B by holding the bandwidth FIXED at multiples of the value the median heuristic
would return AT THE TARGET, h* = 2d / ln K, which removes the feedback loop. If the collapse is
an artefact of the adaptive bandwidth, a fixed h* holds the variance; if it is intrinsic, it does
not, and the only remaining lever is K.
"""
import numpy as np, jax, jax.numpy as jnp, optax, os, json, time
jax.config.update("jax_enable_x64", True)
import msvgd7 as M7

DS = [int(x) for x in os.environ.get("DS", "50,100,325").split(",")]
KS = [int(x) for x in os.environ.get("KS", "10,20,50,100,200,400,800,1600,3200").split(",")]
MAXIT = int(os.environ.get("MAXIT", 3000))
KERNELS = os.environ.get("KERNELS", "standard,reweighted").split(",")
PART = os.environ.get("PART", "1,2,3").split(",")


class Iso:
    def __init__(self):
        self.mu = jnp.zeros((1,))
        self.data = None
        self.logdensity = lambda x, data: -0.5 * jnp.sum(x ** 2)
        self.gradient = jax.jit(jax.vmap(lambda x, data: -x, in_axes=(0, None)))


g = Iso()
res = {}


def h_emp(P):
    """The median-heuristic bandwidth the ensemble would produce, for reporting."""
    return float(M7._pairwise(jnp.asarray(P), -1.0)[1])


if "1" in PART:
    print("--- Part 1: the law, adaptive (median-heuristic) bandwidth", flush=True)
    print(f'{"kernel":>12} {"d":>5} {"K":>6} {"ln K":>7} {"var ratio":>10} {"x d":>8} '
          f'{"(x d)/lnK":>10} {"h_final":>9} {"sec":>6}', flush=True)
    for kern in KERNELS:
        for d in DS:
            for K in KS:
                rng = np.random.default_rng(0)
                X0 = rng.standard_normal((K, d))
                t0 = time.time()
                P, _, _ = M7.run_svgd(g, X0, MAXIT, kernel=kern,
                                      optimizer=optax.contrib.prodigy, optimizer_kwargs={})
                vr = float(np.mean(P.var(0)))
                print(f'{kern:>12} {d:>5} {K:>6} {np.log(K):>7.3f} {vr:>10.5f} {vr*d:>8.3f} '
                      f'{vr*d/np.log(K):>10.3f} {h_emp(P):>9.3f} {time.time()-t0:>6.1f}',
                      flush=True)
                res[f"law|{kern}|{d}|{K}"] = dict(var_ratio=vr, times_d=vr * d, h=h_emp(P))
                json.dump(res, open("exp08_results.json", "w"), indent=1)

if "2" in PART:
    print("\n--- Part 2: the mechanism. Bandwidth FIXED at mult * h*, h* = 2d/ln K "
          "(the median heuristic evaluated at the target)", flush=True)
    print(f'{"kernel":>12} {"d":>5} {"K":>6} {"mult":>7} {"h used":>10} {"var ratio":>10} '
          f'{"x d":>8} {"SteinR":>8} {"sec":>6}', flush=True)
    for kern in KERNELS:
        for d in (100, 325):
            for K in (400,):
                hstar = 2.0 * d / np.log(K)
                for mult in (0.1, 0.3, 1.0, 3.0, 10.0, 100.0):
                    rng = np.random.default_rng(0)
                    X0 = rng.standard_normal((K, d))
                    t0 = time.time()
                    P, _, _ = M7.run_svgd(g, X0, MAXIT, kernel=kern, bandwidth=mult * hstar,
                                          optimizer=optax.contrib.prodigy, optimizer_kwargs={})
                    vr = float(np.mean(P.var(0)))
                    R = float(-np.sum((P - P.mean(0)) * (-P)) / P.size)
                    print(f'{kern:>12} {d:>5} {K:>6} {mult:>7.1f} {mult*hstar:>10.2f} '
                          f'{vr:>10.5f} {vr*d:>8.3f} {R:>8.4f} {time.time()-t0:>6.1f}', flush=True)
                    res[f"bw|{kern}|{d}|{K}|{mult}"] = dict(var_ratio=vr, h=mult * hstar,
                                                            steinR=R)
                    json.dump(res, open("exp08_results.json", "w"), indent=1)

if "3" in PART:
    # ---------------------------------------------------------------------------------------
    # Part 3: is the collapse a fixed point, or just where Prodigy happens to stop?
    #
    # exp07's SGD-at-1e-2 rows are far less collapsed than its Prodigy rows (0.897 vs 0.109 at
    # d = 50, K = 400), which would be a serious confound if it meant the two discretisations had
    # different fixed points. They cannot: the SVGD flow is the same, and the optimizer only sets
    # how fast it is integrated. Two checks.
    #
    # 3a. THE OPTIMIZER-FREE ONE. At exact draws from N(0, I_d) the mean-field velocity vanishes
    #     identically, so any systematic component of the empirical field is finite-K bias -- the
    #     whole mechanism of the collapse, measured with no step size involved. Reported as
    #     d/dt log Var, averaged over R independent exact ensembles, with a standard error. A
    #     correct fixed point gives zero within that error.
    # 3b. SGD run out to 100x longer, to show it is heading to the same place.
    # ---------------------------------------------------------------------------------------
    print("\n--- Part 3a: SVGD velocity field AT exact draws from N(0, I_d), no optimizer",
          flush=True)
    print(f'{"kernel":>12} {"d":>5} {"K":>6} {"d/dt log Var":>16} {"+- se":>10} '
          f'{"d/dt mean^2":>13}', flush=True)
    Rrep = 12
    for kern in KERNELS:
        kfn = M7.KERNELS[kern]

        @jax.jit
        def field(P):
            return -kfn(P, P, -0.5 * jnp.sum(P ** 2, axis=1), -1.0)   # raw_grad = -s = +x

        for d in (10, 50, 100, 325):
            for K in (100, 400):
                rng = np.random.default_rng(0)
                rates, mdr = [], []
                for _ in range(Rrep):
                    X = rng.standard_normal((K, d))
                    phi = np.asarray(field(jnp.asarray(X)), np.float64)
                    rates.append(float(np.mean(2 * ((X - X.mean(0)) * phi).mean(0) / X.var(0))))
                    mdr.append(float(np.mean(2 * X.mean(0) * phi.mean(0))))
                mu, se = float(np.mean(rates)), float(np.std(rates, ddof=1) / np.sqrt(Rrep))
                print(f'{kern:>12} {d:>5} {K:>6} {mu:>16.3e} {se:>10.1e} '
                      f'{np.mean(mdr):>13.3e}', flush=True)
                res[f"field|{kern}|{d}|{K}"] = dict(rate=mu, se=se, mean_rate=float(np.mean(mdr)))
                json.dump(res, open("exp08_results.json", "w"), indent=1)

    print("\n--- Part 3b: SGD at 1e-2, run long. Target of the law: ln(K)/d", flush=True)
    print(f'{"kernel":>12} {"d":>5} {"K":>6} {"iters":>8} {"var ratio":>10} {"x d":>8} '
          f'{"ln K":>7} {"sec":>6}', flush=True)
    for kern in ("standard",):
        for d, K in ((50, 400), (100, 400)):
            rng = np.random.default_rng(0)
            X0 = rng.standard_normal((K, d))
            Y = X0
            tot = 0
            for nit in (2_000, 8_000, 40_000, 200_000):
                t0 = time.time()
                Y, _, _ = M7.run_svgd(g, Y, nit - tot, kernel=kern, optimizer=optax.sgd,
                                      optimizer_kwargs={"learning_rate": 1e-2})
                tot = nit
                vr = float(np.mean(Y.var(0)))
                print(f'{kern:>12} {d:>5} {K:>6} {nit:>8} {vr:>10.5f} {vr*d:>8.3f} '
                      f'{np.log(K):>7.3f} {time.time()-t0:>6.1f}', flush=True)
                res[f"sgdlong|{kern}|{d}|{K}|{nit}"] = dict(var_ratio=vr, times_d=vr * d)
                json.dump(res, open("exp08_results.json", "w"), indent=1)
