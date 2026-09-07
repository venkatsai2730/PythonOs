"""
Stage E: Hyper-Connections and manifold-constrained Hyper-Connections (mHC).

Standard transformers carry ONE residual stream: x = x + f(x). Hyper-
Connections carry `n` streams and learn how to (a) aggregate them into each
sublayer's input, (b) distribute the sublayer output back across them, and
(c) mix them with each other.

Per residual connection, with streams H (n x d):

    x_in  = alpha . H                      aggregate  (alpha: n)
    y     = sublayer(norm(x_in))
    H_out = A @ H + beta (x) y             combine    (A: n x n, beta: n)

Three modes, meant to be run side by side:

  'single'  the Stage A baseline. n = 1, no extra parameters, code path
            untouched. This is the control.
  'hc'      unconstrained A. The mixing matrix is a free n x n parameter.
  'mhc'     A is constrained to the doubly-stochastic manifold (the Birkhoff
            polytope) by Sinkhorn-Knopp projection in log space. Mixing
            becomes a convex, mass-preserving combination of streams, which is
            what stops the unconstrained version from drifting toward norm
            blow-up or a single dominant stream.

INITIALISATION, and why it matters here
---------------------------------------
alpha = beta = readout = e_0 and A = I. So at init, stream 0 carries exactly
the standard residual computation and streams 1..n-1 sit inert at their
initial value. That makes 'hc' **bit-identical to the dense baseline at
initialisation, for any n** — asserted in tests/test_stage_e.py. Any measured
difference is therefore something the model learned, not an artifact of
reparameterisation.

'mhc' cannot be exactly identity-initialised: it reaches A through a Sinkhorn
projection, and identity is a vertex of the Birkhoff polytope only in the
limit. Initialising the logits at `mhc_init_logit * I` puts A within ~e^-logit
of identity (~4.5e-5 at logit 10), so mhc starts *near* but not *at* the
baseline. The tests measure that deviation rather than assuming it away.

THE INITIALISATION TRILEMMA (measured, see STAGES.md)
-----------------------------------------------------
You cannot have all three of: exact baseline equivalence at init, equal
gradient across streams, and n distinguishable streams.

  'e0'      exact equivalence + symmetry broken between stream 0 and the rest
            — but streams 1..n-1 remain mutually interchangeable, so the model
            has only TWO distinct stream roles regardless of n. Measured
            per-stream gradient norms after 300 steps: [0.78, 0.093, 0.093,
            0.093] — the last three are identical to 5 significant figures.
  'uniform' exact equivalence + equal gradient — but PERFECTLY symmetric, so
            all streams stay identical forever. Measured cosine similarity
            after 300 steps: 0.976 (hc) / 0.999 (mhc). Guaranteed collapse;
            useful as a control that demonstrates the failure mode.
  'random'  n distinguishable streams + broken symmetry — but no longer exactly
            equivalent to the baseline at step 0.

This is structural, not a bug: if streams 1..n-1 are inert and identical at
init, they are interchangeable by construction. Any scheme that keeps exact
equivalence inherits that cap.

A NOTE ON THE STREAM-COLLAPSE RISK
----------------------------------
The design review flagged that published interpretability work tends to find
residual streams collapsing toward near-identity mixing with one dominant
stream, rather than n streams carrying n distinct reasoning signals. Nothing
in this module assumes either outcome. The instrumentation exists to measure
which one actually happens on our checkpoints:

  - per-stream gradient norm      is any stream receiving no learning signal?
  - pairwise cosine similarity    have the streams become copies of each other?
  - ||A - I||_F                   has mixing collapsed back to identity?

A result showing collapse is a finding, not a failed run.
"""

import torch
import torch.nn as nn


def sinkhorn_log(logits, iters):
    """Project a logit matrix onto the doubly-stochastic manifold.

    Sinkhorn-Knopp, done in log space: alternately normalise rows and columns
    until both sum to 1. Log space matters — the naive exp/divide form
    underflows for the strongly-diagonal initialisation used here
    (exp(-10) territory) and silently produces NaNs.

    Column normalisation runs last, so columns sum to exactly 1 and rows only
    approximately.

    CONVERGENCE IS NOT GUARANTEED AT A FIXED ITERATION COUNT, and this matters
    more than it looks. Measured worst-case row residual over 900 random
    skewed logit matrices (randn * 2, n in {2,4,8}):

        iters      8      25      50     100     200     400
        residual  1.1e-1  2.7e-2  8.9e-3  4.1e-3  1.4e-3  1.9e-4

    Convergence is linear, so brute force is not a fix: 100 iterations still
    leaves 4e-3 and costs ~3.3ms per call on CPU, which at 2 connections x
    n_layer calls per forward is prohibitive.

    The saving grace is that NEAR-IDENTITY matrices converge immediately —
    residual 6e-8 at just 8 iterations — and mHC is initialised near identity.
    So the default iteration count is cheap and exact where training starts.

    The risk is A drifting to a skewed regime mid-training, at which point A is
    silently NOT doubly stochastic and mHC is no longer manifold-constrained
    at all. That is why `row_sum_dev` is part of diagnostics() and is logged
    every mhc_log_interval steps: if it climbs, raise mhc_sinkhorn_iters.
    Do not assume the projection is holding — check it.
    """
    log_a = logits
    for _ in range(iters):
        log_a = log_a - torch.logsumexp(log_a, dim=1, keepdim=True)
        log_a = log_a - torch.logsumexp(log_a, dim=0, keepdim=True)
    return log_a.exp()


class HyperConnection(nn.Module):
    """One hyper-connection, replacing one residual add.

    Two of these per transformer block: one around attention, one around the
    feed-forward. `n_streams == 1` with mode 'single' is not routed here at
    all — Block keeps the plain residual path so the baseline is untouched.
    """

    def __init__(self, n_streams, mode='hc', sinkhorn_iters=20, init_logit=10.0,
                 init_scheme='e0'):
        super().__init__()
        assert mode in ('hc', 'mhc'), mode
        assert init_scheme in ('e0', 'uniform', 'random'), init_scheme
        assert n_streams >= 1
        self.n = n_streams
        self.mode = mode
        self.sinkhorn_iters = sinkhorn_iters
        self.init_scheme = init_scheme

        # BOTH schemes are exactly residual-equivalent at init. They differ in
        # how gradient is distributed across streams, and that difference
        # decides whether the streams can ever differentiate at all:
        #
        # 'e0'      alpha = beta = e_0. Stream 0 does the standard residual;
        #           streams 1..n-1 sit inert. ASYMMETRIC, so gradients differ
        #           per stream and symmetry can break. But measured per-stream
        #           gradient norms at init are ~[0.53, 1e-4, 1e-4, 1e-4]: the
        #           non-primary streams get ~5000x less signal, so
        #           differentiation may be very slow or never happen.
        #
        # 'uniform' alpha = 1/n, beta = 1. Every stream is treated identically
        #           and all receive equal gradient. But this is PERFECTLY
        #           SYMMETRIC: all streams start equal, receive identical
        #           gradients, and therefore remain identical forever. The
        #           symmetry is exact and cannot break on its own. This is a
        #           guaranteed-collapse control, useful precisely because it
        #           demonstrates the failure mode rather than risking it.
        #
        # Neither is obviously right. Run both — that is the point of Stage E.
        if init_scheme == 'e0':
            vec = torch.zeros(n_streams)
            vec[0] = 1.0
            self.alpha = nn.Parameter(vec.clone())
            self.beta = nn.Parameter(vec.clone())
        elif init_scheme == 'uniform':
            self.alpha = nn.Parameter(torch.full((n_streams,), 1.0 / n_streams))
            self.beta = nn.Parameter(torch.ones(n_streams))
        else:  # 'random'
            # See the trilemma note in the module docstring: exact baseline
            # equivalence at init forces streams 1..n-1 to be interchangeable,
            # which caps the model at 2 distinct stream roles no matter how
            # large n is. This scheme gives that up — it perturbs alpha/beta so
            # every stream is distinguishable from the start, at the cost of
            # no longer reproducing the baseline exactly at step 0.
            vec = torch.zeros(n_streams)
            vec[0] = 1.0
            gen = torch.Generator().manual_seed(1337)
            self.alpha = nn.Parameter(vec + 0.02 * torch.randn(n_streams, generator=gen))
            self.beta = nn.Parameter(vec + 0.02 * torch.randn(n_streams, generator=gen))

        eye = torch.eye(n_streams)
        if mode == 'hc':
            self.A_param = nn.Parameter(eye.clone())
        else:
            # strongly diagonal logits -> Sinkhorn output near identity
            self.A_param = nn.Parameter(init_logit * eye.clone())

    def mixing_matrix(self):
        if self.mode == 'hc':
            return self.A_param
        return sinkhorn_log(self.A_param, self.sinkhorn_iters)

    def aggregate(self, H):
        """(B, T, n, d) streams -> (B, T, d) sublayer input."""
        return torch.einsum('n,btnd->btd', self.alpha.to(H.dtype), H)

    def combine(self, H, y):
        """Mix streams and add the sublayer output back across them."""
        A = self.mixing_matrix().to(H.dtype)
        mixed = torch.einsum('on,btnd->btod', A, H)
        return mixed + self.beta.to(H.dtype).view(1, 1, -1, 1) * y.unsqueeze(2)

    @torch.no_grad()
    def diagnostics(self):
        """Mixing-matrix health. Read `dev_from_identity` for collapse."""
        A = self.mixing_matrix().float()
        n = self.n
        eye = torch.eye(n, device=A.device)
        off = A - torch.diag(torch.diagonal(A))
        return {
            'dev_from_identity': (A - eye).norm().item(),
            'diag_mean': torch.diagonal(A).mean().item(),
            'off_diag_absmean': off.abs().sum().item() / max(n * (n - 1), 1),
            'row_sum_dev': (A.sum(dim=1) - 1).abs().max().item(),
            'col_sum_dev': (A.sum(dim=0) - 1).abs().max().item(),
            'alpha': self.alpha.detach().float().tolist(),
            'beta': self.beta.detach().float().tolist(),
        }


@torch.no_grad()
def stream_cosine_similarity(H):
    """Mean pairwise cosine similarity between streams of (B, T, n, d).

    ~1.0 means the streams have become copies of each other, i.e. the
    n-streams-as-n-signals story has collapsed. Computed on detached values.
    """
    n = H.size(2)
    if n < 2:
        return float('nan')
    V = torch.nn.functional.normalize(H.float(), dim=-1)
    total, pairs = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += (V[:, :, i, :] * V[:, :, j, :]).sum(-1).mean().item()
            pairs += 1
    return total / pairs


@torch.no_grad()
def stream_norms(H):
    """RMS norm per stream — catches one stream dominating in magnitude."""
    return H.float().pow(2).mean(dim=(0, 1, 3)).sqrt().tolist()
