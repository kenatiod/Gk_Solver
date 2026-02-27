#!/usr/bin/env python3
# Gk_Solver.py version 1
"""
Lehmer-Størmer gap-k enumerator by PARI generalized Pell solver

Purpose
-------
Find the minimum gap k such that a π-complete product m·(m+k) exists for each
prime set P_ω = {p1,...,p_ω}.  Building the min_gap(ω) table across ω=9..20+
provides strong computational evidence that no π-complete products of ANY gap
exist for large ω — a key step toward proving n=633,555 gives the last such
product for gap=1.

Mathematical framework
----------------------
For gap=k, substitute x = 2m+k.  Since (x-k)(x+k) = x²-k² = 4m(m+k) = 2qy²
(where m(m+k) = qy²/2, q squarefree P-product), we get:

    x² - 2q·y² = k²                 (generalized Pell equation)
    m = (x - k) / 2

For k=1 this recovers x²-2qy²=1, the equation used by Nr_Solver. ✓

Unlike the standard Pell (one family per mask), x²-2qy²=k² can have multiple
fundamental solution families.  All fundamental solutions satisfy x₀ ≤ k·√ε₁
where ε₁ = x₁+y₁√(2q) is the standard Pell fundamental unit.  Each family is
iterated via the same Pell automorphism:

    x' = x₁·x + 2q·y₁·y
    y' = x₁·y + y₁·x

The number of families per mask is O(d(k²)) — divisors of k² in Z[√(2q)].

Completeness guarantee
----------------------
q-bound (provably complete up to max_m): from x²-2qy²=k² with y≥1,
x ≥ √(k²+2q), so m=(x-k)/2 ≥ (√(k²+2q)-k)/2.  Setting m > max_m gives
q > 2·max_m·(max_m+k).  Masks exceeding this bound are skipped with no loss.

L-bound: each family grows via the Pell automorphism at the same rate as Nr_Solver.
The primitive divisor theorem (Bilu-Hanrot-Voutier 2001) guarantees that any
P-smooth iterate occurs at position n ≤ pmax.  So L = max(3, pmax) iterates per
family suffices.

Operating modes
---------------
  --mode fixed_k   Enumerate all π-complete (m, m+k) pairs for a specific k
  --mode sweep_k   For each ω, find min_gap(ω) = smallest k with a π-complete hit

How to install / run (macOS example)
-------------------------------------
1) Install PARI/GP:
       brew install pari
2) Sweep to find min_gap(ω) for ω=9..17:
       python3 Gk_Solver.py --mode sweep_k --start_omega 9 --end_omega 17 \
         --max_k 200 --workers 10 --gp_path /opt/homebrew/bin/gp
3) Fixed-k enumeration for k=3, ω=9..17:
       python3 Gk_Solver.py --mode fixed_k --gap_k 3 \
         --start_omega 9 --end_omega 17 --workers 10

Use --version to print a machine-readable environment/version block.

Version 1 (Feb 27 2026)
------------------------
* Initial release: generalized Pell x²-2qy²=k² for gap=k > 1
* New GP functions: genpell_fund(D,k) finds all fundamental solutions to
  x²-Dy²=k²; genpell_iterates(D,k,...) generates the solution family
* Multiple-family iteration: worker_chunk iterates all fundamental families
  per mask (O(d(k²)) families vs. 1 family in Nr_Solver)
* Gap-k q-bound: q > 2·max_m·(max_m+k) (generalizes Nr_Solver's k=1 bound)
* Gap-k x-bound: x > 2·max_m+k (same structural role as in Nr_Solver)
* sweep_k mode: for each ω, increments k until a π-complete hit is found
* min_gap_table.csv / min_gap_table.json: summary of min_gap(ω) results
* miss_star tracking in full verifier (was lightweight-only in Nr_Solver)
* Full audit trail identical to Nr_Solver v17 with k-specific extensions

Based on Nr_Solver.py v17 by Ken Clements.

By Ken Clements, Feb 27 2026
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import platform

# Disable Python's safety limit on int() string parsing for trusted local computation.
try:
    sys.set_int_max_str_digits(0)
except Exception:
    pass

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from typing import Dict, List, Optional, Set, Tuple


program_name, program_version = "Gk_Solver", 1


# ----------------------------- switches (set by args) -----------------------------
DEBUG = False
ASSERTIONS = False
ENV: Dict[str, object] = {}


# ----------------------------- small utilities -----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_self_source(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def get_git_commit() -> Optional[str]:
    # Best effort: if this file is inside a git repo, return HEAD commit hash.
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        s = (r.stdout or "").strip()
        return s if s else None
    except Exception:
        return None

def get_gp_version(gp_path: str) -> Optional[str]:
    # Best effort: ask GP for version() in quiet mode.
    try:
        r = subprocess.run(
            [gp_path, "-q"],
            input="print(version());\nquit\n",
            capture_output=True,
            text=True,
            check=True,
        )
        # Use last non-empty line
        lines = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
        return lines[-1] if lines else None
    except Exception:
        return None

def env_block(gp_path: str, script_path: str, argv: List[str]) -> Dict[str, object]:
    src = read_self_source(script_path)
    return {
        "script_path": os.path.abspath(script_path),
        "script_sha256": sha256_bytes(src),
        "git_commit": get_git_commit(),
        "command_line": " ".join(argv),
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "gp_path": gp_path,
        "gp_version": get_gp_version(gp_path),
    }


def primes_first_n(n: int) -> List[int]:
    """Return the first n primes (sympy-free).

    For this project n is small (typically <= 30), so a simple incremental
    trial-division generator is more than fast enough and removes the SymPy
    dependency.
    """
    if n <= 0:
        return []
    primes: List[int] = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        r = int(math.isqrt(candidate))
        for p in primes:
            if p > r:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1 if candidate == 2 else 2  # 2, then odds only
    return primes

def load_int_list(path: str) -> List[int]:
    vals: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            vals.append(int(s))
    return sorted(set(vals))

def write_int_list(path: str, vals: List[int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for v in sorted(vals):
            f.write(f"{v}\n")

def is_P_smooth(x: int, primes: Tuple[int, ...]) -> bool:
    if x <= 0:
        return False
    for p in primes:
        while x % p == 0:
            x //= p
    return x == 1

def factor_merge(a: Dict[int, int], b: Dict[int, int]) -> Dict[int, int]:
    out = dict(a)
    for p, e in b.items():
        out[p] = out.get(p, 0) + e
    return out

def support_tuple(f: Dict[int, int]) -> Tuple[int, ...]:
    return tuple(sorted(f.keys()))

def format_factorization(f: Dict[int, int]) -> str:
    items = sorted(f.items())
    return " * ".join(f"{p}^{e}" for p, e in items) if items else "1"

def factor_over_P(n: int, primes: Tuple[int, ...]) -> Tuple[Dict[int, int], int]:
    """Factor n over the given prime basis.

    Returns (exponent_dict, remainder). If remainder != 1, then n has at least one
    prime factor not in 'primes'. This is used both for smoothness checks and for
    building the exact factorization of N_r over the target prime set.
    """
    rem = n
    out: Dict[int, int] = {}
    for p in primes:
        if rem % p == 0:
            e = 0
            while rem % p == 0:
                rem //= p
                e += 1
            out[p] = e
        if rem == 1:
            break
    return out, rem


# ----------------------------- PARI/GP interface (why and how) -----------------------------
# PARI/GP is a mature, high-performance system for computational number theory. The PARI library (libpari) provides
# fast big-integer arithmetic and specialized algorithms for primes, modular arithmetic, continued fractions, and
# Pell-type Diophantine problems; GP is PARI's small scripting language and REPL front-end. In this project we use
# PARI/GP as a "number theory coprocessor": Python handles orchestration (parallel mask scheduling, I/O, audit logs,
# verification), while GP performs the heavy inner-kernel arithmetic on very large integers.
#
# The integration is intentionally simple and auditable. We launch `gp -q` as a subprocess and keep it alive, then
# send GP source code over stdin to define a single entrypoint `pellxy(D)` (plus helpers). Each evaluation is wrapped
# in sentinel markers (__BEGIN__/__END__) so Python can parse the response unambiguously from stdout. A startup
# handshake self-test (e.g. pellxy(46) and verifying x^2 - D*y^2 = 1) ensures the GP side is correctly initialized.
#
# The GP code implements a Pell solver via continued fractions of sqrt(D). For each D=2q (q from a prime-subset mask),
# we compute the minimal integer solution (x1,y1) to x^2 - D*y^2 = 1 by iterating convergents until the identity
# holds. Python then iterates powers of the fundamental solution using the standard recurrence
#   (x',y') = (x1*x + D*y1*y,  x1*y + y1*x),
# which preserves x^2 - D*y^2 = 1 exactly. Each iterate yields a candidate m = (x-1)/2 (when x is odd), which is then
# filtered for P-smoothness (and later checked for π-completeness by exact factorization and prime-support equality).
# The separation is deliberate: GP supplies certified Pell arithmetic; Python enforces predicates and produces the
# reproducible audit trail (logs + hashes + environment/version metadata).

# ----------------------------- GP / Pell (generalized) -----------------------------
_GENPELLXY_DEF = r"""
/* Return minimal integer [x,y] solving x^2 - D*y^2 = 1  (D positive nonsquare)
   If max_x > 0 and the convergent exceeds max_x before finding the solution,
   return [0,0] as a sentinel (no solution within bound).
   Correctness: convergents p_n of sqrt(D) grow monotonically (p_{n+1} = a*p_n + p_{n-1}
   with a >= 1), so once p_n > max_x, the fundamental solution x >= p_n > max_x. */
pellxy_cf(D, max_x=0)={
  if(D<=0, error("D<=0"));
  if(issquare(D), error("square"));

  my(a0 = sqrtint(D));
  my(m=0, d=1, a=a0);
  my(p0=1, p1=a0);
  my(q0=0, q1=1);

  while(p1^2 - D*q1^2 != 1,
    m = d*a - m;
    d = (D - m^2)/d;
    a = (a0 + m)\d;

    my(p2 = a*p1 + p0);
    my(q2 = a*q1 + q0);
    p0=p1; p1=p2;
    q0=q1; q1=q2;

    if(max_x > 0 && p1 > max_x, return([0,0]));
  );
  [p1, q1];
};

/* Factor D = d*s^2 with d squarefree and return integer [x,y] for x^2 - D*y^2=1
   If max_x > 0, bail out early when the solution exceeds max_x. */
pellxy(D, max_x=0)={
  if(D<=0, error("D<=0"));
  if(issquare(D), error("square"));

  my(F = factor(D));
  my(P = F[,1], E = F[,2]);
  my(d = 1, s = 1);
  for(i=1, #P,
    my(e = E[i]);
    if(e%2, d *= P[i]);
    s *= P[i]^(e\2);
  );

  my(v = pellxy_cf(d, max_x));
  if(v == [0,0], return([0,0]));
  my(a = v[1], b = v[2]);

  if(s==1, return([a,b]));

  my(a1=a, b1=b);
  while(b % s,
    my(aa = a1*a + d*b1*b);
    my(bb = a1*b + b1*a);
    a=aa; b=bb;
    if(max_x > 0 && a > max_x, return([0,0]));
  );

  my(y = b/s);
  if(a^2 - D*y^2 != 1, error("internal check failed"));
  [a,y];
};

/* Find all fundamental solutions [x,y] to x^2 - D*y^2 = k^2 (D positive nonsquare, k>0).
   Fundamental solutions are those with x in [k, ceil(k*sqrt(eps1))] where
   eps1 = x1 + y1*sqrt(D) is the fundamental Pell unit for x^2-D*y^2=1.
   Each such solution generates an infinite family via the Pell automorphism.
   If max_x > 0, the search range is further capped at max_x.
   Returns a vector of [x,y] pairs (may be empty if no solutions exist in range).

   k=1 special case: x^2-D*y^2=1 has exactly ONE fundamental family.  Direct
   call to pellxy(D) avoids an astronomical linear search.

   k>1 algorithm: CRT stride search.  Any solution must satisfy x^2 ≡ k^2 (mod p)
   for each prime p|D, i.e. x ≡ ±k (mod p).  Using CRT this gives at most 2^ω
   residue classes mod D (ω = number of prime factors of D).  We then step through
   each class in strides of D — a dramatic speedup over a linear scan, since for
   large D with bounded max_x the stride (D) often exceeds the search range,
   producing at most 1 candidate per class (i.e., O(2^ω) total checks). */
genpell_fund(D, k, max_x=0)={
  if(D<=0 || issquare(D), return([]));
  /* k=1: single fundamental family = standard Pell solution */
  if(k==1,
    my(v1 = pellxy(D, max_x));
    if(v1 == [0,0], return([]));
    return([v1]);
  );

  /* Compute the bound on fundamental solution x-values.
     bound = ceil(k * sqrt(eps1)) where eps1 = x1 + y1*sqrt(D) is the Pell unit.
     Always cap at max_x if provided. */
  my(v1 = pellxy(D));
  if(v1 == [0,0], return([]));
  my(x1=v1[1], y1=v1[2]);
  my(eps = x1 + y1*sqrt(D));
  my(bound = ceil(k * sqrt(eps)));
  if(max_x > 0 && bound > max_x, bound = max_x);
  if(bound < k, return([]));

  /* Build all residue classes mod D satisfying x^2 ≡ k^2 (mod D) via CRT.
     Each prime p|D contributes x ≡ +k (mod p) or x ≡ -k (mod p). */
  my(F = factor(D));
  my(nf = matsize(F)[1]);
  /* res_list: vector of [residue, modulus] pairs, starting with x ≡ 0 (mod 1) */
  my(res_list = [[0, 1]]);
  for(i = 1, nf,
    my(p = F[1,i]);
    my(pos_k = lift(Mod(k, p)));
    my(neg_k = lift(Mod(-k, p)));
    /* Two choices: x ≡ +k or x ≡ -k (mod p).
       Collapse to one when they coincide (p=2 with odd k, or p|k). */
    my(choices = if(pos_k == neg_k, [pos_k], [pos_k, neg_k]));
    my(new_list = []);
    for(j = 1, #res_list,
      my(r_prev = res_list[j][1], M_prev = res_list[j][2]);
      for(ci = 1, #choices,
        my(c = choices[ci]);
        /* CRT: combine x ≡ r_prev (mod M_prev) with x ≡ c (mod p) */
        my(new_r = lift(chinese(Mod(r_prev, M_prev), Mod(c, p))));
        new_list = concat(new_list, [[new_r, M_prev * p]]);
      );
    );
    res_list = new_list;
  );
  /* All residue classes now have modulus = D. */

  /* Also enforce parity: x ≡ k (mod 2) so that m = (x-k)/2 is an integer.
     Since D = 2q (always even for our application), 2 is already a factor of D
     and the parity constraint is already enforced by the CRT above. */

  my(sols = []);
  for(i = 1, #res_list,
    my(r = res_list[i][1], M = res_list[i][2]);
    /* Find first x >= k in this residue class (step up by M until x >= k) */
    my(x = r);
    while(x < k, x += M);
    while(x <= bound,
      my(rhs = x^2 - k^2);
      if(rhs % D == 0,
        my(y2 = rhs \ D);
        if(issquare(y2),
          sols = concat(sols, [[x, sqrtint(y2)]]);
        );
      );
      x += M;
    );
  );
  sols;
};

/* Return standard Pell fundamental solution [x1,y1] (for use as the automorphism).
   Wrapper so Python can retrieve x1,y1 in one GP call alongside genpell_fund. */
genpell_pell1(D, max_x=0)={
  pellxy(D, max_x);
};
"""

# Keep backward-compatible alias so _gp_start handshake still works
_PELLXY_DEF = _GENPELLXY_DEF

_BEGIN = "__BEGIN__"
_END = "__END__"
_VEC2_INT_RE = re.compile(r"^\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*$")

_GP_PROC: Optional[subprocess.Popen] = None
_GP_PATH: str = "gp"


def _gp_kill() -> None:
    global _GP_PROC
    if _GP_PROC is None:
        return
    try:
        _GP_PROC.kill()
    except Exception:
        pass
    _GP_PROC = None


def _gp_start() -> subprocess.Popen:
    p = subprocess.Popen(
        [_GP_PATH, "-q"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert p.stdin and p.stdout

    # load definitions
    p.stdin.write(_PELLXY_DEF + "\n")

    # handshake self-test: standard Pell + bailout
    p.stdin.write(f'print("{_BEGIN}");\n')
    p.stdin.write("v=pellxy(46); print(v); print(v[1]^2-46*v[2]^2);\n")
    # Bailout self-test: pellxy(46) gives x=24335, so max_x=100 should bail out
    p.stdin.write("vb=pellxy(46, 100); print(vb);\n")
    # genpell_fund test: x^2 - 14*y^2 = 4 (D=14, k=2)
    # Solutions: x=4,y=1 (16-14=2≠4), x=6,y=2 (36-56<0), try D=6,k=2:
    # x^2-6y^2=4: x=4,y=2 (16-24<0), x=2,y=0 (4-0=4 yes), x=10,y=4 (100-96=4 yes)
    # eps1 for D=6: pellxy(6)=[5,2], eps=5+2*sqrt(6)~9.9, bound=ceil(2*sqrt(9.9))=7
    # so search x=2..7 with x-2 even: x=2,4,6
    # x=2: rhs=0, y=0 -> [2,0]; x=4: rhs=12, 12/6=2 not square; x=6: rhs=32, 32/6 not int
    # So genpell_fund(6,2) should return [[2,0]]
    # Non-trivial test: D=2, k=7: x^2-2y^2=49: x=7,y=0; x=9,y=2 (81-8=73≠49);
    # eps1=[3,2] for D=2, eps=3+2*sqrt(2)~5.83, bound=ceil(7*sqrt(5.83))=17
    # x=7(parity ok):rhs=0->y=0; x=9:rhs=32,/2=16,sq->y=4 [9,4]; x=11:rhs=72,/2=36->y=6 [11,6]
    # x=13:rhs=120,/2=60 not sq; x=15:rhs=176,/2=88 not sq; x=17:rhs=240,/2=120 not sq
    # Verify [9,4]: 81-2*16=81-32=49 ✓; [11,6]: 121-2*36=121-72=49 ✓
    p.stdin.write("gf=genpell_fund(2,7); print(gf); print(length(gf));\n")
    p.stdin.write(f'print("{_END}");\n')
    p.stdin.flush()

    lines: List[str] = []
    in_block = False
    while True:
        line = p.stdout.readline()
        if line == "":
            raise RuntimeError("gp handshake EOF")
        s = line.rstrip("\n").strip()
        if s == _BEGIN:
            in_block = True
            continue
        if s == _END:
            break
        if in_block:
            lines.append(s)

    # lines: [pellxy(46) result, "1", "[0, 0]", genpell_fund result, length]
    if len(lines) < 5:
        raise RuntimeError(f"gp handshake unexpected output: {lines!r}")
    if lines[1] != "1":
        raise RuntimeError(f"gp handshake failed Pell certification: {lines!r}")
    if lines[2] != "[0, 0]":
        raise RuntimeError(f"gp handshake bailout test failed (expected [0, 0]): {lines!r}")
    if _VEC2_INT_RE.match(lines[0]) is None:
        raise RuntimeError(f"gp handshake did not return integer vector: {lines!r}")
    # genpell_fund(2,7) should find at least 2 fundamental solutions ([7,0],[9,4],[11,6])
    try:
        nfund = int(lines[4])
    except ValueError:
        nfund = -1
    if nfund < 2:
        raise RuntimeError(f"gp genpell_fund handshake unexpected result (expected >=2 solutions): {lines[3:5]!r}")

    return p


def _gp_eval(expr: str, timeout_sec: float = 0.0) -> str:
    global _GP_PROC
    if _GP_PROC is None:
        _GP_PROC = _gp_start()
    assert _GP_PROC.stdin and _GP_PROC.stdout

    _GP_PROC.stdin.write(f'print("{_BEGIN}");\n')
    _GP_PROC.stdin.write(f"print({expr});\n")
    _GP_PROC.stdin.write(f'print("{_END}");\n')
    _GP_PROC.stdin.flush()

    lines: List[str] = []
    in_block = False
    t0 = time.time()
    while True:
        if timeout_sec and (time.time() - t0) > timeout_sec:
            raise RuntimeError("gp eval timeout")
        line = _GP_PROC.stdout.readline()
        if line == "":
            raise RuntimeError("gp EOF (process died)")
        s = line.rstrip("\n").strip()
        if s == _BEGIN:
            in_block = True
            continue
        if s == _END:
            break
        if in_block:
            lines.append(s)

    return "\n".join(lines).strip()


def _pell_xy_gp(D: int, retries: int = 2, gp_timeout: float = 0.0,
                max_x: int = 0) -> Tuple[int, int, str]:
    """Solve x^2 - D*y^2 = 1 via GP continued fractions.

    If max_x > 0 and the fundamental solution has x > max_x, returns (0, 0, raw)
    as a sentinel indicating the solution exceeds the bound. This avoids computing
    enormous convergents inside GP.
    """
    last_exc: Optional[Exception] = None
    last_out: str = ""
    for _ in range(retries + 1):
        try:
            expr = f"pellxy({D}, {max_x})" if max_x > 0 else f"pellxy({D})"
            out = _gp_eval(expr, timeout_sec=gp_timeout)
            last_out = out

            # reject error banners
            if "***" in out or "error" in out.lower():
                raise RuntimeError(f"gp error banner: {out[:200]!r}")

            m = _VEC2_INT_RE.match(out)
            if not m:
                raise RuntimeError(f"Unexpected gp output format: {out!r}")

            x = int(m.group(1))
            y = int(m.group(2))

            # Handle [0,0] sentinel: solution exceeded max_x bound
            if x == 0 and y == 0:
                return 0, 0, out

            if x * x - D * y * y != 1:
                raise RuntimeError(f"Bad Pell solution: D={D} (x,y)=({x},{y})")

            return x, y, out
        except Exception as e:
            last_exc = e
            _gp_kill()
            continue
    raise RuntimeError(f"gp pellxy failed for D={D}: {last_exc!r}; last_out={last_out[:200]!r}")


# Regex for a GP vector of vectors: [[x1,y1],[x2,y2],...] or []
_VECVEC_RE = re.compile(r"^\[.*\]\s*$", re.DOTALL)

def _genpell_fund_gp(D: int, k: int, retries: int = 2, gp_timeout: float = 0.0,
                     max_x: int = 0) -> List[Tuple[int, int]]:
    """Find all fundamental solutions [x,y] to x^2 - D*y^2 = k^2 via GP.

    Returns a list of (x, y) tuples.  Each generates an infinite family via the
    Pell automorphism for x^2-D*y^2=1.  Returns [] if D is a perfect square or
    no fundamental solutions exist within the bound.
    """
    last_exc: Optional[Exception] = None
    last_out: str = ""
    for _ in range(retries + 1):
        try:
            if max_x > 0:
                expr = f"genpell_fund({D},{k},{max_x})"
            else:
                expr = f"genpell_fund({D},{k})"
            out = _gp_eval(expr, timeout_sec=gp_timeout)
            last_out = out

            if "***" in out or "error" in out.lower():
                raise RuntimeError(f"gp error banner: {out[:200]!r}")

            # Parse GP vector output: [] or [[x1,y1],[x2,y2],...]
            out_s = out.strip()
            if out_s == "[]":
                return []

            # Extract all [x,y] pairs from the nested vector
            pairs = _VEC2_INT_RE.findall(out_s)  # won't work for nested; use findall
            # Use a different approach: find all integer pairs
            pair_matches = re.findall(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", out_s)
            result: List[Tuple[int, int]] = []
            for (xs, ys) in pair_matches:
                x, y = int(xs), int(ys)
                if ASSERTIONS:
                    assert x * x - D * y * y == k * k, \
                        f"Bad genpell solution: D={D} k={k} x={x} y={y}: x²-Dy²={x*x-D*y*y} ≠ {k*k}"
                result.append((x, y))
            return result
        except Exception as e:
            last_exc = e
            _gp_kill()
            continue
    raise RuntimeError(f"gp genpell_fund failed D={D} k={k}: {last_exc!r}; last_out={last_out[:200]!r}")


# A "mask chunk" is a contiguous block (or small batch) of prime-subset bitmasks that a worker processes as a unit.
# Each bitmask encodes a squarefree subset of the current prime set P = (p1,...,p_omega): bit i is 1 iff p_i is
# included, and the mask maps to q = ∏ p_i. For each q we form the Pell parameter D = 2q and compute the minimal
# solution (x1,y1) to x^2 − D y^2 = 1; iterating that fundamental unit yields candidate m = (x−1)/2 values, which
# are then filtered for P-smoothness (and later π-completeness). Chunking masks amortizes gp startup/IPC overhead,
# enables dynamic work distribution across CPU cores, and provides a natural unit for retry/requeue when a small
# subset of masks is unusually expensive ("stragglers") or temporarily fails.

# ----------------------------- mask plumbing -----------------------------
def q_from_mask(mask: int, primes: Tuple[int, ...]) -> int:
    q = 1
    i = 0
    while mask:
        if mask & 1:
            q *= primes[i]
        mask >>= 1
        i += 1
    return q


@dataclass(frozen=True)
class WorkerParams:
    primes: Tuple[int, ...]
    L: int
    r: int
    k: int          # gap size: find pairs (m, m+k) both P-smooth
    nze_enabled: bool
    start_mask: int
    end_mask: int
    striped: bool
    step: int
    gp_timeout: float
    max_m: int  # Upper bound on m; 0 = no bound (full enumeration)


@dataclass(frozen=True)
class ChunkTask:
    a: int = 0
    b: int = 0
    offset: int = 0
    masks: Optional[Tuple[int, ...]] = None


@dataclass
class ChunkResult:
    ok: bool
    task: ChunkTask
    ms: List[int]
    tried: int
    failed: int
    failed_masks: List[int]
    elapsed_sec: float
    err_samples: List[str]
    masks_skipped_q_bound: int = 0
    masks_skipped_x1_bound: int = 0     # fundamental solution search above x_ceiling
    masks_skipped_gp_bailout: int = 0   # pellxy returned [0,0] sentinel
    iterates_early_break: int = 0
    candidates_skipped_nze: int = 0
    total_families: int = 0             # total fundamental solution families found
    masks_no_fund_sols: int = 0         # masks where genpell_fund returned []
    # Debug-only NZE pruning diagnostics
    nze_missing_prime_hist: Optional[Dict[int, int]] = None
    nze_min_missing_count: Optional[int] = None
    nze_min_missing_m: Optional[int] = None


def _init_worker(gp_path: str, debug: bool, assertions: bool) -> None:
    # Ensure the int() parsing limit is disabled in worker processes too.
    try:
        sys.set_int_max_str_digits(0)
    except Exception:
        pass
    import sys
    try:
        sys.set_int_max_str_digits(0)
    except AttributeError:
        pass
    global _GP_PATH, DEBUG, ASSERTIONS
    _GP_PATH = gp_path
    DEBUG = debug
    ASSERTIONS = assertions


def worker_chunk(args: Tuple[ChunkTask, WorkerParams]) -> ChunkResult:
    task, params = args
    primes = params.primes
    L = params.L
    r = params.r
    k = params.k
    nze_enabled = params.nze_enabled
    max_m = params.max_m

    # Gap-k q-bound: from x^2-2qy^2=k^2 with y>=1, x >= sqrt(k^2+2q),
    # so m=(x-k)/2 >= (sqrt(k^2+2q)-k)/2.  Setting m > max_m gives
    # q > 2*max_m*(max_m+k).  (For k=1 this recovers Nr_Solver's bound.)
    q_ceiling = 2 * max_m * (max_m + k) if max_m > 0 else 0
    # x-bound: m=(x-k)/2 <= max_m  iff  x <= 2*max_m+k
    x_ceiling = 2 * max_m + k if max_m > 0 else 0

    t0 = time.time()
    out: Set[int] = set()
    tried = 0
    failed = 0
    failed_masks: List[int] = []
    err_samples: List[str] = []
    masks_skipped_q_bound = 0
    masks_skipped_x1_bound = 0
    masks_skipped_gp_bailout = 0
    iterates_early_break = 0
    candidates_skipped_nze = 0
    total_families = 0
    masks_no_fund_sols = 0

    # NZE pruning diagnostics (only populated when DEBUG and NZE enabled)
    nze_hist: Optional[Dict[int, int]] = None
    nze_min_missing: Optional[int] = None
    nze_min_m: Optional[int] = None
    if DEBUG and nze_enabled:
        nze_hist = {}

    def emit_candidate(m: int) -> None:
        """Filter one candidate m for gap-k P-smoothness and optional NZE pruning.

        For gap-k: the product is m*(m+k), so we check both m and m+k.
        """
        nonlocal candidates_skipped_nze, nze_min_missing, nze_min_m
        # Require P-smoothness for both m and m+k (the gap-k pair)
        if not (is_P_smooth(m, primes) and is_P_smooth(m + k, primes)):
            return
        if not nze_enabled:
            out.add(m)
        else:
            # NZE pruning: only emit if the product m*(m+k) covers all primes in P
            f0, rem0 = factor_over_P(m, primes)
            if rem0 != 1:
                candidates_skipped_nze += 1
                return
            f1, rem1 = factor_over_P(m + k, primes)
            if rem1 != 1:
                candidates_skipped_nze += 1
                return
            fNr = factor_merge(f0, f1)
            if support_tuple(fNr) == primes:
                out.add(m)
            else:
                candidates_skipped_nze += 1
                if DEBUG and nze_hist is not None:
                    supp = set(support_tuple(fNr))
                    missing = [p for p in primes if p not in supp]
                    mc = len(missing)
                    if nze_min_missing is None or mc < nze_min_missing or (
                            mc == nze_min_missing and (nze_min_m is None or m < nze_min_m)):
                        nze_min_missing = mc
                        nze_min_m = m
                    for p in missing:
                        nze_hist[p] = nze_hist.get(p, 0) + 1

    def iterate_family(x0: int, y0: int, x1: int, y1: int, D: int) -> None:
        """Generate L iterates of the Pell family starting at (x0,y0) and emit candidates."""
        nonlocal iterates_early_break
        x, y = x0, y0
        for _ in range(L):
            if x_ceiling > 0 and x > x_ceiling:
                iterates_early_break += 1
                break
            # m = (x - k) / 2 must be a non-negative integer
            if (x - k) >= 0 and ((x - k) & 1) == 0:
                m = (x - k) // 2
                if m > 0:
                    emit_candidate(m)
            # Pell automorphism: (x,y) -> (x1*x + D*y1*y, x1*y + y1*x)
            nx = x1 * x + D * y1 * y
            ny = x1 * y + y1 * x
            x, y = nx, ny
            if ASSERTIONS:
                assert x * x - D * y * y == k * k

    def handle_mask(mask: int) -> None:
        nonlocal tried, failed, masks_skipped_q_bound, masks_skipped_x1_bound
        nonlocal masks_skipped_gp_bailout, total_families, masks_no_fund_sols
        q = q_from_mask(mask, primes)
        if q == 2:
            return

        # --- q-bound pre-filter ---
        # For gap-k: any solution has m >= (sqrt(k^2+2q)-k)/2.
        # If q > 2*max_m*(max_m+k) then m > max_m for ALL iterates — skip.
        if q_ceiling > 0 and q > q_ceiling:
            masks_skipped_q_bound += 1
            return

        D = 2 * q
        tried += 1
        try:
            # Step 1: find all fundamental solutions to x^2-D*y^2=k^2
            fund_sols = _genpell_fund_gp(D, k, retries=2,
                                         gp_timeout=params.gp_timeout,
                                         max_x=x_ceiling)
            if not fund_sols:
                masks_no_fund_sols += 1
                return

            # Step 2: get the standard Pell automorphism (x1,y1) for x^2-D*y^2=1
            x1, y1, _raw = _pell_xy_gp(D, retries=2,
                                        gp_timeout=params.gp_timeout,
                                        max_x=0)   # no bound needed here
            if x1 == 0 and y1 == 0:
                # D is a perfect square — no Pell solutions, skip
                masks_skipped_gp_bailout += 1
                return

            if ASSERTIONS:
                assert x1 * x1 - D * y1 * y1 == 1

            # --- fundamental solution gate ---
            # If all fundamental solutions exceed x_ceiling, no family can
            # produce m <= max_m (since iterates only grow).
            if x_ceiling > 0 and all(x0 > x_ceiling for (x0, _) in fund_sols):
                masks_skipped_x1_bound += 1
                return

            # Step 3: iterate each family
            total_families += len(fund_sols)
            for (x0, y0) in fund_sols:
                if x_ceiling > 0 and x0 > x_ceiling:
                    continue  # this family starts above the bound
                iterate_family(x0, y0, x1, y1, D)

        except Exception as e:
            failed += 1
            failed_masks.append(mask)
            if len(err_samples) < 10:
                if DEBUG:
                    err_samples.append(f"mask={mask} q={q} D={D} k={k} err={repr(e)}")
                else:
                    err_samples.append(repr(e))

    if task.masks is not None:
        for mask in task.masks:
            handle_mask(mask)
    elif params.striped:
        mask = params.start_mask + task.offset
        while mask < params.end_mask:
            handle_mask(mask)
            mask += params.step
    else:
        for mask in range(task.a, task.b):
            handle_mask(mask)

    elapsed = time.time() - t0
    ok = True
    if tried > 0 and failed == tried:
        ok = False

    return ChunkResult(
        ok=ok,
        task=task,
        ms=sorted(out),
        tried=tried,
        failed=failed,
        failed_masks=failed_masks,
        elapsed_sec=elapsed,
        err_samples=err_samples,
        masks_skipped_q_bound=masks_skipped_q_bound,
        masks_skipped_x1_bound=masks_skipped_x1_bound,
        masks_skipped_gp_bailout=masks_skipped_gp_bailout,
        iterates_early_break=iterates_early_break,
        candidates_skipped_nze=candidates_skipped_nze,
        total_families=total_families,
        masks_no_fund_sols=masks_no_fund_sols,
        nze_missing_prime_hist=nze_hist,
        nze_min_missing_count=nze_min_missing,
        nze_min_missing_m=nze_min_m,
    )


def filter_for_gk(ms: List[int], k: int, primes: Tuple[int, ...]) -> List[int]:
    """Filter candidate list: keep only m where both m and m+k are P-smooth.

    For k=1 this is the standard consecutive-pair filter. The worker already
    applies this check via emit_candidate, so this is a final safety pass.
    """
    out: List[int] = []
    for m in ms:
        if is_P_smooth(m, primes) and is_P_smooth(m + k, primes):
            out.append(m)
    return out


def compute_S_pmax_exact(
    primes: List[int],
    workers: int,
    r: int,
    k: int,
    nze_pruning: int,
    chunk_masks: int,
    striped: bool,
    incremental: bool,
    prior_S: Optional[List[int]],
    logf,
    gp_path: str,
    max_mask_attempts: int,
    mask_requeue_batch: int,
    gp_timeout: float,
    max_m: int = 0,
) -> Tuple[List[int], Dict[str, object], List[int]]:
    P = tuple(primes)
    omega = len(P)
    pmax = P[-1]
    L = max(3, pmax)
    nze_enabled = (nze_pruning > 0 and omega >= nze_pruning)

    total_masks = 1 << omega

    base_set: Set[int] = set()
    start_mask = 0
    end_mask = total_masks
    inc_used = False

    # NOTE: We do NOT restrict to the "new half" of masks at ω→ω+1.
    # Even if a new prime p_ω appears in m(m+1), it can occur with an even exponent
    # and therefore may be absent from the squarefree q used in D=2q (Pell). A classic
    # example is m=633555 at ω=8, where 19^2 divides m but 19 is absent from q.
    # Therefore, for correctness we always process the full mask range [0, 2^ω).
    # The prior_S list is still accepted for optional union/deduplication at output time.

    if incremental and prior_S is not None and omega >= 2:
        # When NZE pruning is enabled, we must not seed the current ω run with the
        # previous ω candidate list; those prior candidates may be non-π-complete
        # for the larger prime set and would contaminate counts (and debugging).
        if not nze_enabled:
            base_set = set(prior_S)
            inc_used = True

    tasks: List[ChunkTask] = []
    step = 1
    if striped:
        step = max(1, workers * 64)
        window = end_mask - start_mask
        step = min(step, window) if window > 0 else 1
        tasks = [ChunkTask(offset=o) for o in range(step)]
    else:
        for a in range(start_mask, end_mask, chunk_masks):
            b = min(end_mask, a + chunk_masks)
            tasks.append(ChunkTask(a=a, b=b))

    params = WorkerParams(
        primes=P,
        L=L,
        r=r,
        k=k,
        nze_enabled=nze_enabled,
        start_mask=start_mask,
        end_mask=end_mask,
        striped=striped,
        step=step,
        gp_timeout=gp_timeout,
        max_m=max_m,
    )

    ctx = get_context("fork") if sys.platform == "darwin" else get_context()

    found: Set[int] = set(base_set)

    stats: Dict[str, object] = {
        "omega": omega,
        "pmax": pmax,
        "lehmer_L": L,
        "k": k,
        "nze_pruning": nze_pruning,
        "nze_enabled": nze_enabled,
        "max_m": max_m,
        "incremental_used": inc_used,
        "start_mask": start_mask,
        "end_mask": end_mask,
        "striped": striped,
        "striped_step": step,
        "tasks_initial": len(tasks),
        "total_tried": 0,
        "total_failed": 0,
        "unique_failed_masks": 0,
        "unprocessed_masks": 0,
        "masks_skipped_q_bound": 0,
        "masks_skipped_x1_bound": 0,
        "masks_skipped_gp_bailout": 0,
        "iterates_early_break": 0,
        "candidates_skipped_nze": 0,
        "total_families": 0,
        "masks_no_fund_sols": 0,
        # NZE diagnostics (only populated when DEBUG and nze_enabled)
        "nze_missing_prime_hist": {},
        "nze_min_missing_count": None,
        "nze_min_missing_m": None,
        "err_samples": [],
    }

    mask_attempts: Dict[int, int] = {}
    unprocessed: List[int] = []

    queue: List[ChunkTask] = list(tasks)
    done = 0
    total = len(queue)

    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(gp_path, DEBUG, ASSERTIONS),
    ) as pool:
        while queue:
            batch = queue[:workers * 8]
            queue = queue[workers * 8:]

            async_args = [(t, params) for t in batch]
            for res in pool.imap_unordered(worker_chunk, async_args, chunksize=1):
                done += 1
                stats["total_tried"] = int(stats["total_tried"]) + res.tried
                stats["total_failed"] = int(stats["total_failed"]) + res.failed
                stats["masks_skipped_q_bound"] = int(stats["masks_skipped_q_bound"]) + res.masks_skipped_q_bound
                stats["masks_skipped_x1_bound"] = int(stats["masks_skipped_x1_bound"]) + res.masks_skipped_x1_bound
                stats["masks_skipped_gp_bailout"] = int(stats["masks_skipped_gp_bailout"]) + res.masks_skipped_gp_bailout
                stats["iterates_early_break"] = int(stats["iterates_early_break"]) + res.iterates_early_break
                stats["candidates_skipped_nze"] = int(stats["candidates_skipped_nze"]) + getattr(res, "candidates_skipped_nze", 0)
                stats["total_families"] = int(stats["total_families"]) + getattr(res, "total_families", 0)
                stats["masks_no_fund_sols"] = int(stats["masks_no_fund_sols"]) + getattr(res, "masks_no_fund_sols", 0)

                # Aggregate NZE diagnostics (DEBUG-only payloads)
                if getattr(res, "nze_missing_prime_hist", None):
                    hist = stats.get("nze_missing_prime_hist", {})
                    if not isinstance(hist, dict):
                        hist = {}
                    for p, c in res.nze_missing_prime_hist.items():
                        hist[p] = int(hist.get(p, 0)) + int(c)
                    stats["nze_missing_prime_hist"] = hist
                if getattr(res, "nze_min_missing_count", None) is not None:
                    cur = stats.get("nze_min_missing_count", None)
                    curm = stats.get("nze_min_missing_m", None)
                    nm = int(res.nze_min_missing_count)
                    nmm = int(res.nze_min_missing_m) if res.nze_min_missing_m is not None else None
                    if cur is None or nm < int(cur) or (nm == int(cur) and nmm is not None and (curm is None or nmm < int(curm))):
                        stats["nze_min_missing_count"] = nm
                        stats["nze_min_missing_m"] = nmm

                if res.err_samples:
                    es = stats.get("err_samples", [])
                    if isinstance(es, list) and len(es) < 80:
                        es.extend(res.err_samples)
                        stats["err_samples"] = es[:80]

                for m in res.ms:
                    found.add(m)

                if res.failed_masks:
                    for mask in res.failed_masks:
                        mask_attempts[mask] = mask_attempts.get(mask, 0) + 1
                        if mask_attempts[mask] >= max_mask_attempts:
                            unprocessed.append(mask)
                    retry_masks = [m for m in res.failed_masks if mask_attempts.get(m, 0) < max_mask_attempts]
                    if retry_masks:
                        for i in range(0, len(retry_masks), mask_requeue_batch):
                            queue.append(ChunkTask(masks=tuple(retry_masks[i:i + mask_requeue_batch])))
                            total += 1

                if done % 25 == 0 or not queue:
                    stats["unique_failed_masks"] = len(mask_attempts)
                    stats["unprocessed_masks"] = len(set(unprocessed))
                    logf.write(
                        f"{utc_now_iso()} progress chunks_done={done}/{total} "
                        f"cum_tried={stats['total_tried']} cum_failed={stats['total_failed']} "
                        f"unique_failed_masks={stats['unique_failed_masks']} "
                        f"unprocessed={stats['unprocessed_masks']} "
                        f"q_skipped={stats['masks_skipped_q_bound']} "
                        f"x1_skipped={stats['masks_skipped_x1_bound']} "
                        f"gp_bailout={stats['masks_skipped_gp_bailout']} "
                        f"iter_breaks={stats['iterates_early_break']} "
                        f"nze_skipped={stats['candidates_skipped_nze']}\n"
                    )
                    if DEBUG and nze_enabled and isinstance(stats.get("nze_missing_prime_hist"), dict) and stats["nze_missing_prime_hist"]:
                        # Log top missing primes (by frequency) among NZE-rejected candidates.
                        items = sorted(stats["nze_missing_prime_hist"].items(), key=lambda kv: (-int(kv[1]), int(kv[0])))
                        top = ",".join([f"{p}:{c}" for p, c in items[:10]])
                        logf.write(f"{utc_now_iso()} debug_nze_missing_top10={top} min_missing={stats.get('nze_min_missing_count')} at_m={stats.get('nze_min_missing_m')}\n")
                    if DEBUG and stats.get("err_samples"):
                        logf.write(f"{utc_now_iso()} debug_err_samples={stats['err_samples'][:5]}\n")
                    logf.flush()

    stats["unique_failed_masks"] = len(mask_attempts)
    stats["unprocessed_masks"] = len(set(unprocessed))

    return sorted(found), stats, sorted(set(unprocessed))


def compute_and_verify_lightweight(
    primes: List[int],
    r: int,
    k: int,
    nze_pruning: int,
    workers: int,
    chunk_masks: int,
    striped: bool,
    logf,
    gp_path: str,
    max_mask_attempts: int,
    mask_requeue_batch: int,
    gp_timeout: float,
    max_m: int = 0,
) -> Tuple[Dict[str, object], Dict[str, int], List[int]]:
    """Summary-only mode to avoid huge files and memory blow-ups for large ω.

    Enumerates P-smooth candidates (as produced by the Størmer/Lehmer Pell families),
    verifies each candidate on-the-fly for N_r(m), and returns only aggregated statistics.

    Returns: compute_stats, verify_stats, unprocessed_masks
    """
    P = tuple(primes)
    omega = len(P)
    pmax = P[-1]
    L = max(3, pmax)
    nze_enabled = (nze_pruning > 0 and omega >= nze_pruning)

    total_masks = 1 << omega
    start_mask = 0
    end_mask = total_masks

    # Build tasks
    tasks: List[ChunkTask] = []
    step = 1
    if striped:
        step = max(1, workers * 64)
        window = end_mask - start_mask
        step = min(step, window) if window > 0 else 1
        tasks = [ChunkTask(offset=o) for o in range(step)]
    else:
        for a in range(start_mask, end_mask, chunk_masks):
            b = min(end_mask, a + chunk_masks)
            tasks.append(ChunkTask(a=a, b=b))

    params = WorkerParams(
        primes=P,
        L=L,
        r=r,
        k=k,
        nze_enabled=nze_enabled,
        start_mask=start_mask,
        end_mask=end_mask,
        striped=striped,
        step=step,
        gp_timeout=gp_timeout,
        max_m=max_m,
    )

    ctx = get_context("fork") if sys.platform == "darwin" else get_context()

    compute_stats: Dict[str, object] = {
        "omega": omega,
        "pmax": pmax,
        "lehmer_L": L,
        "k": k,
        "nze_pruning": nze_pruning,
        "nze_enabled": nze_enabled,
        "max_m": max_m,
        "incremental_used": False,
        "start_mask": start_mask,
        "end_mask": end_mask,
        "striped": striped,
        "striped_step": step,
        "tasks_initial": len(tasks),
        "total_tried": 0,
        "total_failed": 0,
        "unique_failed_masks": 0,
        "unprocessed_masks": 0,
        "masks_skipped_q_bound": 0,
        "masks_skipped_x1_bound": 0,
        "masks_skipped_gp_bailout": 0,
        "iterates_early_break": 0,
        "candidates_skipped_nze": 0,
        "total_families": 0,
        "masks_no_fund_sols": 0,
        # NZE diagnostics (only populated when DEBUG and nze_enabled)
        "nze_missing_prime_hist": {},
        "nze_min_missing_count": None,
        "nze_min_missing_m": None,
        "err_samples": [],
        "S_raw_count": 0,
        "report_mode": "summary",
    }

    verify_stats: Dict[str, int] = {
        "count_m_S_pmax": 0,
        f"count_m_after_{r}_filter": 0,
        f"smooth_pass_{r}": 0,
        "pi_complete_hits": 0,
        "miss_star": 10**9,
        "miss_star_m": -1,
        "k": k,
    }

    mask_attempts: Dict[int, int] = {}
    unprocessed: List[int] = []
    queue: List[ChunkTask] = list(tasks)
    done = 0
    total = len(queue)

    target = set(P)

    def _verify_one(m: int) -> None:
        # If NZE pruning is enabled in the worker, every emitted m already:
        #  (i) is P-smooth across all r multipliers, and
        # (ii) has full prime support P (no missing primes).
        # So each m is a π-complete hit and we can update statistics without refactoring.
        if nze_enabled:
            verify_stats[f"count_m_after_{r}_filter"] += 1
            verify_stats[f"smooth_pass_{r}"] += 1
            verify_stats["pi_complete_hits"] += 1
            # miss_star is 0 by definition if any candidate exists
            if 0 < verify_stats["miss_star"]:
                verify_stats["miss_star"] = 0
                verify_stats["miss_star_m"] = m
            return

        fNr: Dict[int, int] = {}
        for j in range(r):
            mj = m + j
            fj, rem = factor_over_P(mj, P)
            if rem != 1:
                return
            fNr = factor_merge(fNr, fj)

        verify_stats[f"count_m_after_{r}_filter"] += 1
        verify_stats[f"smooth_pass_{r}"] += 1

        supp = support_tuple(fNr)
        supp_set = set(supp)
        if supp == P:
            verify_stats["pi_complete_hits"] += 1

        miss = len(target - supp_set)
        if miss < verify_stats["miss_star"]:
            verify_stats["miss_star"] = miss
            verify_stats["miss_star_m"] = m


    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(gp_path, DEBUG, ASSERTIONS),
    ) as pool:
        while queue:
            batch = queue[:workers * 8]
            queue = queue[workers * 8:]

            async_args = [(t, params) for t in batch]
            for res in pool.imap_unordered(worker_chunk, async_args, chunksize=1):
                done += 1
                compute_stats["total_tried"] = int(compute_stats["total_tried"]) + res.tried
                compute_stats["total_failed"] = int(compute_stats["total_failed"]) + res.failed
                compute_stats["masks_skipped_q_bound"] = int(compute_stats["masks_skipped_q_bound"]) + res.masks_skipped_q_bound
                compute_stats["masks_skipped_x1_bound"] = int(compute_stats["masks_skipped_x1_bound"]) + res.masks_skipped_x1_bound
                compute_stats["masks_skipped_gp_bailout"] = int(compute_stats["masks_skipped_gp_bailout"]) + res.masks_skipped_gp_bailout
                compute_stats["iterates_early_break"] = int(compute_stats["iterates_early_break"]) + res.iterates_early_break
                compute_stats["candidates_skipped_nze"] = int(compute_stats["candidates_skipped_nze"]) + getattr(res, "candidates_skipped_nze", 0)
                compute_stats["total_families"] = int(compute_stats["total_families"]) + getattr(res, "total_families", 0)
                compute_stats["masks_no_fund_sols"] = int(compute_stats["masks_no_fund_sols"]) + getattr(res, "masks_no_fund_sols", 0)

                # Aggregate NZE diagnostics (DEBUG-only payloads)
                if getattr(res, "nze_missing_prime_hist", None):
                    hist = compute_stats.get("nze_missing_prime_hist", {})
                    if not isinstance(hist, dict):
                        hist = {}
                    for p, c in res.nze_missing_prime_hist.items():
                        hist[p] = int(hist.get(p, 0)) + int(c)
                    compute_stats["nze_missing_prime_hist"] = hist
                if getattr(res, "nze_min_missing_count", None) is not None:
                    cur = compute_stats.get("nze_min_missing_count", None)
                    curm = compute_stats.get("nze_min_missing_m", None)
                    nm = int(res.nze_min_missing_count)
                    nmm = int(res.nze_min_missing_m) if res.nze_min_missing_m is not None else None
                    if cur is None or nm < int(cur) or (nm == int(cur) and nmm is not None and (curm is None or nmm < int(curm))):
                        compute_stats["nze_min_missing_count"] = nm
                        compute_stats["nze_min_missing_m"] = nmm

                if res.err_samples:
                    es = compute_stats.get("err_samples", [])
                    if isinstance(es, list) and len(es) < 80:
                        es.extend(res.err_samples)
                        compute_stats["err_samples"] = es[:80]

                if res.ms:
                    compute_stats["S_raw_count"] = int(compute_stats["S_raw_count"]) + len(res.ms)
                    verify_stats["count_m_S_pmax"] += len(res.ms)
                    for mval in res.ms:
                        _verify_one(mval)

                if res.failed_masks:
                    for mask in res.failed_masks:
                        mask_attempts[mask] = mask_attempts.get(mask, 0) + 1
                        if mask_attempts[mask] >= max_mask_attempts:
                            unprocessed.append(mask)
                    retry_masks = [m for m in res.failed_masks if mask_attempts.get(m, 0) < max_mask_attempts]
                    if retry_masks:
                        for i in range(0, len(retry_masks), mask_requeue_batch):
                            queue.append(ChunkTask(masks=tuple(retry_masks[i:i + mask_requeue_batch])))
                            total += 1

                if done % 25 == 0 or not queue:
                    compute_stats["unique_failed_masks"] = len(mask_attempts)
                    compute_stats["unprocessed_masks"] = len(set(unprocessed))
                    logf.write(
                        f"{utc_now_iso()} progress chunks_done={done}/{total} "
                        f"cum_tried={compute_stats['total_tried']} cum_failed={compute_stats['total_failed']} "
                        f"unique_failed_masks={compute_stats['unique_failed_masks']} "
                        f"unprocessed={compute_stats['unprocessed_masks']} "
                        f"q_skipped={compute_stats['masks_skipped_q_bound']} "
                        f"x1_skipped={compute_stats['masks_skipped_x1_bound']} "
                        f"gp_bailout={compute_stats['masks_skipped_gp_bailout']} "
                        f"iter_breaks={compute_stats['iterates_early_break']} "
                        f"S_raw_count={compute_stats['S_raw_count']} "
                        f"pi_hits={verify_stats['pi_complete_hits']}\n"
                    )
                    if DEBUG and nze_enabled and isinstance(compute_stats.get("nze_missing_prime_hist"), dict) and compute_stats["nze_missing_prime_hist"]:
                        items = sorted(compute_stats["nze_missing_prime_hist"].items(), key=lambda kv: (-int(kv[1]), int(kv[0])))
                        top = ",".join([f"{p}:{c}" for p, c in items[:10]])
                        logf.write(f"{utc_now_iso()} debug_nze_missing_top10={top} min_missing={compute_stats.get('nze_min_missing_count')} at_m={compute_stats.get('nze_min_missing_m')}\n")
                    logf.flush()

    compute_stats["unique_failed_masks"] = len(mask_attempts)
    compute_stats["unprocessed_masks"] = len(set(unprocessed))

    if verify_stats["miss_star"] == 10**9:
        verify_stats["miss_star"] = -1

    return compute_stats, verify_stats, sorted(set(unprocessed))




@dataclass
class VerifyRow:
    m: int
    r: int
    Nr_support: Tuple[int, ...]
    Nr_factorization: str
    pi_complete: bool
    missing_from_P: Tuple[int, ...]
    extra_primes: Tuple[int, ...]


def verify_and_write_csv_stream(ms: List[int], primes: List[int], k: int, csv_path: str) -> Dict[str, int]:
    """Verify gap-k candidates and write verify CSV incrementally (streaming).

    For each candidate m, checks that m*(m+k) is P-smooth and records
    prime support, factorization, and π-completeness.  Also tracks miss_star
    (minimum number of primes from P missing in the product's support).

    Returns verify_stats dict.
    """
    P = tuple(primes)
    target = set(P)
    stats: Dict[str, int] = {
        "count_m_S_pmax": len(ms),
        "count_m_after_gk_filter": 0,
        "smooth_pass_gk": 0,
        "pi_complete_hits": 0,
        "miss_star": 10**9,
        "miss_star_m": -1,
        "k": k,
    }

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "m",
            "k",
            "m_plus_k",
            "product_prime_support",
            "m_factorization",
            "m_plus_k_factorization",
            "pi_complete_order_omega",
            "missing_from_target_support",
            "extra_primes",
        ])

        for m in ms:
            f0, rem0 = factor_over_P(m, P)
            if rem0 != 1:
                continue
            f1, rem1 = factor_over_P(m + k, P)
            if rem1 != 1:
                continue

            fNr = factor_merge(f0, f1)
            stats["count_m_after_gk_filter"] += 1
            stats["smooth_pass_gk"] += 1

            supp = support_tuple(fNr)
            supp_set = set(supp)
            pi_ok = (supp == P)
            if pi_ok:
                stats["pi_complete_hits"] += 1

            miss = len(target - supp_set)
            if miss < stats["miss_star"]:
                stats["miss_star"] = miss
                stats["miss_star_m"] = m

            w.writerow([
                m,
                k,
                m + k,
                ",".join(map(str, supp)),
                format_factorization(f0),
                format_factorization(f1),
                int(pi_ok),
                ",".join(map(str, sorted(target - supp_set))),
                ",".join(map(str, sorted(supp_set - target))),
            ])

    if stats["miss_star"] == 10**9:
        stats["miss_star"] = -1

    return stats


def verify_stream_lightweight(ms: List[int], primes: List[int], k: int) -> Dict[str, int]:
    """Verify gap-k candidates without writing per-m rows (summary-only mode).

    Tracks smooth count, π-hit count, and miss_star (minimum missing primes).
    """
    P = tuple(primes)
    target = set(P)
    stats: Dict[str, int] = {
        "count_m_S_pmax": len(ms),
        "count_m_after_gk_filter": 0,
        "smooth_pass_gk": 0,
        "pi_complete_hits": 0,
        "miss_star": 10**9,
        "miss_star_m": -1,
        "k": k,
    }

    for m in ms:
        f0, rem0 = factor_over_P(m, P)
        if rem0 != 1:
            continue
        f1, rem1 = factor_over_P(m + k, P)
        if rem1 != 1:
            continue

        fNr = factor_merge(f0, f1)
        stats["count_m_after_gk_filter"] += 1
        stats["smooth_pass_gk"] += 1

        supp = support_tuple(fNr)
        supp_set = set(supp)
        if supp == P:
            stats["pi_complete_hits"] += 1

        miss = len(target - supp_set)
        if miss < stats["miss_star"]:
            stats["miss_star"] = miss
            stats["miss_star_m"] = m

    if stats["miss_star"] == 10**9:
        stats["miss_star"] = -1

    return stats
def write_summary_json(
    path: str,
    omega: int,
    k: int,
    primes: List[int],
    pmax: int,
    L: int,
    max_m: int,
    start_utc: str,
    end_utc: str,
    runtime_sec: float,
    env: Dict[str, object],
    workers: int,
    chunk_masks: int,
    striped: bool,
    incremental_used: bool,
    base_omega: Optional[int],
    s_file: str,
    verify_csv: str,
    log_file: str,
    unprocessed_file: Optional[str],
    compute_stats: Dict[str, object],
    verify_stats: Dict[str, int],
) -> None:
    summary = {
        "environment": env,
        "omega": omega,
        "k": k,
        "gk_equation": "x^2 - 2*q*y^2 = k^2",
        "gk_q_bound": f"q > 2*max_m*(max_m+k) = 2*{max_m}*({max_m}+{k})" if max_m > 0 else "disabled (max_m=0)",
        "primes": primes,
        "pmax": pmax,
        "lehmer_L": L,
        # NZE pruning settings are recorded in compute_stats; repeat here for convenience.
        "nze_pruning": int(compute_stats.get("nze_pruning", 0) or 0),
        "nze_enabled": bool(compute_stats.get("nze_enabled", False)),
        "total_families_found": int(compute_stats.get("total_families", 0)),
        "masks_no_fundamental_sols": int(compute_stats.get("masks_no_fund_sols", 0)),
        "start_utc": start_utc,
        "end_utc": end_utc,
        "runtime_seconds": runtime_sec,
        "workers": workers,
        "chunk_masks": chunk_masks,
        "striped": striped,
        "incremental_used": incremental_used,
        "incremental_base_omega": base_omega,
        "compute_stats": compute_stats,
        "verify_stats": verify_stats,
        "artifacts": {
            "S_file": s_file,
            "S_file_sha256": sha256_file(s_file) if s_file and os.path.isfile(s_file) else None,
            "verify_csv": verify_csv,
            "verify_csv_sha256": sha256_file(verify_csv) if verify_csv and os.path.isfile(verify_csv) else None,
            "log_file": log_file,
            "log_file_sha256": sha256_file(log_file),
            "unprocessed_file": unprocessed_file,
            "unprocessed_file_sha256": sha256_file(unprocessed_file) if unprocessed_file and os.path.isfile(unprocessed_file) else None,
        },
        "complete_enumeration": (int(compute_stats.get("unprocessed_masks", 0)) == 0),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def gk_dir(outdir: str, k: int, omega: int) -> str:
    """Return directory for (k, omega) run: {outdir}/k_{k:03d}/omega_{omega:02d}/"""
    return os.path.join(outdir, f"k_{k:03d}", f"omega_{omega:02d}")

def gk_done(outdir: str, k: int, omega: int) -> bool:
    d = gk_dir(outdir, k, omega)
    return os.path.isfile(os.path.join(d, f"summary_k{k}_omega_{omega:02d}.json"))


def _run_one_omega_k(
    omega: int,
    k: int,
    outdir: str,
    full_mode: bool,
    workers: int,
    args,
) -> Dict[str, object]:
    """Run the Pell enumeration for one (omega, k) pair.  Returns a result dict.

    The result dict has keys:
      omega, k, pmax, pi_complete_hits, miss_star, miss_star_m,
      complete_enumeration, certified, runtime_sec, pi_complete_ms
    where pi_complete_ms is a list of (m, m+k) tuples with pi_complete=True.
    """
    primes = primes_first_n(omega)
    pmax = primes[-1]
    L = max(3, pmax)

    d = gk_dir(outdir, k, omega)
    ensure_dir(d)

    s_path  = os.path.join(d, f"S_p{pmax}_k{k}_omega_{omega:02d}.txt")
    v_path  = os.path.join(d, f"verify_k{k}_omega_{omega:02d}.csv")
    summary_path = os.path.join(d, f"summary_k{k}_omega_{omega:02d}.json")
    log_path = os.path.join(d, f"run_k{k}_omega_{omega:02d}.log")
    unproc_path = os.path.join(d, f"unprocessed_masks_k{k}_omega_{omega:02d}.txt")

    start_utc = utc_now_iso()
    t0 = time.time()

    result: Dict[str, object] = {
        "omega": omega, "k": k, "pmax": pmax,
        "pi_complete_hits": 0, "miss_star": -1, "miss_star_m": -1,
        "complete_enumeration": False, "certified": False,
        "runtime_sec": 0.0, "pi_complete_ms": [],
    }

    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(
            f"{start_utc} START k={k} omega={omega} primes={primes} workers={workers} "
            f"chunk_masks={args.chunk_masks} striped={args.striped} "
            f"max_mask_attempts={args.max_mask_attempts} max_m={args.max_m} "
            f"debug={DEBUG} assertions={ASSERTIONS}\n"
        )
        logf.flush()

        try:
            if full_mode:
                S_list, compute_stats, unprocessed = compute_S_pmax_exact(
                    primes=primes,
                    r=2,
                    k=k,
                    nze_pruning=args.nze_pruning,
                    workers=workers,
                    chunk_masks=args.chunk_masks,
                    striped=args.striped,
                    incremental=False,
                    prior_S=None,
                    logf=logf,
                    gp_path=args.gp_path,
                    max_mask_attempts=args.max_mask_attempts,
                    mask_requeue_batch=args.mask_requeue_batch,
                    gp_timeout=args.gp_timeout,
                    max_m=args.max_m,
                )
                Ptuple = tuple(primes)
                S_gk = filter_for_gk(S_list, k=k, primes=Ptuple)
                write_int_list(s_path, S_gk)
                verify_stats = verify_and_write_csv_stream(S_gk, primes, k=k, csv_path=v_path)
                compute_stats["report_mode"] = "full"
                S_raw_ct = len(S_list)
                S_gk_ct = len(S_gk)

                # Collect pi-complete (m, m+k) pairs for min_gap table
                pi_complete_ms: List[Tuple[int, int]] = []
                if verify_stats["pi_complete_hits"] > 0:
                    P = Ptuple
                    target = set(P)
                    for m in S_gk:
                        f0, r0 = factor_over_P(m, P)
                        if r0 != 1:
                            continue
                        f1, r1 = factor_over_P(m + k, P)
                        if r1 != 1:
                            continue
                        supp = support_tuple(factor_merge(f0, f1))
                        if set(supp) == target:
                            pi_complete_ms.append((m, m + k))
                result["pi_complete_ms"] = pi_complete_ms
            else:
                compute_stats, verify_stats, unprocessed = compute_and_verify_lightweight(
                    primes=primes,
                    r=2,
                    k=k,
                    nze_pruning=args.nze_pruning,
                    workers=workers,
                    chunk_masks=args.chunk_masks,
                    striped=args.striped,
                    logf=logf,
                    gp_path=args.gp_path,
                    max_mask_attempts=args.max_mask_attempts,
                    mask_requeue_batch=args.mask_requeue_batch,
                    gp_timeout=args.gp_timeout,
                    max_m=args.max_m,
                )
                S_list = []
                S_gk = []
                S_raw_ct = int(compute_stats.get("S_raw_count", 0))
                S_gk_ct = int(verify_stats.get("count_m_after_gk_filter", 0))
                result["pi_complete_ms"] = []

            unprocessed_file: Optional[str] = None
            if unprocessed:
                unprocessed_file = unproc_path
                P = tuple(primes)
                with open(unproc_path, "w", encoding="utf-8") as f:
                    f.write("# mask\tq(mask)\tD=2*q\n")
                    for mask in unprocessed:
                        q = 1
                        mm = mask
                        i = 0
                        while mm:
                            if mm & 1:
                                q *= P[i]
                            mm >>= 1
                            i += 1
                        f.write(f"{mask}\t{q}\t{2*q}\n")

            end_utc = utc_now_iso()
            runtime = time.time() - t0

            write_summary_json(
                path=summary_path,
                omega=omega,
                k=k,
                primes=primes,
                pmax=pmax,
                L=L,
                max_m=args.max_m,
                start_utc=start_utc,
                end_utc=end_utc,
                runtime_sec=runtime,
                env=ENV,
                workers=workers,
                chunk_masks=args.chunk_masks,
                striped=args.striped,
                incremental_used=False,
                base_omega=None,
                s_file=(s_path if full_mode else ""),
                verify_csv=(v_path if full_mode else ""),
                log_file=log_path,
                unprocessed_file=unprocessed_file,
                compute_stats=compute_stats,
                verify_stats=verify_stats,
            )

            pi_hits = int(verify_stats["pi_complete_hits"])
            miss_star = int(verify_stats.get("miss_star", -1))
            miss_star_m = int(verify_stats.get("miss_star_m", -1))
            complete = (int(compute_stats.get("unprocessed_masks", 0)) == 0)

            logf.write(
                f"{end_utc} DONE k={k} omega={omega} "
                f"|S_raw|={S_raw_ct} |S_gk|={S_gk_ct} "
                f"pi_hits={pi_hits} miss_star={miss_star} "
                f"complete={complete} unprocessed={len(unprocessed)} "
                f"runtime_sec={runtime:.3f}\n"
            )
            logf.flush()

            result.update({
                "pi_complete_hits": pi_hits,
                "miss_star": miss_star,
                "miss_star_m": miss_star_m,
                "complete_enumeration": complete,
                "certified": complete and (int(compute_stats.get("total_failed", 0)) == 0),
                "runtime_sec": runtime,
            })

        except Exception as e:
            end_utc = utc_now_iso()
            logf.write(f"{end_utc} ERROR k={k} omega={omega} error={repr(e)}\n")
            logf.flush()
            print(f"[!] ERROR k={k} omega={omega}: {e!r} (see {log_path})")
            result["error"] = repr(e)

    return result


def _write_min_gap_table(outdir: str, table: List[Dict[str, object]]) -> None:
    """Write min_gap_table.csv and min_gap_table.json."""
    csv_path  = os.path.join(outdir, "min_gap_table.csv")
    json_path = os.path.join(outdir, "min_gap_table.json")

    fieldnames = [
        "omega", "pmax", "min_gap", "certified",
        "pi_complete_m", "pi_complete_m_plus_k",
        "prime_support", "runtime_sec",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in table:
            w.writerow(row)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, sort_keys=True)


def main() -> None:
    global DEBUG, ASSERTIONS

    ap = argparse.ArgumentParser(
        description="Gk_Solver: find minimum gap-k for π-complete m*(m+k) products.",
    )
    ap.add_argument("--version", action="store_true",
                    help="Print version/environment info and exit")
    ap.add_argument("--mode", default="fixed_k", choices=["fixed_k", "sweep_k"],
                    help="fixed_k: enumerate all π-complete (m,m+k) pairs for a specific k. "
                         "sweep_k: for each ω, find min_gap(ω) = smallest k with a π-complete hit.")
    ap.add_argument("--gap_k", type=int, default=1,
                    help="Gap size k for fixed_k mode (default 1, i.e. consecutive pairs).")
    ap.add_argument("--max_k", type=int, default=200,
                    help="Maximum k to try in sweep_k mode before giving up (default 200).")
    ap.add_argument("--outdir", default=f"{program_name}_v{program_version}_audit_runs")
    ap.add_argument("--start_omega", type=int, default=8)
    ap.add_argument("--end_omega", type=int, default=20)
    ap.add_argument("--full_report_omega_max", type=int, default=20,
                    help="Write full artifacts (S-list + verify CSV) for ω ≤ this. "
                         "Above this threshold run in summary-only mode (no large files).")
    ap.add_argument("--nze_pruning", type=int, default=0,
                    help="Enable NZE (π-complete) pruning in worker for ω ≥ this threshold. "
                         "0 = disabled. For referee runs keep 0 or set above full_report_omega_max.")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--chunk_masks", type=int, default=32)
    ap.add_argument("--striped", action="store_true")
    ap.add_argument("--gp_path", default="gp")
    ap.add_argument("--max_mask_attempts", type=int, default=8)
    ap.add_argument("--mask_requeue_batch", type=int, default=64)
    ap.add_argument("--gp_timeout", type=float, default=0.0)
    ap.add_argument("--max_m", type=int, default=0,
                    help="Upper bound on m for early rejection. "
                         "Masks with q > 2*max_m*(max_m+k) are skipped (provably no solution ≤ max_m). "
                         "Iterate loops break when m > max_m. "
                         "0 = disabled (full unbounded enumeration).")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--assertions", action="store_true")
    args = ap.parse_args()

    # Version/env reporting (for reproducible certification)
    if args.version:
        info = env_block(args.gp_path, __file__, sys.argv)
        print(json.dumps(info, indent=2, sort_keys=True))
        return

    DEBUG = bool(args.debug)
    ASSERTIONS = bool(args.assertions)

    global ENV
    ENV = env_block(args.gp_path, __file__, sys.argv)

    outdir = args.outdir
    ensure_dir(outdir)

    workers = args.workers if args.workers and args.workers > 0 else (os.cpu_count() or 1)

    # Verify gp executable
    try:
        subprocess.run([args.gp_path, "-q"], input="print(1);\nquit\n",
                       text=True, capture_output=True, check=True)
    except Exception:
        print("[!] Could not run gp. Pass --gp_path /opt/homebrew/bin/gp (or your path).")
        raise

    print(f"[+] {program_name} v{program_version}")
    print(f"[+] mode={args.mode} outdir={outdir}")
    print(f"[+] omega range: {args.start_omega}..{args.end_omega}")
    if args.mode == "fixed_k":
        print(f"[+] gap_k={args.gap_k}")
    else:
        print(f"[+] max_k={args.max_k} (sweep will stop at first π-complete hit per ω)")
    print(f"[+] full_report_omega_max={args.full_report_omega_max}")
    print(f"[+] workers={workers} chunk_masks={args.chunk_masks} striped={args.striped}")
    print(f"[+] gp_path={args.gp_path}")
    print(f"[+] max_m={args.max_m}" + (" (early rejection enabled)" if args.max_m > 0 else " (disabled, full enumeration)"))
    print(f"[+] debug={DEBUG} assertions={ASSERTIONS}")
    print(f"[+] start time (UTC): {utc_now_iso()}\n")

    # ----------------------------- fixed_k mode --------------------------------
    if args.mode == "fixed_k":
        k = args.gap_k
        for omega in range(args.start_omega, args.end_omega + 1):
            if gk_done(outdir, k, omega):
                print(f"[=] k={k} omega={omega}: already done, skipping.")
                continue
            full_mode = (omega <= args.full_report_omega_max)
            res = _run_one_omega_k(omega=omega, k=k, outdir=outdir,
                                   full_mode=full_mode, workers=workers, args=args)
            primes = primes_first_n(omega)
            pmax = primes[-1]
            L = max(3, pmax)
            print(
                f"[>] k={k} omega={omega} pmax={pmax} L={L} "
                f"pi_hits={res['pi_complete_hits']} "
                f"miss_star={res['miss_star']} "
                f"complete={res['complete_enumeration']} "
                f"runtime={float(res['runtime_sec'])/60:.2f} min"
            )

    # ----------------------------- sweep_k mode --------------------------------
    elif args.mode == "sweep_k":
        min_gap_table: List[Dict[str, object]] = []

        for omega in range(args.start_omega, args.end_omega + 1):
            primes = primes_first_n(omega)
            pmax = primes[-1]
            full_mode = (omega <= args.full_report_omega_max)

            print(f"\n[sweep] omega={omega} pmax={pmax} — trying k=1..{args.max_k}")
            omega_start_t = time.time()
            found_k: Optional[int] = None
            certified_all_below = True
            first_pi_complete_pair: Optional[Tuple[int, int]] = None
            prime_support_str = ""
            total_omega_runtime = 0.0

            for k in range(1, args.max_k + 1):
                if gk_done(outdir, k, omega):
                    # Load existing summary to get pi_complete_hits
                    d = gk_dir(outdir, k, omega)
                    sjson = os.path.join(d, f"summary_k{k}_omega_{omega:02d}.json")
                    try:
                        with open(sjson, encoding="utf-8") as f:
                            s = json.load(f)
                        pi_hits = int(s.get("verify_stats", {}).get("pi_complete_hits", 0))
                        complete = bool(s.get("complete_enumeration", False))
                        runtime_k = float(s.get("runtime_seconds", 0.0))
                    except Exception:
                        pi_hits = 0
                        complete = False
                        runtime_k = 0.0
                    print(f"  [=] k={k} omega={omega}: already done (pi_hits={pi_hits} complete={complete}), skipping.")
                else:
                    res = _run_one_omega_k(omega=omega, k=k, outdir=outdir,
                                           full_mode=full_mode, workers=workers, args=args)
                    pi_hits = int(res["pi_complete_hits"])
                    complete = bool(res["complete_enumeration"])
                    runtime_k = float(res["runtime_sec"])
                    # If we found π-complete pairs, grab the first one
                    if pi_hits > 0 and res.get("pi_complete_ms"):
                        pair = res["pi_complete_ms"][0]  # type: ignore[index]
                        if first_pi_complete_pair is None:
                            first_pi_complete_pair = pair

                    print(
                        f"  [-] k={k} omega={omega} pi_hits={pi_hits} "
                        f"miss_star={res.get('miss_star', -1)} "
                        f"complete={complete} "
                        f"runtime={runtime_k/60:.2f} min"
                    )

                total_omega_runtime += runtime_k
                if not complete:
                    certified_all_below = False

                if pi_hits > 0:
                    found_k = k
                    # If pi_complete_ms not set yet (summary-only mode), scan S-file if available
                    if first_pi_complete_pair is None and full_mode:
                        d = gk_dir(outdir, k, omega)
                        s_path = os.path.join(d, f"S_p{pmax}_k{k}_omega_{omega:02d}.txt")
                        if os.path.isfile(s_path):
                            P = tuple(primes)
                            target = set(P)
                            for m in load_int_list(s_path):
                                f0, r0 = factor_over_P(m, P)
                                if r0 != 1:
                                    continue
                                f1, r1 = factor_over_P(m + k, P)
                                if r1 != 1:
                                    continue
                                supp = support_tuple(factor_merge(f0, f1))
                                if set(supp) == target:
                                    first_pi_complete_pair = (m, m + k)
                                    prime_support_str = ",".join(map(str, sorted(target)))
                                    break
                    break

            omega_total_t = time.time() - omega_start_t

            # Assemble table row
            row: Dict[str, object] = {
                "omega": omega,
                "pmax": pmax,
                "min_gap": found_k if found_k is not None else f">{args.max_k}",
                "certified": certified_all_below,
                "pi_complete_m": first_pi_complete_pair[0] if first_pi_complete_pair else "",
                "pi_complete_m_plus_k": first_pi_complete_pair[1] if first_pi_complete_pair else "",
                "prime_support": prime_support_str,
                "runtime_sec": round(omega_total_t, 3),
            }
            min_gap_table.append(row)
            _write_min_gap_table(outdir, min_gap_table)

            print(
                f"[sweep] omega={omega} → min_gap={row['min_gap']} "
                f"certified={certified_all_below} "
                f"pair=({row['pi_complete_m']}, {row['pi_complete_m_plus_k']}) "
                f"total_runtime={omega_total_t/60:.2f} min"
            )
            print(f"  min_gap_table written to {outdir}/min_gap_table.csv")

    print(f"\n[+] Finished. End time (UTC): {utc_now_iso()}")


if __name__ == "__main__":
    if sys.platform == "darwin":
        try:
            import multiprocessing as mp
            mp.set_start_method("fork")
        except RuntimeError:
            pass
    main()