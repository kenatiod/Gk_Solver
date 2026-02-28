# Gk_Solver: An Essay on π-Complete Products, Generalized Pell Equations, and the Search for the Last Smooth Pair

*Ken Clements, February 2026*

---

## 1. The Central Question

At the heart of this research lies a deceptively simple question about whole numbers.

Call a positive integer *P-smooth* if every prime factor of that integer belongs to a fixed finite set P. For example, if P = {2, 3, 5}, then 360 = 2³ × 3² × 5 is P-smooth but 77 = 7 × 11 is not.

Now consider two consecutive integers, or more generally two integers separated by a fixed gap k: the pair (m, m+k). When is it possible for their product m · (m+k) to be not merely P-smooth, but *exhaustively* P-smooth — meaning every prime in P actually appears in the factorization of the product?

This exhaustiveness condition is what we call **π-completeness**. Formally, a product m · (m+k) is π-complete for a prime set P_ω = {p₁, p₂, ..., p_ω} if and only if the prime support of m · (m+k) equals P_ω *exactly* — no prime from P_ω is missing, and no prime outside P_ω appears.

The canonical prime sets are the sets of the first ω primes:

```
P₁ = {2}
P₂ = {2, 3}
P₃ = {2, 3, 5}
...
P₈ = {2, 3, 5, 7, 11, 13, 17, 19}
P₉ = {2, 3, 5, 7, 11, 13, 17, 19, 23}
```

The number ω is called the **prime count** or **omega value** of the pair. We can write ω(n) for the number of distinct prime factors of an integer n; for a π-complete pair, ω(m · (m+k)) = ω and the support is exactly P_ω.

These objects — pairs whose product uses every prime up to some bound and no others — turn out to be extraordinarily rare. As ω grows, they become harder and harder to find, and eventually, the computational and theoretical evidence suggests they cease to exist altogether. Gk_Solver is the program that quantifies this scarcity.

---

## 2. The Founding Result: The Last Consecutive Smooth Pair

The story begins with a theorem that can be stated concretely.

**The n=633,555 theorem.** The pair (633,555, 633,556) is the last pair of consecutive integers whose product has prime support exactly P_ω for *any* ω.

To verify: 633,555 = 3³ × 5 × 13 × 19² and 633,556 = 2² × 7 × 11³ × 17. Together, their combined prime factors are {2, 3, 5, 7, 11, 13, 17, 19} = P₈. Every prime in P₈ appears; no prime outside P₈ appears. The product 633,555 × 633,556 is π-complete for P₈ with gap k=1.

This result was established computationally by the companion program **Nr_Solver** (v17), which exhaustively searches for all π-complete consecutive pairs (gap=1) across all ω. The search is made tractable — and certified complete — by a classical theorem of Størmer and Lehmer: for each prime set P, there are only finitely many P-smooth consecutive pairs, and the Størmer method generates all of them. Nr_Solver implements this method at scale, verifying that ω=8 with m=633,555 is the last hit.

The natural next question: what happens if we allow a gap larger than 1?

---

## 3. Why Variable Gaps Matter

With gap k=1, the n=633,555 result is a sharp endpoint. But π-complete products are not intrinsically tied to consecutive integers. A pair (m, m+k) for any gap k could in principle be π-complete. Perhaps there are π-complete products at gap k=2, gap k=5, gap k=100 — and perhaps such products exist for ω=9, 10, and beyond.

If so, the n=633,555 result would be only a partial answer. The *complete* answer to "where do π-complete products end?" would require accounting for all gaps simultaneously.

The program **Gk_Solver** attacks this question through a two-part strategy.

**The min_gap function.** For each prime count ω, define min_gap(ω) as the smallest gap k such that there exists any π-complete product m · (m+k) for P_ω. This turns the question from "does a π-complete pair exist at ω?" into "how large does the gap need to be before one appears?"

**The conjecture.** If min_gap(ω) grows without bound as ω → ∞, then for any fixed gap k, there are only finitely many π-complete products total. Combined with the Baker-type effective bounds on linear forms in logarithms, this would constitute a proof component that π-complete products are finite in number, not just for gap=1 but for every gap simultaneously.

The computational results from Gk_Solver's first runs (ω=9 through 12) already provide striking evidence: min_gap(ω) > 100 for every ω in this range, with max_m = 10⁹. The best candidates found are always several primes short of π-completeness, and that shortfall grows as ω increases.

---

## 4. The Mathematical Framework: Generalized Pell Equations

To search for π-complete pairs systematically, Gk_Solver reduces the problem to a family of Diophantine equations.

### 4.1 Setting up the Equation

Let D = 2q where q is the squarefree product of the primes in a given mask — a subset of P_ω. Any π-complete product m · (m+k) for P_ω can be written as m · (m+k) = q · y² for some integer y, where q is the radical (squarefree core) of the product's prime factorization.

Now substitute x = 2m + k. Then:

```
x² - k² = (2m + k)² - k² = 4m² + 4mk = 4m(m + k) = 4qy²
```

Dividing through:

```
x² - 2q·y² = k²
```

This is the **generalized Pell equation** for gap k. The variable x is related to m by m = (x − k) / 2, so any integer solution (x, y) with x ≥ k and x ≡ k (mod 2) yields a candidate m.

For k = 1, this reduces to x² − 2qy² = 1, exactly the equation solved by Nr_Solver. This is not a coincidence — Gk_Solver was designed to generalize Nr_Solver precisely, and the k=1 case is verified to recover every result that Nr_Solver produces.

### 4.2 What the Mask Represents

The product m · (m+k) must have prime support exactly P_ω. How is this encoded in the equation? Through the **mask** — a subset of the primes {p₁, ..., p_ω} that captures the "squarefree core" of the product's factorization.

The program iterates over all 2^ω possible masks. For each mask, it computes q as the product of the primes in the mask (or 1 if the mask is empty, giving q = 1). The equation x² − 2qy² = k² is then solved for that specific q. Solutions give candidates m; the program then checks whether the full prime support of m · (m+k) equals P_ω.

There are 2^ω masks per (k, ω) pair. For ω=9 this is 512 masks; for ω=12 it is 4,096. The exponential growth in masks with ω is the dominant cost driver.

### 4.3 Multiple Solution Families

Here is the key difference between gap=1 and gap k>1.

For gap=1 (x² − 2qy² = 1), there is exactly one "fundamental" solution per mask: the minimal solution (x₁, y₁) produced by the standard Pell algorithm. All other solutions are generated from this one by the Pell automorphism:

```
x_{n+1} = x₁ · x_n + 2q · y₁ · y_n
y_{n+1} = y₁ · x_n + x₁ · y_n
```

This recurrence generates an infinite sequence of solutions, but only finitely many can yield candidates m ≤ max_m.

For gap k>1 (x² − 2qy² = k²), the equation may have **multiple families** of solutions. Each family is anchored by its own **fundamental solution** (x₀, y₀) — a minimal solution not expressible as an automorphic image of a smaller one — and then generates an infinite sequence via the same Pell automorphism, using the standard Pell solution (x₁, y₁) as the automorphism generator.

The number of fundamental solutions can be bounded (it is related to the number of representations of k² as a norm in the ring Z[√(2q)]), but it is in general greater than 1 for k>1. Gk_Solver must find *all* fundamental solutions, then iterate each family.

---

## 5. The CRT Stride Search: Finding All Fundamental Solutions

The central algorithmic challenge in Gk_Solver is efficiently enumerating all fundamental solutions to x² − 2qy² = k² for a given q and k. A naive approach — test every integer x from k up to the theoretical bound — is completely impractical for large q.

### 5.1 Why Naive Search Fails

The theoretical bound on fundamental solutions is approximately x₀ ≤ k · √ε₁, where ε₁ = x₁ + y₁√(2q) is the fundamental Pell unit (the solution to x² − 2qy² = 1). For large q — and in the ω=12 sweep, q can easily exceed 10^18 — this bound is astronomical. A linear scan from x=k to x=k·√ε₁ would take longer than the age of the universe.

### 5.2 The Key Observation: Modular Constraints

The CRT stride search exploits a simple but powerful observation. If (x, y) solves x² − 2qy² = k², then for every prime p dividing q (and hence dividing D = 2q):

```
x² ≡ k²  (mod p)
x ≡ ±k   (mod p)
```

That is, x must be congruent to either +k or −k modulo every prime p dividing q. For each such prime, there are at most two residue classes. By the **Chinese Remainder Theorem (CRT)**, the combination of constraints across all primes in q gives at most 2^(number of distinct prime factors) residue classes modulo q.

Since q divides D and D = 2q is squarefree (q is squarefree by construction), and since ω primes divide D, there are at most 2^ω residue classes modulo D that any fundamental solution x can belong to.

### 5.3 Stepping Through Residue Classes

For each residue class r (mod D), the set of integers x ≡ r (mod D) with k ≤ x ≤ bound is an arithmetic progression with step D. When D is large (which it is for the large masks that dominate the computation), this arithmetic progression may contain only 0 or 1 integers in the range [k, bound].

For each such candidate x, the program checks whether the remaining condition — that (x² − k²) / D is a perfect square — holds. If it does, (x, √((x²−k²)/D)) is a fundamental solution.

The total work per mask is:
- 2^ω CRT class constructions: O(ω · 2^ω)
- At most one candidate per class in the feasible range: O(2^ω) divisibility checks

This is **O(2^ω) per mask** — a massive improvement over the O(k · √ε₁) naive bound.

### 5.4 The k=1 Special Case

For k=1, the equation x² − 2qy² = 1 has exactly one fundamental solution (x₁, y₁), produced directly by the standard PARI/GP function `pellxy(D)`. In this case, the CRT stride search is bypassed entirely — calling `genpell_fund(D, 1)` simply returns `[pellxy(D)]`. This preserves the exact behavior of Nr_Solver for gap=1 while allowing the same code path to handle k>1 transparently.

### 5.5 Deduplication

One subtlety arises when k is divisible by a prime p: in that case, k ≡ −k (mod p), so the two residue choices +k and −k (mod p) collapse to one. The CRT construction must detect and deduplicate these cases to avoid counting each solution twice.

---

## 6. Completeness Bounds

A numerical search is only scientifically meaningful if it comes with guarantees about what it has and has not missed.

### 6.1 The q-Bound

The most important completeness criterion is the **q-bound**. For a solution (x, y) to x² − 2qy² = k² with y ≥ 1, we have x² ≥ k² + 2q, so:

```
m = (x − k) / 2 ≥ (√(k² + 2q) − k) / 2
```

If this minimum m already exceeds our search limit max_m, then this mask can have no solutions within our range. The condition is:

```
q > 2 · max_m · (max_m + k)  →  skip this mask
```

For k=1, this reduces to q > 2 · max_m · (max_m + 1), which is precisely the bound used by Nr_Solver. Gk_Solver generalizes it naturally. A mask that satisfies this bound is **provably empty** of candidates below max_m — not a failure, but a conclusive negative result.

### 6.2 The L-Bound (BHV Primitive Divisors)

Within each solution family anchored at (x₀, y₀), the Pell automorphism generates an infinite sequence of solutions with rapidly growing x values. How many iterates must be checked before exceeding max_m?

The **Bilu-Hanrot-Voutier (BHV) theorem** (2001) on primitive prime divisors of Lucas sequences provides the answer. For the Pell recurrence, any iterate whose corresponding m value is P_ω-smooth (i.e., all prime factors belong to P_ω) must appear within the first L = max(3, p_max) iterates of its family, where p_max is the largest prime in P_ω.

This is the same L-bound used by Nr_Solver and is one of the two pillars on which the completeness of the k=1 results rests. For k>1, the same theorem applies to each family individually, since each family follows the same Pell recurrence structure.

### 6.3 Certified vs. Bounded

For k=1, the Størmer-Lehmer theorem provides an *unconditional* completeness guarantee: every P-smooth consecutive pair arises from a Pell solution within the first pmax iterates, and the Pell equation has exactly one fundamental solution. The enumeration is mathematically complete.

For k>1, completeness requires in addition that all fundamental solutions have been found and that max_m is large enough. The first condition is satisfied by the CRT stride search (which is exhaustive up to the theoretical bound). The second condition means that results for k>1 carry the caveat `certified=False` — they are bounded rather than provably complete in the Størmer sense. Extending this to certified completeness for k>1 would require Baker-type effective bounds, which remain work in progress.

---

## 7. How the Program Works

### 7.1 Architecture

Gk_Solver is a Python 3 program that delegates number-theoretic computation to **PARI/GP** (a specialized computer algebra system for number theory) via subprocess calls, while handling parallelism, data management, and verification in Python.

The program uses a **multiprocessing worker pool**. Each worker receives a contiguous range of masks to process and handles GP communication independently. A master process collects results and merges them. For a machine with 8 cores, 8 workers process different mask ranges in parallel, reducing wall time by roughly 8×.

The GP interaction is managed through a persistent subprocess per worker. GP is started once, loaded with the custom Pell functions (including `genpell_fund` and `genpell_iterates`), and then queried for each mask. This amortizes the startup cost over thousands of mask evaluations.

### 7.2 The Inner Loop

For each mask, the worker:

1. **Computes q** from the mask bitmask (product of selected primes).
2. **Applies the q-bound**: if q > 2 · max_m · (max_m + k), the mask is skipped (provably no solutions).
3. **Calls `genpell_fund(D, k, max_x)`** via GP to find all fundamental solutions. For k=1, this returns `[pellxy(D)]`; for k>1, it runs the CRT stride search.
4. **Gets the standard Pell solution** (x₁, y₁) via `pellxy(D)` (used as the automorphism generator).
5. **For each fundamental solution (x₀, y₀)**: iterates L = max(3, pmax) times using the Pell automorphism, collecting m = (x − k) / 2 at each step where x ≡ k (mod 2) and x ≥ k.
6. **Filters candidates**: each m is checked for P_ω-smoothness of both m and m+k. Smooth pairs are added to the candidate set.

After all masks are processed, the collected candidates are verified against the full π-completeness criterion: both m and m+k must be P_ω-smooth, and together their prime support must equal P_ω exactly.

### 7.3 Verification and Output

The verifier reads every candidate m, factorizes m and m+k over P_ω, checks that both factorizations are complete (no remainder prime), and records:
- The combined prime support
- Whether it equals P_ω (π-complete flag)
- The number of missing primes from P_ω (contributing to the miss_star statistic)
- Full factorizations of m and m+k

This produces a **verify CSV** for each (k, ω) pair, with one row per candidate. The CSV is the primary audit artifact: it allows any result to be checked independently.

A **summary JSON** records run metadata: the Pell equation used, the q-bound formula, total families found, masks with no solutions, GP failure count, SHA256 hashes of all output files, environment information (Python version, PARI/GP version, OS), and the completeness flag.

### 7.4 Operating Modes

**fixed_k mode** runs a single specified gap k across a range of ω values. This is the analogue of running Nr_Solver for a specific k. It produces the full artifact set (S-file, verify CSV, summary JSON, log) for each (k, ω).

**sweep_k mode** is the primary scientific mode. For each ω in a specified range, it runs k = 1, 2, 3, ... up to max_k, stopping the moment any π-complete hit is found. The result for each ω is `min_gap(ω)` — the smallest gap for which a π-complete product exists — or `>max_k` if no hit was found within the search range. After each ω completes, the current min_gap table is written to disk in both CSV and JSON formats.

---

## 8. The miss_star Statistic: Reading the Near-Misses

A pure count of π-complete hits (either 0 or positive) tells us whether min_gap(ω) exceeds our search range. But it does not tell us *how far* the system is from producing a hit. For this, Gk_Solver tracks **miss_star**.

For each candidate m (a pair where both m and m+k are P_ω-smooth), miss_star counts how many primes from P_ω are absent from the combined support of m · (m+k). A miss_star of 0 means the pair is π-complete. A miss_star of 1 means exactly one prime from P_ω is missing — a near-miss.

The **minimum miss_star** across all candidates and all k values in a sweep gives the closest approach to π-completeness observed for a given ω. This is the key diagnostic metric.

---

## 9. Results: The First Sweep, ω = 9 through 12

The inaugural sweep of Gk_Solver ran with:
- ω range: 9 to 12
- k range: 1 to 100
- max_m: 10⁹ (candidates m must satisfy m ≤ 10⁹)
- Workers: 8 parallel processes

### 9.1 The Main Finding

**Zero π-complete hits across all 400 tested (k, ω) pairs.**

For every combination of ω ∈ {9, 10, 11, 12} and k ∈ {1, ..., 100}, the program found no m ≤ 10⁹ such that m · (m+k) is π-complete for P_ω. The min_gap table reads:

| ω  | pmax | min_gap | min miss_star | Runtime |
|----|------|---------|--------------|---------|
| 9  | 23   | >100    | 1            | 65 min  |
| 10 | 29   | >100    | 2            | 132 min |
| 11 | 31   | >100    | 3            | 268 min |
| 12 | 37   | >100    | 3            | 542 min |

### 9.2 The miss_star Gradient

The minimum miss_star values (1, 2, 3, 3 for ω=9,10,11,12) reveal the structure of the failure.

For ω=9 (P₉ = {2,3,5,7,11,13,17,19,23}), the closest near-misses found — at gaps k=13, 52, 57, 65, 75, 76, and 77 — each have exactly one prime from P₉ missing from their combined support. The system came within one prime of π-completeness but never reached it.

For ω=10, the best candidates are always two primes short of P₁₀-completeness.

For ω=11 and 12, the best candidates are three primes short.

This gradient is significant. It suggests that as ω increases, the "distance" to π-completeness grows — the system is not merely failing to find π-complete pairs, but is systematically failing by an increasing margin. Each additional prime in P_ω makes π-completeness harder to achieve, and the data shows that hardness increasing concretely.

### 9.3 Runtime Scaling

The runtime roughly doubles with each increment in ω:

```
ω= 9:   65 min  (512 masks per k)
ω=10:  132 min  (1,024 masks per k)
ω=11:  268 min  (2,048 masks per k)
ω=12:  542 min  (4,096 masks per k)
```

This is exactly the 2^ω scaling predicted by the mask count. The per-mask computation time (dominated by the PARI/GP Pell solver) is roughly constant across ω. For ω=13 (pmax=41, 8,192 masks per k), the full k=1..100 sweep would require approximately 18 hours.

---

## 10. Connection to OEIS A392344 and the Gap-Plugging Mechanism

The sweep results connect naturally to a sequence in the On-Line Encyclopedia of Integer Sequences (OEIS).

**OEIS A392344** is defined as: a(n) = the greatest m such that m · (m+n) has a complete contiguous set of prime divisors from 2 to its greatest prime divisor (i.e., is π-complete for some P_ω). This is the "last π-complete pair at fixed gap n" sequence, and it is dual to min_gap(ω).

The two largest known terms arise through what we call the **gap-plugging mechanism**. The Nr_Solver search at ω=8 produces not just the π-complete pair (633,555, 633,556) but also several pairs that are P₈-smooth but *not* π-complete — they use all but one prime from P₈.

Two critical near-misses appear in the ω=8 verify output:

**Case 1.** The pair (709,631, 709,632) has combined prime support {2, 3, 7, 11, 13, 17, 19} — all of P₈ except the prime 5.

```
709,631 = 13³ × 17 × 19
709,632 = 2¹⁰ × 3² × 7 × 11
```

Multiply both by 5: m = 5 × 709,631 = 3,548,155 and m+5 = 5 × 709,632 = 3,548,160. Now:

```
m · (m+5) = 25 × 709,631 × 709,632
```

Prime support = {5} ∪ {13, 17, 19} ∪ {2, 3, 7, 11} = {2, 3, 5, 7, 11, 13, 17, 19} = P₈.

The product is π-complete at gap k=5. This gives a(5) = 3,548,155 in A392344.

**Case 2.** The pair (5,909,760, 5,909,761) has combined prime support {2, 3, 5, 11, 13, 17, 19} — all of P₈ except the prime 7.

```
5,909,760 = 2⁸ × 3⁵ × 5 × 19
5,909,761 = 11² × 13² × 17²
```

Multiply both by 7: m = 7 × 5,909,760 = 41,368,320 and m+7 = 7 × 5,909,761 = 41,368,327.

```
m · (m+7) = 49 × 5,909,760 × 5,909,761
```

Prime support = {7} ∪ {2, 3, 5, 19} ∪ {11, 13, 17} = {2, 3, 5, 7, 11, 13, 17, 19} = P₈.

The product is π-complete at gap k=7. This gives a(7) = 41,368,320 in A392344.

The pattern is clear: single-prime-gap pairs from the ω=8 search, when multiplied by the missing prime p, become π-complete at gap k=p. The Gk_Solver sweep confirms that no ω=9 pair (for k ≤ 100, m ≤ 10⁹) plugs in the same way, because ω=9 pairs never even get close to P₉-completeness.

---

## 11. Two Complementary Views of the Same Problem

The relationship between A392344 and min_gap(ω) is best understood as two complementary slices of a two-dimensional landscape. One axis is the gap k; the other is the prime count ω.

| Question | Variable fixed | Quantity sought | Tool |
|----------|---------------|-----------------|------|
| For fixed gap k: what is the largest π-complete pair? | k | max m over all ω | Nr_Solver + Gk_Solver |
| For fixed ω: what is the smallest gap with any π-complete pair? | ω | min k | Gk_Solver sweep_k |

The A392344 sequence scans horizontally across the landscape, asking how far right (in m) you can go for each fixed k. The min_gap table scans vertically, asking how far down (in k) you need to go for each fixed ω.

The sweep result min_gap(ω) > 100 for ω=9..12 places a lower bound on the vertical extent of the "empty zone" — the region of (k, ω) space where no π-complete pairs exist. The combination of increasing miss_star and the 2^ω scaling of computation suggests that this empty zone grows in both dimensions as ω increases.

---

## 12. The Proof Architecture

The computational results are not merely curiosities — they form part of a proof strategy.

**Step 1 (Nr_Solver).** For gap=1, the Størmer-Lehmer theorem makes the search provably complete. Nr_Solver verifies that m=633,555 is the unique π-complete pair for P₈ (gap=1) and that P₉ has none. This is a theorem, not a conjecture.

**Step 2 (Gk_Solver, current).** For gap k>1, the q-bound and BHV L-bound make the search complete within the range m ≤ max_m, but "certified" completeness for all m requires Baker bounds. The current sweep results (min_gap(ω) > 100 for ω=9..12) are strong evidence but not yet a proof.

**Step 3 (Baker bounds, future).** For each ω, Baker's theorem on linear forms in logarithms provides an *effective* upper bound on the size of any P_ω-smooth numbers. Combining this bound with the Pell equation structure would give an explicit max_m beyond which no solutions can exist. This would convert the current `certified=False` results into theorems.

**Step 4 (The limiting argument).** If min_gap(ω) can be shown to grow without bound — or equivalently, if for each fixed k the number of π-complete pairs at gap k is finite — then the totality of π-complete products is finite, and the last one is identified.

The Gk_Solver sweep results are the computational core of Step 2, providing the data that will eventually be certified by Step 3.

---

## 13. What Makes This Hard

Several features of the problem conspire to make computational progress difficult.

**Exponential mask growth.** The 2^ω mask count means each additional prime in P_ω doubles the computation. ω=20 would require 2^20 = 1,048,576 masks per k value — roughly 1,000 times more than ω=9.

**Large Pell discriminants.** For the full P_ω mask (all ω primes present), D = 2 × p₁ × p₂ × ... × p_ω grows as the primorial. At ω=12, D ~ 2 × 37# ≈ 4.4 × 10¹². The PARI/GP Pell solver handles this, but with nontrivial per-call cost.

**No Størmer-type theorem for k>1.** For consecutive pairs (k=1), Størmer's theorem provides a complete characterization. For k>1, no such theorem exists, which is precisely why the program needs Baker bounds as a substitute for certified completeness.

**The sparsity of smooth numbers.** P_ω-smooth numbers below 10⁹ become extraordinarily rare as ω decreases (paradoxically: with fewer primes in the allowed set, smoothness becomes harder to achieve for large m). This means the candidate lists are thin, and π-complete pairs are vanishingly sparse.

---

## 14. Conclusion

Gk_Solver represents the computational frontier of a line of research that begins with the elementary observation that smoothness and completeness are competing properties: the more primes you require in a product, the harder it is to have all of them appear simultaneously in a product of two nearby integers.

The program's core algorithmic contribution — the CRT stride search for fundamental solutions to x² − 2qy² = k² — reduces what would be an astronomically large search to a manageable O(2^ω) computation per mask. Combined with the q-bound and BHV L-bound, this makes the search both efficient and rigorous within its stated range.

The first sweep, covering ω=9..12 and k=1..100 with m ≤ 10⁹, found **zero π-complete hits** across 400 (k, ω) pairs. The minimum miss_star gradient (1, 2, 3, 3) shows the system retreating from π-completeness as ω grows. Together, these results constitute strong computational evidence for the conjecture that min_gap(ω) → ∞ as ω → ∞ — that is, that π-complete products become impossible for all gaps simultaneously as the prime count grows large enough.

If this conjecture is correct, then not only is n=633,555 the last π-complete consecutive pair, but the entire inventory of π-complete products of any fixed gap is finite. The universe of "perfect" smooth pairs is bounded. Gk_Solver is the instrument designed to map that boundary.

---

*Source code, sweep results, and documentation are available at:*
*https://github.com/kenatiod/Gk_Solver*

*Companion program for gap=1:*
*https://github.com/kenatiod/Nr_Solver*
