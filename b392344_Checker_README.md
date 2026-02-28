# b392344_Checker: Analysis of OEIS A392344

## Overview

`b392344_Checker.py` verifies each entry in `b392344.txt` — the b-file for OEIS sequence A392344
— and annotates the structural type of each π-complete product.

**A392344**: a(n) = greatest m such that m·(m+n) is π-complete (prime support = {2, 3, ..., p_ω} exactly for some ω).

Running the checker against all 200 terms of the b-file confirms:
- Every entry is a valid π-complete product.
- The terms fall into three structurally distinct families.

```bash
pip3 install sympy
python3 b392344_Checker.py
```

---

## Structural Families

### 1. Single Gap Fills (dominant mechanism)

A **single gap fill** occurs when m = n × j, where the pair (j, j+1) is P_ω-smooth but missing exactly one prime from P_ω, and that missing prime equals n.

```
m · (m+n) = n² · j · (j+1)
```

Multiplying both j and j+1 by n inserts the missing prime p=n into the product, completing the prime support. This is the primary source of large a(n) values.

**Key examples from the first 200 terms:**

| n | m | Missing prime | Source pair |
|---|---|--------------|-------------|
| 5 | 3,548,155 | 5 | (709,631, 709,632), P₈ missing 5 |
| 7 | 41,368,320 | 7 | (5,909,760, 5,909,761), P₈ missing 7 |
| 10 | 7,096,310 | 5 | (709,631, 709,632) × 2 |
| 15 | 10,644,465 | 5 | (709,631, 709,632) × 3 |
| 17 | 1,361,042,848 | 17 | P₁₁ near-miss, missing 17 |

The checker prints `Single gap fill` when m % n == 0 and j·(j+1) has exactly one prime missing from a contiguous initial sequence.

---

### 2. Double Gap Fills — and Their Family Structure

A **double gap fill** occurs when j·(j+1) is missing **two** primes from P_ω. Any n divisible by both missing primes fills them simultaneously.

#### The j = 8,268,799 family

The pair (8,268,799, 8,268,800) has prime support P₁₀ \ {3, 13} — all primes through 29 except 3 and 13. Any n divisible by both 3 and 13 (i.e., divisible by 39) produces a π-complete P₁₀ product:

| n | m = n × 8,268,799 | Product ω |
|---|-------------------|-----------|
| 39 = 3×13 | 322,483,161 | 10 |
| 78 = 2×3×13 | 644,966,322 | 10 |
| 117 = 3²×13 | 967,449,483 | 10 |
| 156 = 4×3×13 | 1,289,932,644 | 10 |
| 195 = 5×3×13 | 1,612,415,805 | 10 |

All five appear in the b-file. They share identical j, and n=78,117,156,195 are all multiples of 39.

The product factorization for n=39:
```
{2:10, 3:2, 5:2, 7:2, 11:1, 13:2, 17:1, 19:1, 23:2, 29:1}  →  P₁₀-complete
```

The factor 39² = (3×13)² contributes the missing 3² and 13², completing P₁₀.

The checker prints `Double gap fill` when m % n == 0 and j·(j+1) is missing exactly two primes from a contiguous initial sequence.

---

### 3. Direct Pairs (not gap-plug derived)

Many terms have m not divisible by n. These are **intrinsic** π-complete pairs with no gap-plug ancestry. They span a wide range of ω:

| n | m | ω | Note |
|---|---|---|------|
| 29 | 2,437,120 | 9 | Direct P₉ pair, primes through 23 |
| 37 | 373,490 | 7 | Direct P₇ pair, primes through 17 |
| 43 | 5,767,125 | 6 | P₆ pair, pmax=13 |
| 89 | 154,791 | 6 | P₆ pair, pmax=13 |
| 131 | 301,665 | 8 | Direct P₈, not from 633,555 family |
| 137 | 823,543 = 7⁷ | 6 | m is a pure 7th power of 7 |

The n=137 entry is particularly striking: m = 7⁷ = 823,543 is a pure prime power, and its product 7⁷ × (7⁷ + 137) factors over P₆ = {2, 3, 5, 7, 11, 13}.

---

## Omega Distribution

The ω value of each product m·(m+n) across the first 200 terms:

| ω | Largest prime | Example n values |
|---|--------------|-----------------|
| 6 | 13 | 43, 86, 89, 103, 129, 137, 172 |
| 7 | 17 | 37, 59, 74, 111, 118 |
| 8 | 19 | 1–9, 12, 13, 16, 24, 26, 27, ... (majority) |
| 9 | 23 | 29, 47, 61, 125, 167 |
| 10 | 29 | 11, 17, 19, 22, 23, 33, 34, 38–40, ... |
| 11 | 31 | 41, 82, 95, 123, 161 |
| 12 | 37 | 49, 68, 98, 121, 136, 147, 196 |
| 13 | 41 | 95, 190 |

**ω=8 dominates.** The richest source of π-complete raw material is the ω=8 level, fed by three Nr_Solver near-misses: (633,555, 633,556), (709,631, 709,632), and (5,909,760, 5,909,761).

---

## The n=121 Anomaly: A Direct P₁₂ Pair

The entry a(121) = 84,693,984,375 is the most surprising in the first 200 terms:

```
n = 121 = 11²
m = 84,693,984,375
m*(m+121) factorization: {2:4, 3:1, 5:7, 7:1, 11:2, 13:1, 17:1, 19:2, 23:1, 29:1, 31:1, 37:2}
ω = 12,  largest prime = 37  (12th prime)  →  P₁₂-complete
```

This is a **direct P₁₂ pair at gap 121**, not derived from any smaller near-miss. Noteworthy features:
- 5⁷ appears in the factorization — a single prime dominating with a large exponent, a signature of a well-iterated Pell orbit.
- 37² appears — the largest prime squared, typical of a fundamental Pell solution iterated once.
- m ≈ 8.5 × 10¹⁰, well above the max_m=10⁹ used in the Gk_Solver ω=12 sweep. This explains why Gk_Solver did not find it: the sweep would need max_m ≳ 85 billion for ω=12, k=121 to capture this pair.

This entry confirms that isolated high-ω pairs do appear at specific gaps even when the general sweep finds nothing, and that the A392344 sequence is not exclusively an ω=8 phenomenon.

---

## Connection to Gk_Solver

The checker's gap-fill classification maps directly onto the Gk_Solver architecture:

- **Single gap fills** correspond to fundamental Pell solutions in the (ω−1)-prime search scaled up by one prime.
- **Double gap fills** correspond to pairs with gap_count=2 in the supporting smooth-pair database — a rarer but structurally identical construction.
- **Direct pairs** are what Gk_Solver's sweep_k mode hunts: π-complete products that arise without any gap-plug ancestry. The sweep found zero such pairs for ω=9..12 with k=1..100 and m≤10⁹, consistent with the b-file showing no ω≥9 direct pairs for n≤200 below that size threshold.

The n=121 anomaly illustrates the most important limitation of the current sweep: max_m=10⁹ is sufficient for ω=9 and 10 (where the miss_star gradient shows the system is far from π-completeness), but may be insufficient for specific (k, ω) combinations at higher ω where a stray direct pair could exist at larger m.

---

*Part of the Gk_Solver research project by Ken Clements (Feb 2026).*
*https://github.com/kenatiod/Gk_Solver*
