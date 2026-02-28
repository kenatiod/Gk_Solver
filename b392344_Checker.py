#!/usr/bin/env python3
"""
Check b-file for OEIS sequence A392344: a(n) = greatest m such that m*(m+n) is π-complete


By Ken Clements, Feb 01 2026
"""

import sympy

def get_gap_count(n):
    pf = sympy.primefactors(n)
    omega = len(pf)
    pidx = sympy.primepi(pf[-1])
    return pidx - omega

def check_pi_complete(n, m):
    """Check if m*(m+n) is π-complete"""
    product = m * (m + n)
    pf = sympy.primefactors(product)
    
    # For π-complete: if there are k distinct prime factors,
    # the largest should be the k-th prime
    omega = len(pf)  # number of distinct prime factors
    largest_prime = pf[-1]
    kth_prime = sympy.prime(omega)
    
    is_valid = (kth_prime == largest_prime)
    
    return {
        'n': n,
        'm': m,
        'product': product,
        'prime_factors': pf,
        'omega': omega,
        'largest_prime': largest_prime,
        'kth_prime': kth_prime,
        'is_pi_complete': is_valid
    }

def main():
    filename = 'b392344.txt'
    
    print("Checking π-complete property for b-file...")
    print("=" * 70)
    
    errors = []
    checked = 0
    
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            if line[0] == "#": continue    

            parts = line.split()
            if len(parts) != 2:
                print(f"Line {line_num}: Invalid format - {line}")
                continue
            
            try:
                n = int(parts[0])
                m = int(parts[1])
                
                result = check_pi_complete(n, m)
                checked += 1
                
                if not result['is_pi_complete']:
                    errors.append((line_num, result))
                    print(f"\n❌ FAILED at line {line_num}:")
                    print(f"   n = {n}, m = {m}")
                    print(f"   m*(m+n) = {result['product']}")
                    print(f"   Prime factors: {result['prime_factors']}")
                    print(f"   ω = {result['omega']}, largest prime = {result['largest_prime']}")
                    print(f"   Expected: prime({result['omega']}) = {result['kth_prime']}")
                    print(f"   But largest prime is {result['largest_prime']}")
                else:
                    product = result['product']
                    print(f"n = {n} m = {m:,} Product = {product:,} with factorization {sympy.factorint(product)}", end="  ", flush=True) 
                    if m  %n == 0:
                        j = m // n
                        gap_count = get_gap_count(j*(j+1))
                        if gap_count == 1:
                            print(f"Single gap fill")
                        elif gap_count == 2:
                            print(f"Double gap fill")
                        else:
                            print("")
                    else:
                        print("")
       

            except ValueError as e:
                print(f"Line {line_num}: Error parsing '{line}' - {e}")
                continue
    
    print("\n" + "=" * 70)
    print(f"Checked {checked} entries")
    
    if errors:
        print(f"\n⚠️  Found {len(errors)} ERRORS:")
        for line_num, result in errors:
            print(f"   Line {line_num}: n={result['n']}, m={result['m']}")
    else:
        print("\n✓ All entries are valid π-complete numbers!")
    
    return len(errors) == 0

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)

    