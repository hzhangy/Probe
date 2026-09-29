#!/usr/bin/env python3
"""TCVP re-derivation of theta_13 via port equipartition."""
import numpy as np
from math import pi, sqrt


def main():
    print("=" * 60)
    print("  TCVP re-derivation of theta_13")
    print("=" * 60)

    Delta = 1.0 - sqrt(3.0) / 2.0
    ports = 6

    print(f"\n  Step 1. Octahedral ports: {ports}")
    print(f"  Step 2. Locking gap Delta = 1 - sqrt(3)/2 = {Delta:.10f}")
    print(f"  Step 3. Equipartition: Delta/6 = {Delta/6:.10f}")

    sin2_theta = Delta / ports
    theta_rad = np.arcsin(np.sqrt(sin2_theta))
    theta_deg = np.degrees(theta_rad)

    print(f"\n  sin^2(theta_13) = Delta/6 = {sin2_theta:.10f}")
    print(f"  theta_13 (rad) = arcsin(sqrt(Delta/6)) = {theta_rad:.10f}")
    print(f"  theta_13 (deg) = {theta_deg:.6f}")

    obs = 8.58
    dev = 100 * (theta_deg - obs) / obs
    print(f"\n  Observed: {obs} deg")
    print(f"  Deviation: {dev:+.4f}%")

    if abs(dev) < 1.0:
        print("\n  PASS. theta_13 from port equipartition matches observation.")
    else:
        print("\n  FAIL.")


if __name__ == "__main__":
    main()