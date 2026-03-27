from __future__ import annotations

from black_hole_lib import (
    minimum_useful_output,
    schwarzschild_radius,
    solar_masses_to_kg,
)


def main() -> None:
    # Example input set in SI units.
    M = solar_masses_to_kg(10.0)  # 10 solar masses
    astar = 0.7
    r = 12.0 * schwarzschild_radius(M)  # 12 r_s
    mdot = 1.0e15  # kg/s

    output = minimum_useful_output(M=M, astar=astar, r=r, mdot=mdot, L=2.0, eta=0.1, prograde=True)

    print("Input:")
    print(f"  M         = {M:.6e} kg")
    print(f"  a_*       = {astar:.3f}")
    print(f"  r         = {r:.6e} m")
    print(f"  mdot      = {mdot:.6e} kg/s")
    print()

    print("Minimum useful output set:")
    for key, value in output.items():
        if value is None:
            print(f"  {key:28s}: None")
        else:
            print(f"  {key:28s}: {value:.6e}")


if __name__ == "__main__":
    main()
