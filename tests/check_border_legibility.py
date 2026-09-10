"""Can you tell a significant box from an insignificant one, at every ramp step?

An insignificant box has a white border, so that -- not the fill -- is what the
significance border has to contrast with.

    uv run python tests/check_border_legibility.py
"""

from __future__ import annotations

from mlwp_scorecards.colours import SCHEMES, _relative_luminance, contrast_ratio


def main() -> None:
    for name, scheme in sorted(SCHEMES.items()):
        print(f"\n{name}")
        print(f"  {'family':<10} {'level':>5} {'fill':>9} {'border':>9} "
              f"{'vs white':>9} {'vs fill':>8}")
        worst_white = (99.0, None)
        for family in scheme.families:
            for level in (1, 4, 7, 10, 14):
                sw = scheme.swatch(family, level)
                vs_white = contrast_ratio(sw.edge, scheme.insignificant_edge)
                vs_fill = contrast_ratio(sw.edge, sw.fill)
                if vs_white < worst_white[0]:
                    worst_white = (vs_white, (family, level))
                print(f"  {family:<10} {level:>5} {sw.fill:>9} {sw.edge:>9} "
                      f"{vs_white:>8.2f} {vs_fill:>8.2f}")

        print(f"  worst vs white: {worst_white[0]:.2f} at {worst_white[1]}")
        darker = all(
            _relative_luminance(scheme.swatch(f, lv).edge)
            < _relative_luminance(scheme.swatch(f, lv).fill)
            for f in scheme.families
            for lv in range(1, scheme.depth + 1)
        )
        print(f"  every border darker than its fill: {darker}")
        if worst_white[0] < 3.0:
            print("  -> too low: significant and insignificant boxes will look alike")


if __name__ == "__main__":
    main()
