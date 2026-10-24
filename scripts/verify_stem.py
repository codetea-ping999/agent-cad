"""Run the stem verification loop on a parameter set and print the rule table.

    python -m scripts.verify_stem                         # default stem
    python -m scripts.verify_stem --case stem110_-17_7075
    python -m scripts.verify_stem --set length=120 --set material=A7075-T6 --json out/v.json

Exit status 1 when any rule fails, so the command can gate a parameter sweep or CI.
"""
import argparse, dataclasses, json, sys, typing
from agent_cad.parts.stem import StemParams
from agent_cad.verify.stem_verify import verify


def parse_set(p, items):
    types = typing.get_type_hints(StemParams)      # the declared type: a preset may hold 110 for 110.0
    kw = {}
    for it in items:
        k, _, v = it.partition("=")
        if k not in types:
            raise SystemExit(f"unknown parameter {k!r}")
        t = types[k]
        kw[k] = (v.lower() in ("1", "true", "yes")) if t is bool else t(v)
    return dataclasses.replace(p, **kw)


def main(argv=None):
    from scripts.make_stem import CASES
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=sorted(CASES), default="stem90_6061")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--no-fea", action="store_true")
    ap.add_argument("--h", type=float, default=2.0, help="FEA element size, mm")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    p = parse_set(CASES[a.case], a.set)
    r = verify(p, fea=not a.no_fea, h=a.h)
    for x in r["rules"]:
        prov = "  (provisional load)" if x.get("provisional_load") else ""
        print(f"{'OK  ' if x['pass'] else 'FAIL'} {x['check']:<40} {x['value']!s:>10} / {x['limit']!s:<8} {x['basis']}{prov}")
    print("PASS" if r["pass"] else f"FAIL: {', '.join(r['failed'])}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(r, fh, indent=1, ensure_ascii=False, default=str)
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
