#!/usr/bin/env python3
"""Put the bundle of tools/demo_bundle.py into the race page and write it as a standalone HTML document.

    python tools/build_demo.py [--src demo/race.src.html] [--bundle demo/bundle.json] [--out demo/race.html]

The bundle replaces the one /*BUNDLES*/ of the source as a JSON script element. Every '<' of the bundle is written
as \\u003c, so that no text in it can end that element.
"""
import argparse
import json
import sys


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="demo/race.src.html")
    ap.add_argument("--bundle", default="demo/bundle.json")
    ap.add_argument("--out", default="demo/race.html")
    a = ap.parse_args(argv)
    src = open(a.src, encoding="utf-8").read()
    if src.count("/*BUNDLES*/") != 1:
        print("build_demo: the source must hold /*BUNDLES*/ exactly once")
        return 1
    txt = open(a.bundle, encoding="utf-8").read()
    json.loads(txt)
    txt = txt.replace("<", "\\u003c")
    page = src.replace("/*BUNDLES*/", f'<script type="application/json" class="bundle" data-name="BRRR">{txt}</script>')
    head_end = page.index('<div id="wrap">')
    doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
           + page[:head_end] + "</head>\n<body>\n" + page[head_end:] + "\n</body>\n</html>\n")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"build_demo: wrote {a.out}, {len(doc) / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
