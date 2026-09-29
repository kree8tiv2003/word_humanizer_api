#!/usr/bin/env python3
"""Structural check for ComfyUI UI workflows: every link, slot and subgraph instance must agree."""
import json
import sys


def check(name, nodes, links, sgs, sg=None):
    errs = []
    L = {}
    for l in links:
        if isinstance(l, list):
            l = dict(id=l[0], origin_id=l[1], origin_slot=l[2], target_id=l[3], target_slot=l[4], type=l[5])
        L[l["id"]] = l
    N = {n["id"]: n for n in nodes}
    for lid, l in L.items():
        o, t = l["origin_id"], l["target_id"]
        if o == -10:
            if lid not in sg["inputs"][l["origin_slot"]]["linkIds"]:
                errs.append(f"{name}: subgraph input does not list link {lid}")
        elif o not in N or lid not in (N[o]["outputs"][l["origin_slot"]].get("links") or []):
            errs.append(f"{name}: origin of link {lid} inconsistent")
        if t == -20:
            if lid not in sg["outputs"][l["target_slot"]]["linkIds"]:
                errs.append(f"{name}: subgraph output does not list link {lid}")
        elif t not in N or N[t]["inputs"][l["target_slot"]].get("link") != lid:
            errs.append(f"{name}: target of link {lid} inconsistent")
    for n in nodes:
        for i in n.get("inputs", []):
            if i.get("link") is not None and i["link"] not in L:
                errs.append(f"{name}: node {n['id']} input {i['name']} -> missing link {i['link']}")
        for o in n.get("outputs", []):
            for lid in o.get("links") or []:
                if lid not in L:
                    errs.append(f"{name}: node {n['id']} output {o['name']} -> missing link {lid}")
        if n["type"] in sgs:
            d = sgs[n["type"]]
            if [i["name"] for i in n["inputs"]] != [i["name"] for i in d["inputs"]]:
                errs.append(f"{name}: instance {n['id']} inputs differ from subgraph definition")
            if len(n["outputs"]) != len(d["outputs"]):
                errs.append(f"{name}: instance {n['id']} outputs differ from subgraph definition")
            if n.get("properties", {}).get("proxyWidgetErrorQuarantine"):
                errs.append(f"{name}: instance {n['id']} has quarantined widgets")
    return errs


def main(paths):
    bad = 0
    for p in paths:
        wf = json.load(open(p))
        sgs = {s["id"]: s for s in wf.get("definitions", {}).get("subgraphs", [])}
        errs = check("root", wf["nodes"], wf["links"], sgs)
        for s in sgs.values():
            errs += check(s["name"], s["nodes"], s["links"], sgs, s)
        print(("OK   " if not errs else "FAIL ") + p, f"({len(wf['nodes'])} nodes, {len(sgs)} subgraphs)")
        for e in errs:
            print("   ", e)
        bad += bool(errs)
    sys.exit(bad)


if __name__ == "__main__":
    main(sys.argv[1:])
