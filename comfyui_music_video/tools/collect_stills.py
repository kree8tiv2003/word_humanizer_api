#!/usr/bin/env python3
"""Copy the newest output/god_inside_scene_XX_*.png from workflow 02 to input/god_inside_scene_XX.png.

  python3 tools/collect_stills.py /path/to/ComfyUI
"""
import glob
import os
import shutil
import sys

comfy = sys.argv[1] if len(sys.argv) > 1 else "."
for i in range(1, 28):
    hits = sorted(glob.glob(os.path.join(comfy, "output", f"god_inside_scene_{i:02d}_*.png")), key=os.path.getmtime)
    if not hits:
        print(f"scene {i:02d}: no still yet")
        continue
    dst = os.path.join(comfy, "input", f"god_inside_scene_{i:02d}.png")
    shutil.copyfile(hits[-1], dst)
    print(f"scene {i:02d}: {os.path.basename(hits[-1])} -> {dst}")
