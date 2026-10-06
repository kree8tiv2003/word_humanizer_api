"""Draw an OpenPose-style layout guide for the character sheet.

Layout (1344x768, an SDXL-friendly bucket):
  left   : 4 full-body turnarounds (front, 3/4, side, back)
  right  : 3 head-and-shoulders busts (front, 3/4, profile)
The ControlNet (OpenPose) locks this layout so the model paints one character
in every panel.
"""
import math
import sys

from PIL import Image, ImageDraw

W, H = 1344, 768

# OpenPose COCO-18 order: 0 nose,1 neck,2 rsho,3 relb,4 rwri,5 lsho,6 lelb,7 lwri,
# 8 rhip,9 rkne,10 rank,11 lhip,12 lkne,13 lank,14 reye,15 leye,16 rear,17 lear
# x,y in body-height units (y down, x=0 on the centre line), z = depth toward viewer's facing.
BASE = {
    0: (0.0, 0.065, 0.05), 1: (0.0, 0.145, 0.0),
    2: (-0.10, 0.16, 0.0), 3: (-0.16, 0.30, -0.01), 4: (-0.21, 0.43, -0.03),
    5: (0.10, 0.16, 0.0), 6: (0.16, 0.30, 0.01), 7: (0.21, 0.43, 0.03),
    8: (-0.055, 0.52, 0.0), 9: (-0.06, 0.72, 0.02), 10: (-0.065, 0.93, 0.03),
    11: (0.055, 0.52, 0.0), 12: (0.06, 0.72, -0.02), 13: (0.065, 0.93, -0.03),
    14: (-0.012, 0.052, 0.03), 15: (0.012, 0.052, 0.03),
    16: (-0.028, 0.062, -0.02), 17: (0.028, 0.062, -0.02),
}
# 0-indexed limb pairs, same order/colours as controlnet_aux's OpenPose renderer.
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
         (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
          (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
          (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
          (255, 0, 255), (255, 0, 170), (255, 0, 85)]


def project(angle_deg):
    a = math.radians(angle_deg)
    return {i: (x * math.cos(a) + z * math.sin(a), y) for i, (x, y, z) in BASE.items()}


def draw_figure(d, angle, cx, top, height, keep, hide=()):
    pts = project(angle)
    px = {i: (cx + x * height, top + y * height) for i, (x, y) in pts.items()
          if i in keep and i not in hide}
    width = max(3, int(height / 150))
    for (a, b), col in zip(LIMBS, COLORS):
        if a in px and b in px:
            d.line([px[a], px[b]], fill=tuple(int(c * 0.6) for c in col), width=width * 2)
    for i, (x, y) in px.items():
        r = width + 1
        d.ellipse([x - r, y - r, x + r, y + r], fill=COLORS[i])


def main(out):
    img = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(img)
    full = set(range(18))
    fig_h, top = 640, 60
    for k, (angle, hide) in enumerate([(0, ()), (40, ()), (90, ()), (180, (0, 14, 15))]):
        draw_figure(d, angle, 150 + k * 235, top, fig_h, full, hide)
    bust = {0, 1, 2, 5, 14, 15, 16, 17}
    for k, (angle, hide) in enumerate([(0, ()), (40, ()), (90, ())]):
        # head ~100px tall inside a 250px cell on the right
        draw_figure(d, angle, 1200, 25 + k * 250, 760, bust, hide)
    img.save(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pose_sheet.png")
