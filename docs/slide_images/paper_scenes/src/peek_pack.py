"""Cars20.blend の 20 台を 1 台ずつ、屋根を切った真上と、車内の斜め上から描く（スクリプトは実行しない）。
blender --background --factory-startup --python peek_pack.py -- Cars20.blend OUT_DIR"""
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
RES = int(argv[2]) if len(argv) > 2 else 420
ONLY = argv[3].split(",") if len(argv) > 3 else None
path, out = argv[0], argv[1]
os.makedirs(out, exist_ok=True)
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
with bpy.data.libraries.load(path, link=False) as (src, dst):
    dst.objects = [n for n in src.objects if n.startswith("Car")]
sc = bpy.context.scene
for o in bpy.data.objects:
    sc.collection.objects.link(o)
sc.render.engine = "BLENDER_WORKBENCH"
sc.display.shading.light = "STUDIO"
sc.display.shading.color_type = "MATERIAL"
sc.display.shading.show_cavity = True
sc.render.resolution_x, sc.render.resolution_y = RES, RES
cd = bpy.data.cameras.new("Peek")
cam = bpy.data.objects.new("Peek", cd)
sc.collection.objects.link(cam)
sc.camera = cam
bodies = [o for o in bpy.data.objects if o.type == "MESH" and "Tire" not in o.name and (ONLY is None or o.name in ONLY)]
for b in bodies:
    for o in bpy.data.objects:
        if o.type == "MESH":
            o.hide_render = not (o == b or o.parent == b)
    bpy.context.view_layer.update()
    pts = [b.matrix_world @ Vector(c) for c in b.bound_box]
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    c = (lo + hi) / 2
    H = hi.z - lo.z
    cd.type = "ORTHO"
    cd.ortho_scale = max(hi - lo) * 1.05
    cam.location = Vector((c.x, c.y, hi.z + 2.0))
    cam.rotation_euler = (0, 0, 0)
    cd.clip_start = 2.0 + H * 0.42
    cd.clip_end = 100
    sc.render.filepath = os.path.join(out, f"{b.name}_top.png")
    bpy.ops.render.render(write_still=True)
    print("CAR", b.name, "size", [round(x, 3) for x in (hi - lo)], "min", [round(x, 3) for x in lo], "max", [round(x, 3) for x in hi])
