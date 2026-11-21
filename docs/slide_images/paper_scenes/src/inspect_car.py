"""車の .blend の部品ごとの大きさ・位置・モディファイア・使っている材質を一覧にし、車内を真上（屋根を切る）と運転席の斜め上から描く。
スクリプトは実行しない（libraries.load で部品だけを読む）。
blender --background --factory-startup --python inspect_car.py -- CAR.blend OUT_DIR"""
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
path, out = argv[0], argv[1]
os.makedirs(out, exist_ok=True)
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
with bpy.data.libraries.load(path, link=False) as (src, dst):
    dst.objects = [n for n in src.objects if n.startswith(("Compact Car Part", "Wheel"))]
sc = bpy.context.scene
for o in bpy.data.objects:
    sc.collection.objects.link(o)
bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get()
lo_all, hi_all = Vector((1e9,) * 3), Vector((-1e9,) * 3)
for o in sorted(bpy.data.objects, key=lambda o: o.name):
    if o.type != "MESH":
        continue
    ev = o.evaluated_get(dg)
    me = ev.to_mesh()
    vs = [ev.matrix_world @ v.co for v in me.vertices]
    used = sorted({o.material_slots[p.material_index].material.name for p in me.polygons
                   if p.material_index < len(o.material_slots) and o.material_slots[p.material_index].material})
    lo = Vector([min(v[i] for v in vs) for i in range(3)]) if vs else Vector()
    hi = Vector([max(v[i] for v in vs) for i in range(3)]) if vs else Vector()
    lo_all = Vector([min(lo_all[i], lo[i]) for i in range(3)])
    hi_all = Vector([max(hi_all[i], hi[i]) for i in range(3)])
    print("PART", o.name, "eval_verts", len(vs), "mods", [m.type for m in o.modifiers], "parent", o.parent.name if o.parent else None,
          "min", [round(x, 3) for x in lo], "max", [round(x, 3) for x in hi], "used_mats", used)
    ev.to_mesh_clear()
print("ALL", [round(x, 3) for x in lo_all], [round(x, 3) for x in hi_all])
sc.render.engine = "BLENDER_WORKBENCH"
sc.display.shading.light = "STUDIO"
sc.display.shading.color_type = "MATERIAL"
sc.display.shading.show_cavity = True
sc.render.resolution_x, sc.render.resolution_y = 1200, 900
cd = bpy.data.cameras.new("Peek")
cam = bpy.data.objects.new("Peek", cd)
sc.collection.objects.link(cam)
sc.camera = cam
c = (lo_all + hi_all) / 2
H = hi_all.z - lo_all.z
cd.type = "ORTHO"
cd.ortho_scale = max(hi_all - lo_all) * 1.02
cam.location = Vector((c.x, c.y, hi_all.z + 3.0))
cam.rotation_euler = (0, 0, 0)
cd.clip_start = 3.0 + H * 0.33
cd.clip_end = 100
sc.render.filepath = os.path.join(out, "car_top_cut.png")
bpy.ops.render.render(write_still=True)
cd.clip_start = 3.0 + H * 0.12
sc.render.filepath = os.path.join(out, "car_top_cut_high.png")
bpy.ops.render.render(write_still=True)
