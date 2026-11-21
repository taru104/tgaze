"""落とした .blend を屋根を切った真上と横から簡単に描いて、内装があるかを見る（スクリプトは実行しない）。
blender --background --factory-startup --python peek_blend.py -- FILE.blend OUT_PREFIX"""
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
path, out = argv[0], argv[1]
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
with bpy.data.libraries.load(path, link=False) as (src, dst):
    dst.objects = list(src.objects)
sc = bpy.context.scene
for o in bpy.data.objects:
    if o.type == "MESH":
        sc.collection.objects.link(o)
bpy.context.view_layer.update()
meshes = [o for o in sc.objects if o.type == "MESH" and len(o.data.vertices) > 8]
pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
lo = Vector([min(p[i] for p in pts) for i in range(3)])
hi = Vector([max(p[i] for p in pts) for i in range(3)])
c = (lo + hi) / 2
sc.render.engine = "BLENDER_WORKBENCH"
sc.display.shading.light = "STUDIO"
sc.display.shading.color_type = "MATERIAL"
sc.display.shading.show_cavity = True
sc.render.resolution_x, sc.render.resolution_y = 900, 900
cd = bpy.data.cameras.new("Peek")
cam = bpy.data.objects.new("Peek", cd)
sc.collection.objects.link(cam)
sc.camera = cam
size = max(hi - lo)
# 真上（屋根を切る：カメラの手前の切り取り距離を、車の高さの 25% 下まで）
cd.type = "ORTHO"
cd.ortho_scale = size * 1.05
cam.location = Vector((c.x, c.y, hi.z + 2.0))
cam.rotation_euler = (0, 0, 0)
cd.clip_start = 2.0 + (hi.z - lo.z) * 0.28
cd.clip_end = 100
sc.render.filepath = out + "_top_cut.png"
bpy.ops.render.render(write_still=True)
# 横（車の幅の真ん中で切る）
cam.location = Vector((hi.x + 2.0, c.y, c.z))
cam.rotation_euler = Vector((-1, 0, 0)).to_track_quat("-Z", "Z").to_euler()
cd.clip_start = 2.0 + (hi.x - lo.x) * 0.30
sc.render.filepath = out + "_side_cut.png"
bpy.ops.render.render(write_still=True)
print("BBOX", [round(x, 3) for x in lo], [round(x, 3) for x in hi])
