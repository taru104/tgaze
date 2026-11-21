"""落とした .blend の中身を一覧にする（スクリプトは実行しない：libraries.load で読むだけ）。
blender --background --factory-startup --python inspect_blend.py -- FILE.blend [PREVIEW.png]"""
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
path = argv[0]
with bpy.data.libraries.load(path, link=False) as (src, dst):
    print("TEXTS", list(src.texts))
    print("SCENES", list(src.scenes))
    print("N_OBJECTS", len(src.objects))
    dst.objects = list(src.objects)
for o in bpy.data.objects:
    if o.users_collection == ():
        pass
    bpy.context.scene.collection.objects.link(o)
bpy.context.view_layer.update()
lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
for o in sorted(bpy.data.objects, key=lambda o: o.name):
    if o.type == "MESH":
        bb = [o.matrix_world @ Vector(c) for c in o.bound_box]
        mn = Vector([min(v[i] for v in bb) for i in range(3)])
        mx = Vector([max(v[i] for v in bb) for i in range(3)])
        lo = Vector([min(lo[i], mn[i]) for i in range(3)])
        hi = Vector([max(hi[i], mx[i]) for i in range(3)])
        print("OBJ", o.name, o.type, "verts", len(o.data.vertices), "mats", [m.name for m in o.data.materials if m],
              "min", [round(x, 3) for x in mn], "max", [round(x, 3) for x in mx], "parent", o.parent.name if o.parent else None)
    else:
        print("OBJ", o.name, o.type, "loc", [round(x, 3) for x in o.matrix_world.translation])
print("ALL_MIN", [round(x, 3) for x in lo], "ALL_MAX", [round(x, 3) for x in hi])
