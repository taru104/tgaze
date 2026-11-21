"""車の部品 1 つを、つながった塊（loose part）ごとに分けて、大きさ・形（輪かどうか）を一覧にする。スクリプトは実行しない。
blender --background --factory-startup --python inspect_wheelpart.py -- CAR.blend PART_NAME"""
import sys

import bmesh
import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
path, part = argv[0], argv[1]
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
with bpy.data.libraries.load(path, link=False) as (src, dst):
    dst.objects = [part]
ob = bpy.data.objects[part]
bpy.context.scene.collection.objects.link(ob)
bm = bmesh.new()
bm.from_mesh(ob.data)
bm.verts.ensure_lookup_table()
mw = np.array(ob.matrix_world)
seen = set()
parts = []
for v in bm.verts:
    if v.index in seen:
        continue
    stack, comp = [v], []
    seen.add(v.index)
    while stack:
        x = stack.pop()
        comp.append(x.index)
        for e in x.link_edges:
            y = e.other_vert(x)
            if y.index not in seen:
                seen.add(y.index)
                stack.append(y)
    parts.append(comp)
co = np.array([v.co for v in bm.verts])
co = co @ mw[:3, :3].T + mw[:3, 3]
mats = [ob.material_slots[i].material.name if ob.material_slots[i].material else None for i in range(len(ob.material_slots))]
for k, comp in enumerate(sorted(parts, key=len, reverse=True)):
    p = co[comp]
    c = p.mean(0)
    w, U = np.linalg.eigh((p - c).T @ (p - c))
    n = U[:, 0]
    rel = p - c
    r = np.linalg.norm(rel - np.outer(rel @ n, n), axis=1)
    fm = {}
    for f in bm.faces:
        if f.verts[0].index in set(comp[:1]) or False:
            pass
    print("LOOSE", k, "nverts", len(comp), "center", c.round(3).tolist(), "size", (p.max(0) - p.min(0)).round(3).tolist(),
          "normal", n.round(3).tolist(), "r_min", round(float(r.min()), 3), "r_max", round(float(r.max()), 3))
    if k >= 25:
        break
print("MATS", mats)
