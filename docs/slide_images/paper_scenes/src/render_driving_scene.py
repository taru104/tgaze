"""発表スライド用：運転中の視線の撮り方を 3DCG で再現したイメージ（論文の図の写しではない。学習には使わない）。
blender --background --python render_driving_scene.py -- OUT_DIR [RES_X RES_Y SAMPLES] [--nogear]
左ハンドルの車内。運転者は座ってハンドルを握り、前の道路を見る。頭に着ける視線計測器とカメラ。
車のカメラ 2 台：front = フロントガラスの下の中央（道路向き）、back = ダッシュボード中央の画面のあたり（運転者向き）。
向きは仮（元の論文が分かったら直す）。書き出しは助手席側から運転者を見る 1 枚。"""
import json
import math
import os
import sys

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

PROJ = os.path.expanduser("~/mpfb2-gaze-generator-poc")
sys.path.insert(0, os.path.join(PROJ, "scripts"))
os.chdir(PROJ)
import create_macbook_gaze_poc as mbk  # noqa: E402
import create_full_face_poc as p1  # noqa: E402
import gaze_validation as gv  # noqa: E402
import generate_gaze_dataset_1000 as g1  # noqa: E402
import generate_gaze_dataset_v3 as v3  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
NOGEAR = "--nogear" in args
HANDCHECK = "--handcheck" in args      # 手のアップ（助手席側から・真上から）も描く
SKIPMAIN = "--skipmain" in args        # 全体の絵は描かない（手の確認だけ）
args = [x for x in args if not x.startswith("--")]
OUT_DIR = args[0]
RES_X = int(args[1]) if len(args) > 1 else 1920
RES_Y = int(args[2]) if len(args) > 2 else 1080
SAMPLES = int(args[3]) if len(args) > 3 else 128
os.makedirs(OUT_DIR, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
PERSON = {"identity": 250, "clothes": "female_casualsuit02", "shoes": "shoes02"}
f64 = v3.v2.f64
upd = bpy.context.view_layer.update

# ---------------------------------------------------------------- 人
cfg = json.load(open(os.path.join(PROJ, "configs", "gaze_dataset_v2.json")))
_orig_build_human = p1.build_human


def build_human_with_clothes(shape_fn=None):
    p1.MHCLO_ASSETS.append(("clothes", f"{PERSON['clothes']}.mhclo", "Clothes"))
    p1.MHCLO_ASSETS.append(("clothes", f"{PERSON['shoes']}.mhclo", "Shoes"))
    return _orig_build_human(shape_fn)


p1.build_human = build_human_with_clothes
v3.GeneratorV3.set_lighting = lambda self, L: None
gen = v3.GeneratorV3(cfg, PERSON["identity"], 1, os.path.join(OUT_DIR, "_scene_tmp"), "train")
gen.build()
cfg = gen.cfg
scene, rig, eyes, eye_info = gen.scene, gen.rig, gen.eyes, gen.eye_info
mb_root = bpy.data.objects["MacBookAirM4"]
for o in list(mb_root.children_recursive) + [mb_root]:
    bpy.data.objects.remove(o)
for n in ("KeyLight", "FillLight"):
    if n in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[n])
mbk.add_mpfb_eye_ik(rig)
upd()


def rot_bone(name, axis, deg):
    pb = rig.pose.bones[name]
    upd()
    m = pb.matrix.copy()
    h = m.translation.copy()
    pb.matrix = Matrix.Translation(h) @ Matrix.Rotation(math.radians(deg), 4, axis) @ Matrix.Translation(-h) @ m
    upd()


# 運転席に座る：太ももは少し上向きに前へ、すねは前へ伸ばしてペダルへ。背もたれに合わせて上体を少し後ろへ（人は -Y を向く。人の右 = -X）
for s, sgn in (("L", 1), ("R", -1)):
    rot_bone(f"upperleg01.{s}", "X", float(os.environ.get("THIGH_DEG", "-86")))   # 太もも：水平より少し下向き（ハンドルの下を通す）
    rot_bone(f"upperleg01.{s}", "Z", sgn * float(os.environ.get("LEG_SPREAD", "0")))   # 足の開き（足元が狭いので小さく）
    rot_bone(f"lowerleg01.{s}", "X", float(os.environ.get("SHIN_DEG", "40")))   # すね：前へ出してペダルの方へ
rot_bone("spine05", "X", float(os.environ.get("RECLINE_DEG", "-18")))   # 上体を背もたれに合わせてうしろへ
rot_bone("neck01", "X", float(os.environ.get("NECK_DEG", "12")))      # 首で前を向き直す
bw = lambda n: rig.matrix_world @ rig.pose.bones[n].head  # noqa: E731


def eval_world_verts(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    mw = np.array(ev.matrix_world)
    v = np.array([vv.co for vv in me.vertices])
    ev.to_mesh_clear()
    return v @ mw[:3, :3].T + mw[:3, 3]


shoes = next(o for o in bpy.data.objects if o.name.endswith("." + PERSON["shoes"]))
sv = eval_world_verts(shoes)
FLOOR = float(sv[:, 2].min()) - 0.01
hipz = float(min(bw(f"upperleg02.{s}").z for s in "LR"))
SEAT = hipz - 0.085
pel = bw("root")


def eye_c_now():
    m = mbk.measure(rig, eyes, eye_info)
    return {s: f64(m[s][0]) for s in m}


ec0 = eye_c_now()
cyc0 = (ec0["left"] + ec0["right"]) / 2
cl = (bw("clavicle.L") + bw("clavicle.R")) / 2

# ---------------------------------------------------------------- 材質・形
def principled(name, rgb, rough=0.5, metal=0.0, emit=0.0, transmission=0.0):
    mt = bpy.data.materials.new(name)
    mt.use_nodes = True
    b = mt.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit:
        b.inputs["Emission Color"].default_value = (*rgb, 1)
        b.inputs["Emission Strength"].default_value = emit
    if transmission:
        b.inputs["Transmission Weight"].default_value = transmission
        b.inputs["IOR"].default_value = 1.45
        mt.use_backface_culling = False
    return mt


def emission_only(name, rgb, strength=1.0):
    mt = bpy.data.materials.new(name)
    mt.use_nodes = True
    n = mt.node_tree.nodes
    n.remove(n["Principled BSDF"])
    e = n.new("ShaderNodeEmission")
    e.inputs["Color"].default_value, e.inputs["Strength"].default_value = (*rgb, 1), strength
    mt.node_tree.links.new(e.outputs["Emission"], n["Material Output"].inputs["Surface"])
    return mt


def add_mesh(name, verts, faces, mt=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], faces)
    ob = bpy.data.objects.new(name, me)
    scene.collection.objects.link(ob)
    if mt is not None:
        me.materials.append(mt)
    return ob


def box(name, x0, x1, y0, y1, z0, z1, mt):
    v = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return add_mesh(name, v, f, mt)


def oriented_box(name, center, ax, ay, az, sx, sy, sz, mt):
    c = np.array(center, dtype=float)
    v = [c + ax * x * sx / 2 + ay * y * sy / 2 + az * z * sz / 2 for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return add_mesh(name, v, f, mt)


def tube(name, pts, radius, mt, closed=False):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 3
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for pt, p in zip(sp.points, pts):
        pt.co = (*map(float, p), 1.0)
    sp.use_cyclic_u = closed
    ob = bpy.data.objects.new(name, cu)
    scene.collection.objects.link(ob)
    cu.materials.append(mt)
    return ob


def cylinder_between(name, p0, p1, r, mt, verts=24):
    p0, p1 = Vector(tuple(map(float, p0))), Vector(tuple(map(float, p1)))
    d = p1 - p0
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=d.length, location=(p0 + p1) / 2)
    ob = bpy.context.active_object
    ob.name = name
    ob.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    ob.data.materials.append(mt)
    return ob


def frame_axes(n, up_hint=(0, 0, 1.0)):
    n = np.array(n, dtype=float)
    n /= np.linalg.norm(n)
    ax = np.cross(up_hint, n)
    ax /= np.linalg.norm(ax)
    ay = np.cross(n, ax)
    return ax, ay, n


dark = principled("InteriorDark", (0.05, 0.05, 0.055), 0.6)
trim = principled("InteriorTrim", (0.22, 0.22, 0.24), 0.5)
seat_m = principled("SeatFabric", (0.30, 0.22, 0.16), 0.85)          # 茶色の布シート
body_paint = principled("BodyPaint", (0.75, 0.78, 0.80), 0.3, 0.6)   # 銀色の車体
glass = principled("Glass", (0.92, 0.96, 0.98), 0.0, 0.0, 0.0, 1.0)
glass.node_tree.nodes["Principled BSDF"].inputs["IOR"].default_value = 1.02   # 映り込みを弱くした薄いガラス
plastic = principled("GearBlack", (0.015, 0.015, 0.018), 0.35)
camgrey = principled("GearGrey", (0.18, 0.18, 0.20), 0.4, 0.5)
lens_m = principled("Lens", (0.02, 0.03, 0.05), 0.05)
carcam_m = principled("CarCamBody", (0.55, 0.56, 0.58), 0.35, 0.6)   # 黒いダッシュボードの上で見分けやすい明るいグレー
carcam_ring = principled("CarCamRing", (0.85, 0.86, 0.88), 0.25, 1.0)

# ---------------------------------------------------------------- 車：既存のモデル（SHC MCE 1、denniswoo1993、CGTrader、Royalty Free License（no AI））
# スライドの絵にだけ使う。学習データの生成には使わない。モデルのファイルはリポジトリに入れない（~/assets/car/ に置いたものを読む）。
# .blend の中のスクリプトは実行しない（libraries.load で部品だけを読む）。
import bmesh  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402

CAR_BLEND = os.environ.get("CAR_BLEND", os.path.expanduser("~/assets/car/extracted/SHC MCE1/SHC MCE1 [blender 2.8 only].blend"))
CAR_DIR = os.path.dirname(CAR_BLEND)
WHEEL_PART, DRIVER_SEAT_PART, PASSENGER_SEAT_PART, ROOF_PART = "Compact Car Part.032", "Compact Car Part.041", "Compact Car Part.039", "Compact Car Part.092"
WHEEL_OUTER_D = 0.37
CAR_INFO = {"model": "BLENDER EEVEE Brandless Small 4 Door Hatchback（車名 SHC MCE 1）", "author": "denniswoo1993",
            "url": "https://www.cgtrader.com/free-3d-models/car/car/blender-eevee-brandless-small-4-door-hatchback",
            "license": "CGTrader Royalty Free License（no AI）。ページの表示は Free for personal and commercial use",
            "use": "発表スライドの絵にだけ使う。学習データの生成・学習には使わない。モデルのファイルは公開リポジトリに置かない",
            "file": "SHC MCE1 [blender 2.8 only].blend（~/assets/car/、リポジトリの外）",
            "sha256_of_rar": "8074e3317ee72953c88a616543f1ab412fff50320d0683ef5da3442b2543c918"}
with bpy.data.libraries.load(CAR_BLEND, link=False) as (src, dst):
    dst.objects = [n for n in src.objects if n.startswith(("Compact Car Part", "Wheel"))]
car_root = bpy.data.objects.new("CarRoot", None)
scene.collection.objects.link(car_root)
CAR_PARTS = []
for ob in bpy.data.objects:
    if ob.name.startswith(("Compact Car Part", "Wheel")) and ob.users_scene == ():
        scene.collection.objects.link(ob)
        if ob.parent is None:
            ob.parent = car_root
        CAR_PARTS.append(ob)
for img in bpy.data.images:          # テクスチャは .blend と同じフォルダにある
    if img.filepath and not os.path.exists(bpy.path.abspath(img.filepath)):
        cand = os.path.join(CAR_DIR, os.path.basename(img.filepath.replace("\\", "/")))
        if os.path.exists(cand):
            img.filepath = cand
upd()


def part_world_verts(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    mw = np.array(ev.matrix_world)
    v = np.array([vv.co for vv in me.vertices])
    polys = [list(p.vertices) for p in me.polygons]
    mats = [p.material_index for p in me.polygons]
    ev.to_mesh_clear()
    return (v @ mw[:3, :3].T + mw[:3, 3]) if len(v) else v, polys, mats


# (1) ハンドルの縁を、ハンドルの部品の中の「輪の形の塊」として探す（車の元の座標で）
wpart = bpy.data.objects[WHEEL_PART]
wv0, wpolys, _ = part_world_verts(wpart)
bm = bmesh.new()
bm.from_mesh(wpart.data)                 # 元のメッシュのつながり（モディファイアなし：頂点の番号は wv0 と同じ）
bm.verts.ensure_lookup_table()
assert len(bm.verts) == len(wv0), (len(bm.verts), len(wv0))
seen, comps = set(), []
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
    comps.append(comp)
bm.free()
best = None
for comp in comps:
    if len(comp) < 50:
        continue
    p = wv0[comp]
    c0 = p.mean(0)
    _, U = np.linalg.eigh((p - c0).T @ (p - c0))
    n0 = U[:, 0]
    rel = p - c0
    r = np.linalg.norm(rel - np.outer(rel @ n0, n0), axis=1)
    if r.min() > 0.75 * r.max() and (best is None or r.max() > best[3]):
        best = (c0, n0, float(r.min()), float(r.max()), comp)
rim_c0, rim_n0, rim_rin0, rim_rout0, RIM_COMP = best
# ハンドルの真ん中のエンブレム（飾り）を無地にする：縁の中心から縁の半径の 42% 以内にある、光沢の黒い飾り（Glossy Black Trim）と
# 色の付いた面（FrontColor）を、つや消しの黒い無地に替える
plain = principled("EmblemPlain", (0.03, 0.03, 0.032), 0.6)
wpart.data.materials.append(plain)
_mi = len(wpart.data.materials) - 1
EMBLEM_FACES = 0
for poly in wpart.data.polygons:
    mt = wpart.material_slots[poly.material_index].material
    if mt is None or not mt.name.startswith(("Glossy Black Trim", "FrontColor")):
        continue
    c_ = wv0[list(poly.vertices)].mean(0)
    rel_ = c_ - rim_c0
    if float(np.linalg.norm(rel_ - (rel_ @ rim_n0) * rim_n0)) <= 0.42 * rim_rout0:
        poly.material_index = _mi
        EMBLEM_FACES += 1
print("EMBLEM faces made plain:", EMBLEM_FACES)
S_WHEEL = WHEEL_OUTER_D / (2 * rim_rout0)          # ハンドルの外径を 37 cm にする縮尺
CAR_SCALE = S_WHEEL
R_CAR = np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])      # Z まわりに 180°：車の前（元の +Y）を人の前（-Y）へ
# 左ハンドルか：元の車の座標で、ハンドルが車の中心より「前を向いた人の左」（元の -X）にあるか
_allv = np.concatenate([part_world_verts(o)[0] for o in CAR_PARTS if o.type == "MESH"])
car_cx0 = float((_allv[:, 0].min() + _allv[:, 0].max()) / 2)
LEFT_HAND_DRIVE = bool(rim_c0[0] < car_cx0)


def car_xform(p, t):
    return CAR_SCALE * (np.asarray(p) @ R_CAR.T) + t


# (2) 運転席の形を光線で測る（座面の上の高さ・背もたれの前の面）。車は縮めて回しただけ（t = 0）の座標
seat0, seat_polys, _ = part_world_verts(bpy.data.objects[DRIVER_SEAT_PART])
roof0, roof_polys, _ = part_world_verts(bpy.data.objects[ROOF_PART])


SEAT_SHRINK = float(os.environ.get("SEAT_SHRINK", "0.88"))   # 運転席だけ小さく（車を頭が入る大きさに拡大したぶん、座面の奥行きが長すぎるため）
SEAT_PIVOT = None                                              # 縮める中心（t = 0 の座標）：背もたれの前の面と座面の上が交わる所


def measure_seat():
    global seat_s, seat_bvh, scx, sz0, sz1, sy0, sy1
    seat_s = car_xform(seat0, np.zeros(3))
    if SEAT_PIVOT is not None:
        seat_s = SEAT_PIVOT + (seat_s - SEAT_PIVOT) * SEAT_SHRINK
    seat_bvh = BVHTree.FromPolygons([Vector(tuple(p)) for p in seat_s], seat_polys, all_triangles=False)
    scx = float((seat_s[:, 0].min() + seat_s[:, 0].max()) / 2)
    sz0, sz1 = float(seat_s[:, 2].min()), float(seat_s[:, 2].max())
    sy0, sy1 = float(seat_s[:, 1].min()), float(seat_s[:, 1].max())


def seat_top(y):          # 座面の上（x = 座席の真ん中、前後 y）
    hit = seat_bvh.ray_cast(Vector((scx, y, sz1 + 1.0)), Vector((0, 0, -1)))
    return hit[0].z if hit[0] is not None else None


def seat_back_front(z):   # 背もたれの前の面（高さ z）。人の前は -Y なので、前から +Y 向きに光線を当てる
    hit = seat_bvh.ray_cast(Vector((scx, sy0 - 1.0, z)), Vector((0, 1, 0)))
    return hit[0].y if hit[0] is not None else None


def cushion_and_roof():
    """座面（お尻が乗る後ろ寄り）の高さと、その真上の天井の内側の高さ（t = 0 の座標）。"""
    bfl = seat_back_front(sz0 + 0.55 * (sz1 - sz0))
    cus = [(y, seat_top(y)) for y in np.arange(sy0 + 0.03, bfl - 0.02, 0.01)]
    cus = [(y, z) for y, z in cus if z is not None]
    rear = [(y, z) for y, z in cus if y > sy0 + 0.65 * (bfl - sy0)]
    ct = float(np.median([z for _, z in rear])) if rear else float(max(z for _, z in cus))
    yr = float(np.mean([y for y, _ in rear])) if rear else (sy0 + bfl) / 2
    roof_s = car_xform(roof0, np.zeros(3))
    rb = BVHTree.FromPolygons([Vector(tuple(p)) for p in roof_s], roof_polys, all_triangles=False)
    h = rb.ray_cast(Vector((scx, yr, ct + 0.05)), Vector((0, 0, 1)))
    return bfl, ct, (h[0].z if h[0] is not None else None)


# (3) 座った体の形（お尻・太ももの裏の低さ、背中のうしろの面）
_bv = np.concatenate([eval_world_verts(o) for o in bpy.data.objects if o.type == "MESH" and o.name.startswith("Human") and o.visible_get()])
_py = float(pel.y)
_cen = np.abs(_bv[:, 0]) < 0.15
butt = _bv[_cen & (_bv[:, 1] > _py - 0.22) & (_bv[:, 1] < _py + 0.10)]
butt_z = float(butt[:, 2].min())
back_bins = []
for zb in np.arange(butt_z + 0.15, butt_z + 0.60, 0.04):
    m_ = _cen & (_bv[:, 2] >= zb) & (_bv[:, 2] < zb + 0.04)
    if m_.sum():
        back_bins.append((zb + 0.02, float(_bv[m_][:, 1].max())))
# 頭が天井に当たらない縮尺：座面から天井までが「座った体の頭の高さ + 余裕 HEAD_CLEAR」になるまで車を大きくする
HEAD_CLEAR = 0.04
head_h = float(_bv[:, 2].max()) - butt_z                     # 座面（お尻の下）から頭（髪・頭の機材を含む）のいちばん上まで
measure_seat()
bf_low, cushion_top, roof_z = cushion_and_roof()
S_HEAD = CAR_SCALE * (head_h + 0.008 + HEAD_CLEAR) / (roof_z - cushion_top) if roof_z else CAR_SCALE
if S_HEAD > CAR_SCALE:
    CAR_SCALE = S_HEAD
    measure_seat()
    bf_low, cushion_top, roof_z = cushion_and_roof()
WHEEL_SHRINK = S_WHEEL / CAR_SCALE                           # ハンドルの部品だけ、外径 37 cm に戻す
SEAT_PIVOT = np.array([scx, bf_low, cushion_top])
measure_seat()
bf_low, cushion_top, roof_z = cushion_and_roof()
# (4) 車の位置 t：座面が体に 8 mm、背もたれが背中に 10 mm まで近づく位置（めり込まない）。人は動かさない
tx = 0.0 - scx
tz = butt_z - 0.008 - cushion_top
ty = _py + 0.05 - (sy0 + bf_low) / 2
for _ in range(4):
    need_ty = [bb + 0.010 - f for zc, bb in back_bins for f in [seat_back_front(zc - tz)] if f is not None and f > bf_low - 0.08]
    if need_ty:
        ty = max(need_ty)
T_CAR = np.array([tx, ty, tz])
M = np.eye(4)
M[:3, :3] = CAR_SCALE * R_CAR
M[:3, 3] = T_CAR
car_root.matrix_world = Matrix(M.tolist())
upd()
# 運転席を前へずらす（実際の車と同じ）：座席と人はそのまま、座席以外の車を人の方（+Y）へ動かし、ハンドルを手の届く所へ
_wc = car_xform(rim_c0, T_CAR)
_cl = (bw("clavicle.L") + bw("clavicle.R")) / 2
SEAT_SLIDE = float(os.environ.get("SEAT_SLIDE", str(max(0.0, (float(_cl.y) - 0.50) - float(_wc[1])))))
_seat_ob = bpy.data.objects[DRIVER_SEAT_PART]
_sp = SEAT_PIVOT + T_CAR                                     # 運転席を縮める中心（世界の座標）
_seat_ob.matrix_world = Matrix.Translation(Vector(tuple(map(float, _sp)))) @ Matrix.Scale(SEAT_SHRINK, 4) @ Matrix.Translation(Vector(tuple(map(float, -_sp)))) @ _seat_ob.matrix_world
upd()
for ob in CAR_PARTS:
    if ob is not _seat_ob:
        ob.matrix_world = Matrix.Translation(Vector((0.0, SEAT_SLIDE, 0.0))) @ ob.matrix_world
upd()
_wc = _wc + np.array([0.0, SEAT_SLIDE, 0.0])
wpart.matrix_world = Matrix.Translation(Vector(tuple(map(float, _wc)))) @ Matrix.Scale(WHEEL_SHRINK, 4) @ Matrix.Translation(Vector(tuple(map(float, -_wc)))) @ wpart.matrix_world
upd()
# (5) ハンドル：縁の円（中心・向き・半径・太さ）。向きは運転者の方
wheel_c = car_xform(rim_c0, T_CAR) + np.array([0.0, SEAT_SLIDE, 0.0])
wheel_n = R_CAR @ rim_n0
wheel_n /= np.linalg.norm(wheel_n)
if wheel_n @ (cyc0 - wheel_c) < 0:
    wheel_n = -wheel_n
wx_, wy_, _ = frame_axes(wheel_n)              # wx_ = 人の右（-X）、wy_ = ハンドル面の上
WR = S_WHEEL * (rim_rin0 + rim_rout0) / 2
RIM_R = S_WHEEL * (rim_rout0 - rim_rin0) / 2
wheel_obj = wpart
WHEEL_INFO = {"model": "車のモデルのハンドル（" + WHEEL_PART + " の中の輪の形の塊）", "outer_diameter_m": round(2 * S_WHEEL * rim_rout0, 4),
              "wheel_part_shrink_relative_to_car": round(WHEEL_SHRINK, 4),
              "emblem_made_plain": {"faces": EMBLEM_FACES, "how": "ハンドルの真ん中（縁の半径の 42% 以内）の光沢の黒い飾りと色の付いた面を、つや消しの黒い無地に替えた"},
              "rim_center_radius_m": round(WR, 4), "rim_tube_radius_m": round(RIM_R, 4), "center_world_m": [round(float(x), 4) for x in wheel_c],
              "normal_world": [round(float(x), 4) for x in wheel_n]}
CAR_INFO.update({"driver_seat_slid_forward_m": round(SEAT_SLIDE, 4), "driver_seat_shrink": SEAT_SHRINK, "scale": round(CAR_SCALE, 4), "scale_if_wheel_37cm": round(S_WHEEL, 4), "scale_for_headroom": round(S_HEAD, 4),
                 "headroom_note": "ハンドルの外径 37 cm に合わせた縮尺では、座面から天井までが座った人の頭の高さより低い（頭が天井に入る）。"
                                  "そこで車は頭が入る大きさまで拡大し、ハンドルの部品だけを外径 37 cm に縮めた",
                 "seat_cushion_to_roof_m": round(float(roof_z - cushion_top), 4) if roof_z else None, "seated_head_height_m": round(head_h, 4), "rotation_z_deg": 180, "translation_m": [round(float(x), 4) for x in T_CAR],
                 "left_hand_drive_as_modeled": LEFT_HAND_DRIVE, "mirrored": False,
                 "length_m_after_scale": round(float(CAR_SCALE * (_allv[:, 1].max() - _allv[:, 1].min())), 3),
                 "height_m_after_scale": round(float(CAR_SCALE * (_allv[:, 2].max() - _allv[:, 2].min())), 3)})
if not LEFT_HAND_DRIVE:
    raise SystemExit("右ハンドルのモデル：左右反転が要る（このスクリプトはまだ反転していない）")
SEAT_INFO = {"driver_seat_part": DRIVER_SEAT_PART, "butt_bottom_z_m": round(butt_z, 4), "cushion_top_z_m": round(cushion_top + tz, 4), "gap_seat_top_to_body_m": 0.008, "gap_backrest_to_back_m": 0.010}
print("CAR", json.dumps(CAR_INFO, ensure_ascii=False), json.dumps(WHEEL_INFO))
# 車の大きさ・位置（世界の座標）
_allw = car_xform(_allv, T_CAR)
CAR_GROUND_Z = float(min(part_world_verts(o)[0][:, 2].min() for o in CAR_PARTS if o.name.startswith("Wheel")))   # 置いたあとの世界の座標
CX = float((_allw[:, 0].min() + _allw[:, 0].max()) / 2)          # 車の中心の x
ROOF = float(_allw[:, 2].max()) - 0.06
_ps = part_world_verts(bpy.data.objects[PASSENGER_SEAT_PART])[0]                 # 置いたあとの世界の座標
PCX = float((_ps[:, 0].min() + _ps[:, 0].max()) / 2)              # 助手席の真ん中の x

# ---------------------------------------------------------------- 外：道路・草地・空
road = principled("Asphalt", (0.17, 0.17, 0.18), 0.85)
grass = principled("Grass", (0.24, 0.40, 0.18), 0.9)
paint = principled("LinePaint", (0.95, 0.95, 0.92), 0.6)
gz = CAR_GROUND_Z                     # 道路の面 = 車のタイヤの下
box("Road", -5.0, 3.0, -400, 30, gz - 0.02, gz, road)
box("GrassL", 3.0, 60, -400, 30, gz - 0.03, gz - 0.005, grass)
box("GrassR", -60, -5.0, -400, 30, gz - 0.03, gz - 0.005, grass)
for y in np.arange(20, -400, -12):
    box("LaneDash", -1.0 - 0.07, -1.0 + 0.07, y - 4, y, gz, gz + 0.004, paint)       # 中央の破線（運転者の右側）
box("EdgeLine", 2.6, 2.75, -400, 30, gz, gz + 0.004, paint)
rng = np.random.default_rng(5)
treem = principled("Tree", (0.16, 0.32, 0.14), 0.9)
for y in np.arange(-30, -400, -22):
    for side in (1, -1):
        x = side * (9 + rng.uniform(0, 6))
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=float(rng.uniform(2.0, 3.2)), location=(x, float(y + rng.uniform(-5, 5)), gz + 3.0))
        bpy.context.active_object.data.materials.append(treem)

# ---------------------------------------------------------------- 腕と手：左手は 10 時、右手は 2 時の縁を握る
# 手順：(1) 腕は IK（肘のポールは肘の下・外側）で手首を置き、FK に焼き付ける
#       (2) 手の向き：手の長さ方向 h は前腕の向きを縁の接線に垂直な面へ写したもの（手首の曲がりを小さく）、
#           手のひら p は縁の方（ハンドルの内側・奥）を向き、手の甲は運転者側の外向き。縁は指の付け根の手のひら側に来る
#       (3) 前腕のねじれは lowerarm01 / lowerarm02 に半分ずつ、残り（振り）だけを wrist に入れる
#       (4) 4 本の指は付け根・真ん中・先の 3 関節を、指先が縁に近づくまで少しずつ曲げる（めり込まない距離で止める）
#       (5) 親指は縁の運転者側に沿わせて軽く伸ばす
from mathutils import Quaternion  # noqa: E402

HAND_LOG = {}


def P(n):
    return np.array(rig.matrix_world @ rig.pose.bones[n].head)


def T(n):
    return np.array(rig.matrix_world @ rig.pose.bones[n].tail)


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def rot_world(name, axis, deg):
    """骨 name を、その付け根を中心に、ワールドの軸ベクトル axis まわりに deg 度回す（子も一緒に回る）。"""
    pb = rig.pose.bones[name]
    upd()
    m = pb.matrix.copy()
    h = m.translation.copy()
    pb.matrix = Matrix.Translation(h) @ Matrix.Rotation(math.radians(deg), 4, Vector(tuple(map(float, unit(axis))))) @ Matrix.Translation(-h) @ m
    upd()


def rim_dist(p):
    """点 p からハンドルの縁（管の中心の円）までの距離。"""
    v = p - wheel_c
    vp = v - (v @ wheel_n) * wheel_n
    q = wheel_c + WR * unit(vp)
    return float(np.linalg.norm(p - q))


# ハンドルの表面までの距離（読み込んだモデルの形そのものに対して）
from mathutils.bvhtree import BVHTree as _BVH  # noqa: E402
_wv = [wheel_obj.matrix_world @ v.co for v in wheel_obj.data.vertices]
_wheel_bvh = _BVH.FromPolygons(_wv, [list(pp.vertices) for pp in wheel_obj.data.polygons], all_triangles=False)
_wv_np = np.array([tuple(v) for v in _wv])


def wheel_dist(p):
    hit = _wheel_bvh.find_nearest(Vector(tuple(map(float, p))))
    return float(hit[3]) if hit[0] is not None else 1.0


def rim_section(rdir):
    """rdir（ハンドルの中心から外向き）の向きにある縁の断面を、モデルの頂点から測る：断面の中心と半径。"""
    rel = _wv_np - wheel_c
    inpl = rel - np.outer(rel @ wheel_n, wheel_n)
    r_ = np.linalg.norm(inpl, axis=1)
    cosang = (inpl @ rdir) / np.maximum(r_, 1e-9)
    sel = (cosang > math.cos(math.radians(5))) & (r_ > 0.6 * WR)
    pts = _wv_np[sel]
    rr = pts - wheel_c
    ca = np.stack([rr @ rdir, rr @ wheel_n], 1)              # 断面の座標（外向き, 運転者向き）
    cen2 = (ca.max(0) + ca.min(0)) / 2
    rad_ = float(((ca.max(0) - ca.min(0)) / 2).mean())
    return wheel_c + rdir * cen2[0] + wheel_n * cen2[1], rad_, (ca.max(0) - ca.min(0)).tolist()


# 休みの姿勢での手の形（骨の位置から）：手の長さ方向・親指側・手のひらの向き
rest = {}
for s in ("L", "R"):
    b = rig.data.bones
    wh = np.array(rig.matrix_world @ b[f"wrist.{s}"].head_local)
    mid = np.array(rig.matrix_world @ b[f"finger3-1.{s}"].head_local)
    idx = np.array(rig.matrix_world @ b[f"finger2-1.{s}"].head_local)
    lit = np.array(rig.matrix_world @ b[f"finger5-1.{s}"].head_local)
    h_r = unit(mid - wh)
    th = idx - lit
    th_r = unit(th - (th @ h_r) * h_r)
    p_r = np.cross(h_r, th_r) if s == "L" else np.cross(th_r, h_r)       # 左手：親指 = p×h、右手：親指 = h×p
    rest[s] = {"h": h_r, "p": unit(p_r), "dK": float(np.linalg.norm(mid - wh)),
               "R": np.array(rig.matrix_world.to_3x3() @ b[f"wrist.{s}"].matrix_local.to_3x3())}

ARM = ["upperarm01", "upperarm02", "lowerarm01", "lowerarm02"]
grip_pts = {}
for s, sgn in (("L", 1), ("R", -1)):
    ang = math.radians(60)                                    # 12 時から 60°：左手 10 時、右手 2 時
    rdir = math.cos(ang) * wy_ - sgn * math.sin(ang) * wx_    # 縁の点への向き（ハンドルの中心から外向き）。wx_ は人の右（-X）
    g, RIM_LOC, SEC_WH = rim_section(rdir)                   # 握る所の縁の断面（中心・半径）をモデルから測る
    grip_pts[s] = g
    t = unit(np.cross(wheel_n, rdir))                         # 縁の接線
    sh = bw(f"upperarm01.{s}")
    te = bpy.data.objects.new(f"HandTarget.{s}", None)
    scene.collection.objects.link(te)
    pe = bpy.data.objects.new(f"ElbowPole.{s}", None)
    scene.collection.objects.link(pe)
    pe.location = Vector((float(sh.x) + sgn * 0.30, float(sh.y) - 0.05, float(sh.z) - 0.60))     # 肘の下・外側
    c_ = rig.pose.bones[f"lowerarm02.{s}"].constraints.new("IK")
    c_.target, c_.pole_target, c_.chain_count, c_.pole_angle, c_.iterations = te, pe, 4, math.radians(-90), 3000
    # 手首の位置を、手の向きと一緒に少しずつ合わせる
    W = g + wheel_n * 0.08 - rdir * 0.02
    for it in range(6):
        te.location = Vector(tuple(map(float, W)))
        upd()
        f = unit(T(f"lowerarm02.{s}") - P(f"lowerarm01.{s}"))              # 前腕の向き
        h = unit(f - 0.6 * (f @ t) * t)                                    # 前腕の向きから縁の接線方向の成分を 6 割除く（縁は手のひらを少し斜めに横切る）
        p = unit(np.cross(t, h) - (np.cross(t, h) @ h) * h)
        if p @ (rdir * 0.75 + wheel_n * 0.66) > 0:                         # 手のひらは内側・奥（手の甲が運転者側の外）
            p = -p
        # 縁の管の中心 = 指の付け根の少し手前 + 手のひら側へ（縁の半径 + 手の厚みの半分）
        W = g - p * (RIM_LOC + 0.033) - h * (rest[s]["dK"] - 0.006)        # 手のひらの厚みの半分 + 余裕
    te.location = Vector(tuple(map(float, W)))
    upd()
    # IK を FK に焼き付ける
    mats = {n: rig.pose.bones[f"{n}.{s}"].matrix.copy() for n in ARM}
    rig.pose.bones[f"lowerarm02.{s}"].constraints.remove(c_)
    upd()
    for n in ARM:
        rig.pose.bones[f"{n}.{s}"].matrix = mats[n]
        upd()
    # 手の目標の回転（ワールド）：休みの姿勢の手の座標 [h, p, h×p] を目標の [h, p, h×p] へ
    B_des = np.stack([h, p, np.cross(h, p)], 1)
    B_rest = np.stack([rest[s]["h"], rest[s]["p"], np.cross(rest[s]["h"], rest[s]["p"])], 1)
    R_des = B_des @ B_rest.T @ rest[s]["R"]
    # 前腕のねじれを分ける：今の手首の向き R_cur から R_des までの回転を、手首の骨の軸（Y）まわりのねじれと振りに分解
    R_cur = np.array(rig.pose.bones[f"wrist.{s}"].matrix.to_3x3())
    X = Matrix(R_cur.T @ R_des).to_quaternion()
    swing, twist = X.to_swing_twist("Y")
    for n in ("lowerarm01", "lowerarm02"):
        ax_ = np.array(rig.pose.bones[f"{n}.{s}"].matrix.to_3x3().col[1])
        rot_world(f"{n}.{s}", ax_, math.degrees(twist) * 0.5)
    wpb = rig.pose.bones[f"wrist.{s}"]
    wpb.matrix = Matrix.Translation(wpb.matrix.translation) @ Matrix(R_des).to_4x4()
    upd()
    bend = math.degrees(math.acos(max(-1, min(1, float(unit(T(f"lowerarm02.{s}") - P(f"lowerarm01.{s}")) @ h)))))
    # 指：付け根・真ん中・先を、指先が縁に近づくまで少しずつ曲げる（曲げる軸 k = h × p、指先は手のひら側 p へ）
    k = np.cross(h, p)
    curls = {}
    for fi in (2, 3, 4, 5):
        tot = [0.0, 0.0, 0.0]
        segs = [f"finger{fi}-{j}.{s}" for j in (1, 2, 3)]
        FING_CLEAR = 0.0110                                     # 指の骨（指の中心）とハンドルの表面の距離の下限（指の厚みの半分 + 余裕）
        dist_now = lambda: min(min(wheel_dist(T(sg)), wheel_dist(P(sg))) for sg in segs[1:])  # noqa: E731
        for step in range(80):                                  # 2° ずつ曲げ、近づきすぎたら 1 段戻して止める
            moved = False
            for j, wgt in ((0, 1.0), (1, 1.15), (2, 0.8)):
                if tot[j] < (95, 105, 75)[j]:
                    rot_world(segs[j], k, 2.0 * wgt)
                    tot[j] += 2.0 * wgt
                    moved = True
            if not moved:
                break
            if dist_now() < FING_CLEAR:
                for j, wgt in ((2, 0.8), (1, 1.15), (0, 1.0)):
                    rot_world(segs[j], k, -2.0 * wgt)
                    tot[j] -= 2.0 * wgt
                break
        curls[f"finger{fi}"] = [round(x, 1) for x in tot]
    # 親指：縁の運転者側に沿わせる（付け根を縁の方へ少し寄せ、先は軽く伸ばす）
    # 親指：縁の運転者側に沿わせて軽く伸ばす。表面から 9.5 mm より近い間は、付け根・真ん中を 2° ずつ縁から離す（最大 30°）
    thumb_dist = lambda: min(min(wheel_dist(T(f"finger1-{j}.{s}")), wheel_dist(P(f"finger1-{j}.{s}"))) for j in (1, 2, 3))  # noqa: E731
    thumb_open = 0.0
    while thumb_dist() < 0.0085 and thumb_open < 30:
        rot_world(f"finger1-1.{s}", k, -2.0)
        rot_world(f"finger1-2.{s}", k, -1.0)
        thumb_open += 2.0
    tips = {fi: round(wheel_dist(T(f"finger{fi}-3.{s}")) * 1000, 1) for fi in (2, 3, 4, 5)}
    palm_c = (P(f"finger3-1.{s}") + P(f"wrist.{s}")) / 2
    HAND_LOG[s] = {"grip_point_m": [round(float(x), 4) for x in g], "clock": "10 時" if s == "L" else "2 時",
                   "wrist_bend_deg(forearm_vs_hand)": round(bend, 1), "forearm_twist_deg_total": round(math.degrees(twist), 1),
                   "forearm_twist_split": "lowerarm01 と lowerarm02 に半分ずつ", "finger_curl_deg_[base,middle,tip]": curls,
                   "fingertip_bone_to_wheel_surface_mm": tips, "thumb_opened_deg": thumb_open, "thumb_bone_to_wheel_surface_mm": round(thumb_dist() * 1000, 1), "rim_section_at_grip_mm[outward,toward_driver]": [round(x * 1000, 1) for x in SEC_WH],
                   "knuckle_bone_to_wheel_surface_mm": round(wheel_dist(P(f"finger3-1.{s}")) * 1000, 1),
                   "palm_center_to_wheel_surface_mm": round(wheel_dist(palm_c) * 1000, 1),
                   "elbow_m": [round(float(x), 3) for x in P(f"lowerarm01.{s}")], "shoulder_m": [round(float(x), 3) for x in P(f"upperarm01.{s}")]}
    print("HAND", s, json.dumps(HAND_LOG[s], ensure_ascii=False))

# ---------------------------------------------------------------- めり込みの確認（三角形どうしの交差を BVH で数える）
from mathutils.bvhtree import BVHTree  # noqa: E402


def bvh_of(objs):
    dg = bpy.context.evaluated_depsgraph_get()
    verts, polys = [], []
    for ob in objs:
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        mw = ev.matrix_world
        base = len(verts)
        verts += [mw @ v.co for v in me.vertices]
        polys += [[base + i for i in p.vertices] for p in me.polygons]
        ev.to_mesh_clear()
    bvh_of.last_polys = polys
    return BVHTree.FromPolygons(verts, polys, all_triangles=False), verts


def human_parts(names):
    return [o for o in bpy.data.objects if o.type == "MESH" and o.name.startswith("Human") and o.visible_get() and any(n in o.name for n in names)]


def min_dist(verts_a, bvh_b):
    best = 1e9
    for v in verts_a:
        hit = bvh_b.find_nearest(v)
        if hit[0] is not None:
            best = min(best, hit[3])
    return best


REGION_BONES = {"head": ["head", "neck03"], "torso": ["spine01", "spine02", "spine03", "spine04"], "arm": ["upperarm02.L", "upperarm02.R", "lowerarm01.L", "lowerarm01.R"],
                "hand": ["wrist.L", "wrist.R"], "thigh": ["upperleg02.L", "upperleg02.R"], "shin": ["lowerleg01.L", "lowerleg01.R", "lowerleg02.L", "lowerleg02.R"],
                "foot": ["foot.L", "foot.R"]}


def car_overlap_by_region(body_b, body_v, body_polys, objs):
    segs = [(rg, Vector(tuple(P(b))), Vector(tuple(T(b)))) for rg, bl in REGION_BONES.items() for b in bl]
    out = {}
    for o in objs:
        pairs = body_b.overlap(bvh_of([o])[0])
        if not pairs:
            continue
        cnt = {}
        for ib, _ in pairs:
            cen = sum((body_v[i] for i in body_polys[ib]), Vector()) / len(body_polys[ib])
            best = min(segs, key=lambda sg: (cen - (sg[1] + (sg[2] - sg[1]) * max(0.0, min(1.0, (cen - sg[1]).dot(sg[2] - sg[1]) / max((sg[2] - sg[1]).length_squared, 1e-12))))).length)
            cnt[best[0]] = cnt.get(best[0], 0) + 1
        out[o.name] = cnt
    return out


def interference_report():
    upd()
    wheel_b, wheel_v = bvh_of([wheel_obj])
    _rim_set = set(RIM_COMP)
    _wvw = [wheel_obj.matrix_world @ v.co for v in wheel_obj.data.vertices]
    rim_b = BVHTree.FromPolygons(_wvw, [list(p.vertices) for p in wheel_obj.data.polygons if all(i in _rim_set for i in p.vertices)], all_triangles=False)
    rim_v = [_wvw[i] for i in RIM_COMP]
    body = [o for o in bpy.data.objects if o.type == "MESH" and o.name.startswith("Human") and o.visible_get()]
    body_b, body_v = bvh_of(body)
    dash_objs = [o for o in CAR_PARTS if o is not wheel_obj and o.name not in (DRIVER_SEAT_PART, PASSENGER_SEAT_PART) and not o.name.startswith("Wheel")]
    dash_b, _ = bvh_of(dash_objs)
    seat_objs = [bpy.data.objects[DRIVER_SEAT_PART]]
    seat_b, _ = bvh_of(seat_objs)
    # 太もも：体のうち、腰より前・ハンドルより下の頂点
    hipy = float(bw("root").y)
    # 手：手首より先の頂点（手首の骨の位置からの距離 < 0.2 m）
    wl, wr = Vector(tuple(P("wrist.L"))), Vector(tuple(P("wrist.R")))
    el, er = Vector(tuple(P("lowerarm01.L"))), Vector(tuple(P("lowerarm01.R")))
    near_arm = lambda v: min((v - wl).length, (v - wr).length) < 0.22 or min((v - el).length, (v - er).length) < 0.20  # noqa: E731
    thigh_v = [v for v in body_v if v.y < hipy - 0.10 and v.z < wheel_c[2] + 0.05 and abs(v.x) < 0.35 and not near_arm(v)]
    hand_v = [v for v in body_v if (v - wl).length < 0.20 or (v - wr).length < 0.20]
    body_b, body_v = bvh_of(body)
    body_polys = bvh_of.last_polys
    pairs = wheel_b.overlap(body_b)
    cls = {"hand": 0, "thigh": 0, "other": 0}
    for _, ib in pairs:
        cen = sum((body_v[i] for i in body_polys[ib]), Vector()) / len(body_polys[ib])
        if (cen - wl).length < 0.20 or (cen - wr).length < 0.20:
            cls["hand"] += 1
        elif cen.y < hipy - 0.10:
            cls["thigh"] += 1
        else:
            cls["other"] += 1
    # 手の交差が、どの骨のまわりで起きているか
    hand_bones = [f"{n}.{sd}" for sd in "LR" for n in ["wrist", "metacarpal1", "metacarpal2", "metacarpal3", "metacarpal4"] +
                  [f"finger{fi}-{j}" for fi in range(1, 6) for j in (1, 2, 3)]]
    segs_ = [(n, Vector(tuple(P(n))), Vector(tuple(T(n)))) for n in hand_bones]
    by_bone = {}
    for _, ib in pairs:
        cen = sum((body_v[i] for i in body_polys[ib]), Vector()) / len(body_polys[ib])
        if not ((cen - wl).length < 0.20 or (cen - wr).length < 0.20):
            continue
        best, bn = 1e9, None
        for n, h0, t0 in segs_:
            d = t0 - h0
            tt = max(0.0, min(1.0, (cen - h0).dot(d) / max(d.length_squared, 1e-12)))
            dd = (cen - (h0 + d * tt)).length
            if dd < best:
                best, bn = dd, n
        by_bone[bn] = by_bone.get(bn, 0) + 1
    rep_bones = dict(sorted(by_bone.items(), key=lambda kv: -kv[1]))
    rep = {
        "hand_intersections_by_nearest_bone": rep_bones,
        "wheel_x_body_triangle_pairs": cls,
        "rim_x_car_interior_triangle_pairs": len(rim_b.overlap(dash_b)),
        "wheel_part_x_car_interior_triangle_pairs_incl_column": len(wheel_b.overlap(dash_b)),
        "body_x_driver_seat_triangle_pairs": len(body_b.overlap(seat_b)),
        "body_x_driver_seat_by_region": car_overlap_by_region(body_b, body_v, body_polys, seat_objs),
        "body_x_car_interior_triangle_pairs_by_part": car_overlap_by_region(body_b, body_v, body_polys, dash_objs),
        "wheel_to_thigh_min_distance_mm": round(min_dist(thigh_v, wheel_b) * 1000, 1) if thigh_v else None,
        "rim_to_car_interior_min_distance_mm": round(min_dist(rim_v, dash_b) * 1000, 1),
        "hand_to_wheel_min_distance_mm": round(min_dist(hand_v, wheel_b) * 1000, 1) if hand_v else None,
    }
    # 手と縁の交差は、指・手のひらがハンドルの面に触れている分も数える（指の先・腹が縁の表面に触れると 0 mm）
    rep["note"] = "triangle_pairs = 交差している三角形の組の数（0 ならめり込みなし）。手は縁を握るので、手と縁は数 mm まで近づく"
    return rep


INTERF = interference_report()
# 頭と天井：頭（髪・頭の機材を含む）のいちばん上と、その真上の天井の内側の面の高さ
_hv = np.concatenate([eval_world_verts(o) for o in bpy.data.objects if o.type == "MESH" and o.name.startswith("Human") and o.visible_get()])
_top = _hv[np.argmax(_hv[:, 2])]
_roof = bpy.data.objects["Compact Car Part.092"]
_rb = bvh_of([_roof])[0]
_hit = _rb.ray_cast(Vector((float(_top[0]), float(_top[1]), float(_top[2]) - 0.30)), Vector((0, 0, 1)))
INTERF["head_top_z_m"] = round(float(_top[2]), 4)
INTERF["roof_inner_z_above_head_m"] = round(float(_hit[0].z), 4) if _hit[0] is not None else None
INTERF["car_seat_cushion_to_roof_m"] = round(float(_hit[0].z) - (cushion_top + tz), 4) if _hit[0] is not None else None
print("INTERF", json.dumps(INTERF, ensure_ascii=False))

# ---------------------------------------------------------------- 目を前の道路へ（自車線の 25 m 先）
a = np.array([0.15, float(cyc0[1]) - 25.0, gz + 0.02])
_, hist = mbk.aim_with_mpfb_ik(rig, eyes, eye_info, a)
upd()
m = mbk.measure(rig, eyes, eye_info)
r_head = gv.bone_delta_world(rig, "head")
pitch = {s: g1.dir_angles(r_head.T @ m[s][2])[1] - g1.dir_angles(f64(i["f_rest"]))[1] for s, i in eye_info.items()}
gv.apply_eyelid_follow(rig, r_head, pitch, cfg["eyelid_follow"]["upper_gain"], cfg["eyelid_follow"]["lower_gain"])
upd()
m = mbk.measure(rig, eyes, eye_info)
eye_c = {s: f64(m[s][0]) for s in m}
resid = {s: float(mbk.angle_deg(m[s][2], a - eye_c[s])) for s in m}
cyc = (eye_c["left"] + eye_c["right"]) / 2

# ---------------------------------------------------------------- 機材
fwd = r_head @ np.array([0.0, -1.0, 0.0])
lat = r_head @ np.array([1.0, 0.0, 0.0])
upv = r_head @ np.array([0.0, 0.0, 1.0])
gear = {}
# 車のカメラ 2 台（向きは仮）
# 車のカメラ：ダッシュボードの真ん中、フロントガラスの手前に、スマホ 1 台をホルダーで縦に立てる（約 15 × 7 × 0.8 cm）。
# 画面は運転者に向ける。画面側のカメラが運転者を、裏側のカメラが道路を撮る。
dg = bpy.context.evaluated_depsgraph_get()


def is_glass(ob, face_index):
    try:
        me = ob.data
        mi = me.polygons[face_index].material_index
        mt = ob.material_slots[mi].material
        return mt is not None and "glass" in mt.name.lower()
    except Exception:
        return False


def cast_skip_glass(o, d, max_hits=20):
    """o から d へ光線を当て、ガラス以外の最初の当たり（車の部品だけ）を返す。"""
    o, d = Vector(tuple(map(float, o))), Vector(tuple(map(float, d))).normalized()
    for _ in range(max_hits):
        hit, loc, nor, fi, ob, _ = scene.ray_cast(dg, o, d)
        if not hit:
            return None, None
        if ob in CAR_PARTS and not is_glass(ob, fi):
            return np.array(loc), ob
        o = loc + d * 1e-4
    return None, None


# ダッシュボードの上の面を、車の真ん中の x で、運転者の前から前へたどる。フロントガラスの手前（上に 17 cm 以上の空き）でいちばん前の所に置く
phone_base, dash_hit_obj = None, None
for yy in np.arange(float(wheel_c[1]) - 0.10, float(wheel_c[1]) - 0.60, -0.01):
    top, ob_ = cast_skip_glass((CX, yy, ROOF - 0.02), (0, 0, -1))
    if top is None or top[2] < float(wheel_c[2]) - 0.30 or top[2] > float(wheel_c[2]) + 0.35:   # 床・座席や、天井・ミラーは使わない
        continue
    hit_up = scene.ray_cast(dg, Vector((CX, yy, float(top[2]) + 0.005)), Vector((0, 0, 1)))
    clear = (hit_up[1].z - float(top[2])) if hit_up[0] else 1.0
    if clear >= 0.045 + 0.150 + 0.010:                    # ホルダーの台 + スマホの高さ + 余裕
        phone_base, dash_hit_obj = top, ob_                  # 前へたどって、空きが足りるいちばん前の所
if phone_base is None:
    raise SystemExit("スマホを置ける場所がダッシュボードの上に見つからない")
to_drv = cyc - phone_base
to_drv[2] = 0
to_drv /= np.linalg.norm(to_drv)
tilt = math.radians(12)                                   # 画面を少し上へ向ける（運転者の目の方）
ph_n = to_drv * math.cos(tilt) + np.array([0, 0, 1.0]) * math.sin(tilt)   # 画面の向き
ph_up = np.array([0, 0, 1.0]) - (np.array([0, 0, 1.0]) @ ph_n) * ph_n
ph_up /= np.linalg.norm(ph_up)
ph_x = np.cross(ph_up, ph_n)
PH_H, PH_W, PH_T = 0.150, 0.070, 0.008
holder_h = 0.045                                          # ホルダーの台の高さ（スマホの下の端まで）
ph_c = phone_base + np.array([0, 0, holder_h]) + ph_up * (PH_H / 2)
phone_m = principled("PhoneBody", (0.04, 0.04, 0.045), 0.35, 0.3)
glass_m = principled("PhoneScreen", (0.01, 0.012, 0.02), 0.05)
screen_glow = emission_only("PhoneScreenGlow", (0.10, 0.16, 0.26), 0.6)
holder_m = principled("HolderPlastic", (0.08, 0.08, 0.09), 0.5)
ph = oriented_box("Phone", ph_c, ph_x, ph_up, ph_n, PH_W, PH_H, PH_T, phone_m)
pbev = ph.modifiers.new("Round", "BEVEL")
pbev.width, pbev.segments = 0.004, 3
oriented_box("PhoneScreen", ph_c + ph_n * (PH_T / 2 + 0.0003), ph_x, ph_up, ph_n, PH_W - 0.006, PH_H - 0.008, 0.0004, screen_glow)
# 画面側のカメラ（運転者を撮る）：画面の上の端の真ん中
fcam = ph_c + ph_up * (PH_H / 2 - 0.009) + ph_n * (PH_T / 2 + 0.0008)
cylinder_between("PhoneFrontCam", fcam, fcam + ph_n * 0.0006, 0.0022, lens_m, 16)
# 裏側のカメラ（道路を撮る）：裏の上の角に出っぱり＋レンズ 2 つ
bump_c = ph_c + ph_up * (PH_H / 2 - 0.020) - ph_x * (PH_W / 2 - 0.020) - ph_n * (PH_T / 2 + 0.0015)
oriented_box("PhoneCamBump", bump_c, ph_x, ph_up, -ph_n, 0.030, 0.030, 0.003, phone_m)
for k_, off in enumerate((0.007, -0.007)):
    lc = bump_c + ph_up * off - ph_n * 0.0015
    cylinder_between(f"PhoneRearLens{k_}", lc, lc - ph_n * 0.0015, 0.0045, lens_m, 24)
    cylinder_between(f"PhoneRearLensRing{k_}", lc + ph_n * 0.0002, lc - ph_n * 0.0012, 0.0055, principled("LensRing", (0.6, 0.6, 0.62), 0.25, 1.0), 24)
# ホルダー：ダッシュボードに貼る台、首、スマホの裏の板、左右からはさむ爪
cylinder_between("HolderBase", phone_base - np.array([0, 0, 0.002]), phone_base + np.array([0, 0, 0.010]), 0.032, holder_m, 32)
back_pt = ph_c - ph_n * (PH_T / 2 + 0.006) - ph_up * 0.02
cylinder_between("HolderNeck", phone_base + np.array([0, 0, 0.010]), back_pt, 0.008, holder_m, 16)
oriented_box("HolderPlate", ph_c - ph_n * (PH_T / 2 + 0.003) - ph_up * 0.02, ph_x, ph_up, ph_n, 0.050, 0.060, 0.006, holder_m)
for sgn_ in (1, -1):
    oriented_box(f"HolderClaw{sgn_}", ph_c + ph_x * sgn_ * (PH_W / 2 + 0.002) - ph_up * 0.02, ph_x, ph_up, ph_n, 0.006, 0.030, PH_T + 0.010, holder_m)
gear["phone"] = {"where": "ダッシュボードの真ん中（車の中心の x）、フロントガラスの手前。ホルダーで縦に立てる", "size_m": [PH_W, PH_H, PH_T],
                 "center_world_m": [round(float(x), 4) for x in ph_c], "screen_normal_world": [round(float(x), 4) for x in ph_n],
                 "front_camera": {"location_world_m": [round(float(x), 4) for x in fcam], "facing": "運転者（画面側）"},
                 "rear_camera": {"location_world_m": [round(float(x), 4) for x in bump_c], "facing": "道路（裏側）"},
                 "dashboard_part_under": dash_hit_obj.name if dash_hit_obj else None}
fc, bc = bump_c, fcam            # 絵の上の位置（image_px）用：fc = 道路側のカメラ、bc = 運転者側のカメラ
if not NOGEAR:
    # 頭に着ける視線計測器：頭を一周するバンド、額のカメラ（前向き）、こめかみから伸びる細いアームの先に目を撮るカメラ
    dg = bpy.context.evaluated_depsgraph_get()
    head_c = cyc - fwd * 0.075 + upv * 0.045

    def surface(p_dir, base, off=0.006):
        d = Vector(tuple(map(float, p_dir))).normalized()
        o = Vector(tuple(map(float, base))) + d * 0.4
        for _ in range(30):
            hit, loc, nor, _, ob, _ = scene.ray_cast(dg, o, -d)
            if not hit:
                return np.array(base) + np.array(d) * 0.1
            if ob.name.startswith("Human"):
                return np.array(loc) + np.array(d) * off
            o = loc - d * 1e-4
        return np.array(base) + np.array(d) * 0.1

    band_base = head_c
    band = [surface(math.cos(t) * fwd + math.sin(t) * lat + upv * 0.10, band_base, 0.005) for t in np.linspace(0, 2 * math.pi, 56, endpoint=False)]
    tube("HeadBand", band, 0.0055, plastic, closed=True)
    fcam = surface(fwd + upv * 0.10, band_base, 0.026)
    oriented_box("HeadSceneCam", fcam, lat, upv, fwd, 0.040, 0.026, 0.026, camgrey)
    cylinder_between("HeadSceneCamLens", fcam + fwd * 0.012, fcam + fwd * 0.020, 0.009, lens_m)
    for s, sd in (("left", 1), ("right", -1)):
        temple = surface(lat * sd + fwd * 0.55 + upv * 0.10, band_base, 0.005)
        tip = eye_c[s] + fwd * 0.040 - upv * 0.022 + lat * sd * 0.020
        mid = temple + fwd * 0.04 + lat * sd * 0.012 - upv * 0.03
        tube(f"EyeCamArm_{s}", [temple, mid, tip], 0.0018, plastic)
        look = (eye_c[s] - tip) / np.linalg.norm(eye_c[s] - tip)
        ax, ay, _ = frame_axes(look, tuple(upv))
        oriented_box(f"EyeCam_{s}", tip, ax, ay, look, 0.009, 0.009, 0.012, camgrey)
        cylinder_between(f"EyeCamLens_{s}", tip + look * 0.006, tip + look * 0.008, 0.0028, lens_m, 16)
    gear["head_eye_tracker"] = {"parts": "頭を一周するバンド、額のカメラ（前向き）、こめかみから伸びるアームの先の目を撮るカメラ（左右）"}
else:
    red = emission_only("Gaze", (0.95, 0.03, 0.02), 1.0)
    for s in eye_c:
        tube(f"GazeLine_{s}", [eye_c[s], eye_c[s] + (a - eye_c[s]) / np.linalg.norm(a - eye_c[s]) * 3.0], 0.0026, red)

# ---------------------------------------------------------------- 光・描画
sun = bpy.data.lights.new("Sun", "SUN")
sun.energy, sun.angle, sun.color = 3.0, math.radians(2), (1.0, 0.97, 0.92)
so = bpy.data.objects.new("Sun", sun)
scene.collection.objects.link(so)
so.rotation_euler = (math.radians(50), 0.0, math.radians(-140))
world = scene.world
nt = world.node_tree
for n in list(nt.nodes):
    if n.type == "BACKGROUND":
        n.inputs["Color"].default_value = (0.55, 0.72, 0.95, 1)
        n.inputs["Strength"].default_value = 1.0
fill = bpy.data.lights.new("CabinFill", "AREA")
fill.size, fill.energy = 0.6, 25.0
fo = bpy.data.objects.new("CabinFill", fill)
scene.collection.objects.link(fo)
fo.visible_camera = False
fo.location = (PCX + 0.1, float(cyc[1]) - 0.2, ROOF - 0.05)
fo.rotation_euler = (Vector(tuple(map(float, cyc))) - fo.location).to_track_quat("-Z", "Y").to_euler()

pos = Vector((PCX + 0.05, float(cyc[1]) + 0.32, float(cyc[2]) + 0.02))       # 助手席の頭のあたり、少しうしろ
look = Vector((0.02, float(cyc[1]) - 0.32, float(cyc[2]) - 0.17))
cd = bpy.data.cameras.new("PassengerView")
cd.lens, cd.sensor_width, cd.sensor_fit, cd.clip_start = 20, 36, "HORIZONTAL", 0.02
cam = bpy.data.objects.new("PassengerView", cd)
scene.collection.objects.link(cam)
cam.location = pos
cam.rotation_euler = (look - pos).to_track_quat("-Z", "Y").to_euler()
scene.camera = cam
for ob in bpy.data.objects:      # 視点の手前にある助手席の背もたれ・頭当ては描かない
    if ob.name == PASSENGER_SEAT_PART:      # 助手席は視点のまわりにあるので、カメラには写さない（影・映り込みは残す）
        ob.visible_camera = False

scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
for dev_type in ("OPTIX", "CUDA"):
    try:
        prefs.compute_device_type = dev_type
        prefs.get_devices()
        if any(d.type == dev_type for d in prefs.devices):
            break
    except TypeError:
        continue
for dv in prefs.devices:
    dv.use = dv.type != "CPU"
scene.cycles.device = "GPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
r = scene.render
r.resolution_x, r.resolution_y, r.resolution_percentage = RES_X, RES_Y, 100
r.film_transparent = False
r.image_settings.file_format = "PNG"
r.image_settings.color_mode = "RGB"
tag = "_nogear" if NOGEAR else ""
out_png = os.path.join(OUT_DIR, f"driving_passenger_view{tag}_{RES_X}x{RES_Y}.png")
r.filepath = out_png
upd()
if not SKIPMAIN:
    bpy.ops.render.render(write_still=True)
check_imgs = {}
if HANDCHECK:            # 手のアップ：助手席側から／真上から（全体の視点・光は変えず、確認用のカメラを足すだけ）
    hc = (grip_pts["L"] + grip_pts["R"]) / 2
    views = {"passenger": (hc + np.array([-0.62, 0.20, 0.10]), hc + np.array([0.0, 0.0, -0.02]), 40),
             "top": (hc + np.array([0.0, 0.10, 0.55]), hc + np.array([0.0, 0.0, -0.01]), 35)}
    for vn, (cpos, clook, lens_) in views.items():
        cdd = bpy.data.cameras.new(f"HandCheck_{vn}")
        cdd.lens, cdd.sensor_width, cdd.clip_start = lens_, 36, (0.30 if vn == "top" else 0.01)   # 真上からは、手前 30 cm（車の天井）を写さない
        co_ = bpy.data.objects.new(f"HandCheck_{vn}", cdd)
        scene.collection.objects.link(co_)
        co_.location = Vector(tuple(map(float, cpos)))
        upv_ = (0, -1, 0) if vn == "top" else (0, 0, 1)
        dirv = Vector(tuple(map(float, clook - cpos)))
        co_.rotation_euler = dirv.to_track_quat("-Z", "Y" if vn != "top" else "Y").to_euler()
        if vn == "top":      # 真上から：画像の上が車の前（-Y）になるように回す
            m3 = Matrix(((-1, 0, 0), (0, 1, 0), (0, 0, -1))).transposed()
            co_.matrix_world = Matrix.Translation(co_.location) @ Matrix(((-1, 0, 0), (0, -1, 0), (0, 0, 1))).to_4x4()   # 下向き、画像の上 = 車の前
        scene.camera = co_
        for ob in bpy.data.objects:        # 真上から見るときは屋根を外す
            if ob.name == "Roof":
                ob.hide_render = vn == "top"
        r.filepath = os.path.join(OUT_DIR, f"driving_hands_check_{vn}{tag}.png")
        upd()
        bpy.ops.render.render(write_still=True)
        check_imgs[vn] = {"image": os.path.basename(r.filepath), "camera_location_m": [round(float(x), 3) for x in cpos], "lens_mm": lens_}
    for ob in bpy.data.objects:
        if ob.name == "Roof":
            ob.hide_render = False
    scene.camera = cam


def px(p):
    q = world_to_camera_view(scene, cam, Vector(tuple(map(float, p))))
    return [round(q.x * RES_X, 1), round((1 - q.y) * RES_Y, 1)]


vec = lambda p: [round(float(x), 4) for x in p]  # noqa: E731
info = {
    "note": "論文の撮り方を 3DCG で再現したイメージ。論文の図の写しではない。学習には使っていない（学習データの生成・学習はしていない）",
    "paper": {"name": "（未確定）運転中の視線データの撮り方", "reproduced_from": "依頼文に書かれた撮り方（左ハンドルの車内、頭に着ける視線計測器とカメラ、車のカメラ 2 台）。"
              "元の論文は未確定のため、車のカメラの向きは仮（front = 道路向き、back = 運転者向き）"},
    "image": os.path.basename(out_png), "size_px": [RES_X, RES_Y], "origin": "top-left, x right, y down", "nogear_version": NOGEAR,
    "person": {"generator": "scripts/generate_gaze_dataset_v3.py GeneratorV3.build() (Phase 8 v3) + clothes + shoes; seated by rotating leg / spine bones; hands on the wheel by Blender IK",
               "identity": PERSON["identity"], "appearance": {k: gen.app[k] for k in ("sex_for_appearance", "race_for_appearance", "skin", "hair", "iris")},
               "clothes": PERSON["clothes"], "shoes": PERSON["shoes"], "pose": "運転席に座り、ハンドルを 10 時と 2 時で握る"},
    "gaze": {"target": "前の道路（自車線、約 25 m 先）", "target_world_m": vec(a), "aim": "create_macbook_gaze_poc.aim_with_mpfb_ik",
             "residual_deg": {s: round(v, 4) for s, v in resid.items()}, "eye_center_world_m": {s: vec(v) for s, v in eye_c.items()}},
    "gear": gear,
    "car": {"drive": "左ハンドル（運転席は車の左、助手席は運転者の右 = -X）", "model": CAR_INFO, "car_center_x_m": round(CX, 4),
            "steering_wheel": WHEEL_INFO, "world_axes": "運転者は -Y（前）を向く。運転者の右 = -X、上 = +Z"},
    "camera": {"view": "助手席側から運転者を見る", "location_world_m": vec(pos), "look_at_world_m": vec(look), "lens_mm": 20, "sensor_width_mm": 36,
               "hidden_for_view": ["助手席（" + PASSENGER_SEAT_PART + "）は視点のまわりにあるので、カメラには写さない（影・映り込みは残す）"]},
    "image_px": {"eyes": px(cyc), "phone_rear_camera_road": px(fc), "phone_front_camera_driver": px(bc), "phone_center": px(ph_c)},
    "hands": HAND_LOG, "hand_check_images": check_imgs, "interference_check": INTERF, "driver_seat": SEAT_INFO,
    "changes_v4": [
        "車内を既存の車のモデル（SHC MCE 1、denniswoo1993、CGTrader、Royalty Free License（no AI））に置き換えた。スライドの絵にだけ使い、学習データには使わない",
        "車はハンドルの外径が 0.37 m になるように縮め（倍率は car.model.scale）、Z まわりに 180° 回して前を -Y にした。モデルはもともと左ハンドルなので反転していない",
        "前に自分で作った車内の部品（座席・ダッシュボード・メーター・画面・送風口・窓・柱・屋根・ドア・床・ボンネット）と Savino のハンドルは消し、ハンドルはモデルのものを使った",
        "車の位置：運転席の座面と背もたれの形を光線で測り、座った体（服を含む）に座面が 8 mm、背もたれが 10 mm まで近づく位置に車を置いた（人は動かしていない）",
        "手：前回と同じ握り方（左手 10 時・右手 2 時、手のひらで縁を包む、手の甲は運転者側の外向き、指は表面まで 9 mm で止める）。縁の円はモデルのハンドルの輪の形の塊から測った",
        "車のカメラ：前の箱 2 つを消し、ダッシュボードの真ん中・フロントガラスの手前に、スマホ（15 × 7 × 0.8 cm）をホルダーで縦に立てた。画面は運転者へ向け、画面側のカメラが運転者、裏側のカメラが道路"],
    "render": {"engine": "CYCLES", "device": "GPU", "samples": SAMPLES, "denoise": True, "view_transform": "Standard"},
}
jp = os.path.join(OUT_DIR, f"driving_passenger_view{tag}_{RES_X}x{RES_Y}.json")
json.dump(info, open(jp, "w"), indent=1, ensure_ascii=False)
print("DRIVING_DONE", jp)
