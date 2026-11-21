"""発表スライド用：視線推定の論文にある「データの撮り方」を 3DCG で再現したイメージ（論文の図の写しではない。学習には使わない）。
blender --background --python render_paper_scene.py -- SCENE SHOT OUT_DIR [RES_X RES_Y SAMPLES] [--nogear]
  SCENE = gafa（本棚の通路・監視カメラ。GAFA, Nonaka ら CVPR 2022 の撮り方）
  SHOT  = camN（監視カメラ N 番から 1 枚。N = 0..5） | all（監視カメラ全部から描いて 1 枚に並べる）
  --nogear : 機材（メガネ型計測器・胸のカメラ・ベルト）を外し、目から見ている本へ赤い線を引いた版
人 = generate_gaze_dataset_v3.GeneratorV3.build()（Phase 8 v3 の identity）+ 服と靴。目は aim_with_mpfb_ik で本へ向ける。"""
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
import create_gaze_poc as p2  # noqa: E402
import gaze_validation as gv  # noqa: E402
import generate_gaze_dataset_1000 as g1  # noqa: E402
import generate_gaze_dataset_v3 as v3  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MARKERS = os.path.join(HERE, "markers")
args = sys.argv[sys.argv.index("--") + 1:]
NOGEAR = "--nogear" in args
args = [a for a in args if not a.startswith("--")]
SCENE, SHOT, OUT_DIR = args[0], args[1], args[2]
RES_X = int(args[3]) if len(args) > 3 else 1920
RES_Y = int(args[4]) if len(args) > 4 else 1080
SAMPLES = int(args[5]) if len(args) > 5 else 128
os.makedirs(OUT_DIR, exist_ok=True)
assert SCENE == "gafa"

PERSON = {"identity": 228, "clothes": "male_casualsuit03", "shoes": "shoes04"}
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
# 学習用の MacBook・照明は使わない
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


# 立って棚の本を見る：腕を下ろし、頭を少し下げて本の方へ（人は -Y を向く。人の右 = -X）
rot_bone("upperarm01.L", "Y", 40)
rot_bone("upperarm01.R", "Y", -40)
rot_bone("lowerarm01.L", "X", 48)
rot_bone("lowerarm01.R", "X", 48)
rot_bone("neck01", "X", 6)
rot_bone("head", "X", 7)
rot_bone("head", "Z", -6)


def eval_world_verts(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    mw = np.array(ev.matrix_world)
    v = np.array([vv.co for vv in me.vertices])
    ev.to_mesh_clear()
    return v @ mw[:3, :3].T + mw[:3, 3]


shoes = next(o for o in bpy.data.objects if o.name.endswith("." + PERSON["shoes"]))
FLOOR = float(eval_world_verts(shoes)[:, 2].min())
bw = lambda n: rig.matrix_world @ rig.pose.bones[n].head  # noqa: E731


# ---------------------------------------------------------------- 材質・形
def principled(name, rgb, rough=0.5, metal=0.0, emit=0.0):
    mt = bpy.data.materials.new(name)
    mt.use_nodes = True
    b = mt.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit:
        b.inputs["Emission Color"].default_value = (*rgb, 1)
        b.inputs["Emission Strength"].default_value = emit
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


def image_mat(name, path):
    mt = bpy.data.materials.new(name)
    mt.use_nodes = True
    nt = mt.node_tree
    b = nt.nodes["Principled BSDF"]
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = bpy.data.images.load(path)
    t.interpolation = "Closest"
    nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.8
    return mt


def add_mesh(name, verts, faces, mt=None, uv=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], faces)
    ob = bpy.data.objects.new(name, me)
    scene.collection.objects.link(ob)
    if mt is not None:
        me.materials.append(mt)
    if uv is not None:
        lay = me.uv_layers.new(name="UVMap")
        for lp in me.loops:
            lay.data[lp.index].uv = uv[lp.vertex_index]
    return ob


def box(name, x0, x1, y0, y1, z0, z1, mt):
    v = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return add_mesh(name, v, f, mt)


def oriented_box(name, center, ax, ay, az, sx, sy, sz, mt):
    """center と直交する軸 ax, ay, az（単位ベクトル）、各軸方向の大きさ sx, sy, sz の箱。"""
    c = np.array(center)
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


# ---------------------------------------------------------------- 本棚の通路（棚は通路の両側。人の前 = -Y 側の棚）
rng = np.random.default_rng(2022)
shelf_wood = principled("ShelfWood", (0.58, 0.60, 0.62), 0.55, 0.3)       # 明るいグレーのスチール棚
floor_m = principled("Floor", (0.36, 0.40, 0.36), 0.7)                     # 緑がかったグレーの床
wall_m = principled("Wall", (0.88, 0.87, 0.84), 0.9)
ceil_m = principled("Ceiling", (0.92, 0.92, 0.92), 0.9)
book_cols = [(0.62, 0.16, 0.14), (0.15, 0.30, 0.55), (0.20, 0.42, 0.30), (0.78, 0.66, 0.38), (0.30, 0.24, 0.42),
             (0.85, 0.82, 0.74), (0.12, 0.12, 0.14), (0.55, 0.38, 0.22), (0.40, 0.55, 0.62), (0.72, 0.45, 0.50)]
book_mats = [principled(f"Book{i}", c, 0.6) for i, c in enumerate(book_cols)]
AISLE = 0.72                     # 通路の半分の幅（棚の前面の y = ±AISLE）
SH_D, SH_H, SH_X = 0.34, 2.05, 2.4
LEVELS = [0.08, 0.44, 0.80, 1.16, 1.52, 1.88]
BAYS = np.linspace(-SH_X, SH_X, 7)
H_ = FLOOR

box("Floor", -3.6, 3.6, -1.6, 1.6, H_ - 0.02, H_, floor_m)
box("Ceiling", -3.6, 3.6, -1.6, 1.6, H_ + 2.85, H_ + 2.87, ceil_m)
for sx in (-1, 1):
    box(f"EndWall{sx}", sx * 3.5, sx * 3.55, -1.6, 1.6, H_, H_ + 2.85, wall_m)
    box(f"BackWall{sx}", -3.6, 3.6, sx * 1.45, sx * 1.5, H_, H_ + 2.85, wall_m)

target_book = None
marker_slots = {(0, 1, 4), (1, 4, 2), (0, 5, 1), (1, 1, 3)}     # (棚, 区画, 段) を ArUco の札にする
markers_info = []
mk_id = 0
for side, sgn in ((0, -1), (1, 1)):          # 0 = 人の前の棚（-Y）、1 = うしろの棚（+Y）
    yf = sgn * AISLE                          # 前面（通路側）
    yb = sgn * (AISLE + SH_D)
    y0, y1 = min(yf, yb), max(yf, yb)
    box(f"ShelfBack{side}", -SH_X, SH_X, (yb - sgn * 0.01) if sgn > 0 else yb, (yb if sgn > 0 else yb + 0.01), H_, H_ + SH_H, shelf_wood)
    for z in LEVELS + [SH_H - 0.02]:
        box(f"Board{side}", -SH_X, SH_X, y0, y1, H_ + z - 0.02, H_ + z, shelf_wood)
    for x in BAYS:
        box(f"Upright{side}", x - 0.015, x + 0.015, y0, y1, H_, H_ + SH_H, shelf_wood)
    for bi in range(len(BAYS) - 1):
        for li, z in enumerate(LEVELS):
            xa, xb = BAYS[bi] + 0.02, BAYS[bi + 1] - 0.02
            if (side, bi, li) in marker_slots:     # この区画の真ん中に ArUco の札を立てる
                path = os.path.join(MARKERS, f"aruco_4x4_50_id{mk_id}.png")
                s_ = 0.16
                xc, zc = (xa + xb) / 2, H_ + z + 0.005 + s_ / 2
                yv = yf + sgn * 0.02
                if sgn < 0:   # 人の前の棚：面は +Y（通路）向き
                    vs = [(xc - s_ / 2, yv, zc - s_ / 2), (xc + s_ / 2, yv, zc - s_ / 2), (xc + s_ / 2, yv, zc + s_ / 2), (xc - s_ / 2, yv, zc + s_ / 2)]
                else:         # うしろの棚：面は -Y 向き（左右を入れ替えて、通路から見て正しい向きに）
                    vs = [(xc + s_ / 2, yv, zc - s_ / 2), (xc - s_ / 2, yv, zc - s_ / 2), (xc - s_ / 2, yv, zc + s_ / 2), (xc + s_ / 2, yv, zc + s_ / 2)]
                add_mesh(f"ArUco{mk_id}", vs, [(0, 1, 2, 3)], image_mat(f"ArUcoMat{mk_id}", path), uv=[(0, 0), (1, 0), (1, 1), (0, 1)])
                markers_info.append({"id": mk_id, "dict": "DICT_4X4_50", "where": "shelf", "center_m": [round(xc, 3), round(yv, 3), round(zc, 3)], "size_m": s_})
                mk_id += 1
                # 札の両わきは本で埋める
                xa_skip, xb_skip = xc - s_ / 2 - 0.01, xc + s_ / 2 + 0.01
            else:
                xa_skip = xb_skip = None
            x = xa
            while x < xb - 0.02:
                w = float(rng.uniform(0.018, 0.05))
                if x + w > xb:
                    break
                if xa_skip is not None and x + w > xa_skip and x < xb_skip:
                    x = xb_skip
                    continue
                h = float(rng.uniform(0.20, 0.31))
                d = float(rng.uniform(0.16, 0.24))
                if rng.random() < 0.06:            # ところどころ隙間
                    x += w
                    continue
                zf = H_ + z
                yfront = yf + sgn * 0.015               # 棚の前面から少し奥
                yback = yfront + sgn * d
                bk = box(f"Book{side}_{bi}_{li}", x, x + w, min(yfront, yback), max(yfront, yback), zf, zf + h, book_mats[int(rng.integers(len(book_mats)))])
                if side == 0 and bi == 3 and li == 4 and target_book is None and x > -0.10:
                    # 見ている本：少し手前に引き出した黄色い本
                    for o in [bk]:
                        bpy.data.objects.remove(o)
                    w = 0.045
                    tb_mat = principled("TargetBook", (0.92, 0.74, 0.12), 0.5)
                    yo = -sgn * 0.05                         # 通路側へ 5 cm 引き出す
                    target_book = box("TargetBook", x, x + w, min(yfront + yo, yback + yo), max(yfront + yo, yback + yo), zf, zf + 0.27, tb_mat)
                    target_center = np.array([x + w / 2, yfront + yo, zf + 0.15])    # 背表紙の真ん中
                x += w + 0.001
assert target_book is not None
# 床の ArUco（通路に 3 枚）
for k, (fx, fy) in enumerate([(-1.3, 0.25), (1.1, -0.3), (2.0, 0.3)]):
    s_ = 0.20
    path = os.path.join(MARKERS, f"aruco_4x4_50_id{mk_id}.png")
    add_mesh(f"ArUco{mk_id}", [(fx - s_ / 2, fy - s_ / 2, H_ + 0.001), (fx + s_ / 2, fy - s_ / 2, H_ + 0.001), (fx + s_ / 2, fy + s_ / 2, H_ + 0.001),
                               (fx - s_ / 2, fy + s_ / 2, H_ + 0.001)], [(0, 1, 2, 3)], image_mat(f"ArUcoMat{mk_id}", path),
             uv=[(0, 0), (1, 0), (1, 1), (0, 1)])
    markers_info.append({"id": mk_id, "dict": "DICT_4X4_50", "where": "floor", "center_m": [fx, fy, round(H_, 3)], "size_m": s_})
    mk_id += 1

# ---------------------------------------------------------------- 目を本へ
a = target_center
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
gear = []
plastic = principled("GearBlack", (0.015, 0.015, 0.018), 0.35)
camgrey = principled("GearGrey", (0.18, 0.18, 0.20), 0.4, 0.5)
lens_m = principled("Lens", (0.02, 0.03, 0.05), 0.05, 0.0)
strap_m = principled("Strap", (0.05, 0.05, 0.06), 0.8)
fwd = r_head @ np.array([0.0, -1.0, 0.0])        # 頭の前
lat = r_head @ np.array([1.0, 0.0, 0.0])         # 頭の左（人の左 = +X）
upv = r_head @ np.array([0.0, 0.0, 1.0])
if not NOGEAR:
    # (1) メガネ型の視線計測器：細い枠、目を撮る小さなカメラ（左右）、前を撮るカメラ（鼻の上）
    rims = {}
    for s in ("left", "right"):
        cen = eye_c[s] + fwd * 0.027 + upv * 0.003
        pts = [cen + lat * 0.025 * math.cos(t) + upv * 0.017 * math.sin(t) for t in np.linspace(0, 2 * math.pi, 40, endpoint=False)]
        gear.append(tube(f"GlassRim_{s}", pts, 0.0016, plastic, closed=True))
        rims[s] = cen
        side = 1 if s == "left" else -1
        outer = cen + lat * side * 0.025
        hinge = outer + lat * side * 0.012 - fwd * 0.004
        ear = hinge - fwd * 0.105 - upv * 0.012 + lat * side * 0.004
        gear.append(tube(f"GlassTemple_{s}", [outer, hinge, ear, ear - upv * 0.025 + fwd * 0.012], 0.0016, plastic))
        # 目を撮るカメラ：枠の下の内側（鼻寄り）から、上向きに目を見る
        ec = cen - upv * 0.017 - lat * side * 0.012 - fwd * 0.004
        look = unit_ = (eye_c[s] - ec) / np.linalg.norm(eye_c[s] - ec)
        ax2 = np.cross(look, lat)
        ax2 /= np.linalg.norm(ax2)
        ax1 = np.cross(ax2, look)
        gear.append(oriented_box(f"EyeCam_{s}", ec, ax1, ax2, look, 0.007, 0.007, 0.010, camgrey))
        gear.append(cylinder_between(f"EyeCamLens_{s}", ec + look * 0.005, ec + look * 0.0065, 0.0022, lens_m, 16))
    bridge = (rims["left"] + rims["right"]) / 2
    gear.append(tube("GlassBridge", [rims["left"] - lat * 0.025 + upv * 0.004, bridge + upv * 0.009, rims["right"] + lat * 0.025 + upv * 0.004], 0.0012, plastic))
    sc = bridge + upv * 0.016 + fwd * 0.002
    gear.append(oriented_box("SceneCam", sc, lat, upv, fwd, 0.014, 0.010, 0.010, camgrey))
    gear.append(cylinder_between("SceneCamLens", sc + fwd * 0.005, sc + fwd * 0.007, 0.0035, lens_m, 16))

    # (2) 胸のカメラ：肩から胸にかけたベルト（X 字）＋ 胸を一周するベルト、胸の真ん中にカメラ
    dg = bpy.context.evaluated_depsgraph_get()

    def surface(p_guess, axis_pt, off=0.008):
        """axis_pt から p_guess の方向へ外側から光線を当て、体（服）の表面のすぐ外の点を返す。"""
        d = Vector(tuple(map(float, p_guess - axis_pt))).normalized()
        o = Vector(tuple(map(float, axis_pt))) + d * 0.6
        for _ in range(30):                      # 人（体・服）以外に当たったら、その先からもう一度
            hit, loc, nor, _, ob, _ = scene.ray_cast(dg, o, -d)
            if not hit:
                return np.array(p_guess)
            if ob.name.startswith("Human"):
                return np.array(loc) + np.array(d) * off
            o = loc - d * 1e-4
        return np.array(p_guess)

    sp2, cl = bw("spine02"), bw("clavicle.L")
    z_chest = float(sp2.z + 0.55 * (cl.z - sp2.z))
    axis = lambda z: np.array([0.0, float(sp2.y) + 0.01, z])  # noqa: E731
    plate = surface(np.array([0.0, -1.0, z_chest]), axis(z_chest), 0.010)
    straps = []
    for sgn in (1, -1):    # 左右の肩から胸の板へ、背中側にも回す
        sh = bw(f"clavicle.{'L' if sgn > 0 else 'R'}")
        path = []
        for t in np.linspace(0, 1, 14):          # 胸の板 → 肩の上
            z = z_chest + t * (float(sh.z) + 0.07 - z_chest)
            x = sgn * (0.02 + t * 0.10)
            ang = math.radians(-90 + t * 80)     # 前（-Y）から上へ
            g = np.array([x, float(sp2.y) + 0.01 + math.sin(ang) * -0.3 * (1 - t), z + t * 0.2])
            path.append(surface(g, np.array([x, float(sp2.y) + 0.01, z - 0.05 * t])))
        for t in np.linspace(0, 1, 12)[1:]:      # 肩の上 → 背中
            z = float(sh.z) + 0.07 - t * (float(sh.z) + 0.07 - z_chest)
            x = sgn * (0.12 - t * 0.10)
            g = np.array([x, float(sp2.y) + 0.3, z + (1 - t) * 0.2])
            path.append(surface(g, np.array([x, float(sp2.y) + 0.01, z - 0.05 * (1 - t)])))
        straps.append(tube(f"HarnessStrap_{sgn}", path, 0.004, strap_m))
    loop = [surface(axis(z_chest - 0.04) + np.array([math.cos(t), math.sin(t), 0.0]), axis(z_chest - 0.04)) for t in np.linspace(0, 2 * math.pi, 48, endpoint=False)]
    straps.append(tube("HarnessChestLoop", loop, 0.004, strap_m, closed=True))
    gear += straps
    body_fwd = np.array([0.0, -1.0, 0.0])
    gear.append(oriented_box("ChestPlate", plate + body_fwd * 0.003, np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), body_fwd, 0.075, 0.075, 0.006, strap_m))
    cc = plate + body_fwd * 0.024
    gear.append(oriented_box("ChestCam", cc, np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), body_fwd, 0.062, 0.044, 0.034, camgrey))
    gear.append(cylinder_between("ChestCamLens", cc + body_fwd * 0.017, cc + body_fwd * 0.030, 0.011, lens_m, 24))
    chest_cam_pos = cc
else:
    red = emission_only("Gaze", (0.95, 0.03, 0.02), 1.0)
    for s in eye_c:
        tube(f"GazeLine_{s}", [eye_c[s], a], 0.0026, red)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.009, location=tuple(map(float, a + np.array([0, 0.004, 0]))))
    bpy.context.active_object.data.materials.append(red)

# ---------------------------------------------------------------- 監視カメラ（高さ約 2.5 m、人を見下ろす）
look_pt = np.array([float(cyc[0]), float(cyc[1]) - 0.15, float(H_ + 1.25)])
CCTV = [(-3.0, -0.40, 2.50), (3.0, 0.40, 2.50), (-1.9, 0.45, 2.60), (1.9, -0.45, 2.60), (1.5, 0.50, 2.55), (-1.5, -0.50, 2.55)]
CCTV_LENS = 22
house_m = principled("CCTVHousing", (0.93, 0.93, 0.92), 0.35)
cams = []
for i, (x, y, z) in enumerate(CCTV):
    p = np.array([x, y, H_ + z])
    d = (look_pt - p) / np.linalg.norm(look_pt - p)
    ax = np.cross(d, [0, 0, 1.0])
    ax /= np.linalg.norm(ax)
    ay = np.cross(ax, d)
    oriented_box(f"CCTVBody{i}", p - d * 0.09, ax, ay, d, 0.085, 0.075, 0.18, house_m)
    oriented_box(f"CCTVShade{i}", p - d * 0.08 + ay * 0.045, ax, ay, d, 0.10, 0.008, 0.22, house_m)
    cylinder_between(f"CCTVLens{i}", p - d * 0.005, p + d * 0.002, 0.022, lens_m)
    top = np.array([x, y, H_ + 2.85])
    cylinder_between(f"CCTVMount{i}", p - d * 0.09 + ay * 0.04, top, 0.012, house_m, 12)
    cd = bpy.data.cameras.new(f"CCTV{i}")
    cd.lens, cd.sensor_width, cd.sensor_fit = CCTV_LENS, 36, "HORIZONTAL"
    cd.clip_start = 0.02
    cam = bpy.data.objects.new(f"CCTV{i}", cd)
    scene.collection.objects.link(cam)
    cam.location = Vector(tuple(p + d * 0.01))
    cam.rotation_euler = Vector(tuple(d)).to_track_quat("-Z", "Y").to_euler()
    cams.append(cam)

# 照明：天井の照明パネル 3 枚 + 弱い環境光
for k, x in enumerate((-2.0, 0.0, 2.0)):
    ld = bpy.data.lights.new(f"Ceil{k}", "AREA")
    ld.shape, ld.size, ld.size_y, ld.energy, ld.color = "RECTANGLE", 1.2, 0.6, 45.0, (1.0, 0.97, 0.92)
    lo = bpy.data.objects.new(f"Ceil{k}", ld)
    scene.collection.objects.link(lo)
    lo.location = (x, 0.0, H_ + 2.80)
    box(f"CeilPanel{k}", x - 0.6, x + 0.6, -0.3, 0.3, H_ + 2.835, H_ + 2.85, emission_only(f"PanelGlow{k}", (1.0, 0.98, 0.94), 1.0))
scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.8, 0.8, 0.8, 1)
scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.08

# ---------------------------------------------------------------- 描画
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
r.resolution_percentage = 100
r.film_transparent = False
r.image_settings.file_format = "PNG"
r.image_settings.color_mode = "RGB"
tag = "_nogear" if NOGEAR else ""


def render(cam, path, rx, ry):
    scene.camera = cam
    r.resolution_x, r.resolution_y = rx, ry
    r.filepath = path
    upd()
    bpy.ops.render.render(write_still=True)


def px(cam, p, rx, ry):
    q = world_to_camera_view(scene, cam, Vector(tuple(map(float, p))))
    return [round(q.x * rx, 1), round((1 - q.y) * ry, 1)]


outputs = []
if SHOT.startswith("cam"):
    i = int(SHOT[3:])
    path = os.path.join(OUT_DIR, f"gafa_cctv{i}{tag}_{RES_X}x{RES_Y}.png")
    render(cams[i], path, RES_X, RES_Y)
    outputs.append({"image": os.path.basename(path), "camera": f"CCTV{i}", "size_px": [RES_X, RES_Y],
                    "image_px": {"target_book": px(cams[i], a, RES_X, RES_Y), "eyes": px(cams[i], cyc, RES_X, RES_Y)}})
vec = lambda p: [round(float(x), 4) for x in p]  # noqa: E731
if SHOT == "all":
    tiles = []
    for i, cam in enumerate(cams):
        pth = os.path.join(OUT_DIR, "_tiles", f"gafa_cctv{i}{tag}_{RES_X}x{RES_Y}.png")
        os.makedirs(os.path.dirname(pth), exist_ok=True)
        render(cam, pth, RES_X, RES_Y)
        tiles.append(pth)
    outputs.append({"tiles": [os.path.basename(t) for t in tiles], "size_px_each": [RES_X, RES_Y], "layout": "3 列 × 2 行、左上から CCTV0..5"})

info = {
    "note": "論文の撮り方を 3DCG で再現したイメージ。論文の図の写しではない。学習には使っていない（学習データの生成・学習はしていない）",
    "paper": {"name": "GAFA", "title": "Dynamic 3D Gaze from Afar: Deep Gaze Estimation from Temporal Eye-Head-Body Coordination",
              "authors": "Soma Nonaka, Shohei Nobuhara, Ko Nishino", "venue": "CVPR 2022",
              "reproduced_from": "依頼文に書かれた撮り方（本棚の通路・メガネ型の視線計測器・胸のカメラ・ArUco マーカー・高い位置の監視カメラ）。論文の図は写していない。家具の配置・色・服は独自"},
    "gear": [] if NOGEAR else [
        {"name": "メガネ型の視線計測器", "parts": "細い黒い枠、目を撮る小さなカメラ（左右、枠の下の内側から目を見上げる）、前を撮るカメラ（鼻の上）"},
        {"name": "胸のカメラ", "parts": "肩から胸にかけた X 字のベルト＋胸を一周するベルト、胸の板、前向きのカメラ", "location_world_m": vec(chest_cam_pos)}],
    "nogear_version": NOGEAR,
    "person": {"generator": "scripts/generate_gaze_dataset_v3.py GeneratorV3.build() (Phase 8 v3) + clothes + shoes; arms lowered and head turned by rotating bones",
               "identity": PERSON["identity"], "appearance": {k: gen.app[k] for k in ("sex_for_appearance", "race_for_appearance", "skin", "hair", "iris")},
               "clothes": PERSON["clothes"], "shoes": PERSON["shoes"], "pose": "立って前の棚の本を見る"},
    "gaze": {"target": "前の棚（人の正面）の上から 2 段目、少し引き出した黄色い本の背表紙の真ん中", "target_world_m": vec(a),
             "aim": "create_macbook_gaze_poc.aim_with_mpfb_ik", "residual_deg": {s: round(v, 4) for s, v in resid.items()},
             "eye_center_world_m": {s: vec(v) for s, v in eye_c.items()}},
    "room": {"aisle_width_m": 2 * AISLE, "shelf": {"height_m": SH_H, "depth_m": SH_D, "length_m": 2 * SH_X, "levels_m": LEVELS},
             "floor_z_m": round(H_, 4), "ceiling_height_m": 2.85, "aruco_markers": markers_info,
             "world_axes": "人は -Y を向く（前の棚）。人の右 = -X、上 = +Z。床の高さ floor_z_m"},
    "cameras": [{"name": f"CCTV{i}", "location_world_m": vec(c.location), "height_above_floor_m": round(float(c.location.z - H_), 3),
                 "look_at_world_m": vec(look_pt), "lens_mm": CCTV_LENS, "sensor_width_mm": 36} for i, c in enumerate(cams)],
    "outputs": outputs,
    "render": {"engine": "CYCLES", "device": "GPU", "samples": SAMPLES, "denoise": True, "view_transform": "Standard"},
}
jp = os.path.join(OUT_DIR, f"gafa_{SHOT}{tag}_{RES_X}x{RES_Y}.json")
json.dump(info, open(jp, "w"), indent=1, ensure_ascii=False)
print("PAPER_DONE", jp)
