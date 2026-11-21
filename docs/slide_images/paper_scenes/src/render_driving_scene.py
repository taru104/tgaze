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
args = [x for x in args if not x.startswith("--")]
OUT_DIR = args[0]
RES_X = int(args[1]) if len(args) > 1 else 1920
RES_Y = int(args[2]) if len(args) > 2 else 1080
SAMPLES = int(args[3]) if len(args) > 3 else 128
os.makedirs(OUT_DIR, exist_ok=True)
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
    rot_bone(f"upperleg01.{s}", "X", -98)
    rot_bone(f"upperleg01.{s}", "Z", sgn * 5)
    rot_bone(f"lowerleg01.{s}", "X", 62)
rot_bone("spine05", "X", -8)
rot_bone("neck01", "X", 6)
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

# ---------------------------------------------------------------- 車内（運転者の x = 0、助手席は -X 側、車の中心は x = CX）
CX = -0.37
X_L, X_R = 0.40, CX * 2 - 0.40          # 運転席側の内壁 / 助手席側の内壁
wheel_c = np.array([0.0, float(cl.y) - 0.43, float(cl.z) - 0.30])
wheel_n = np.array([0.0, 0.88, 0.47])          # ハンドルの面は運転者の方を向き、少し上を向く
wheel_n /= np.linalg.norm(wheel_n)
wx_, wy_, _ = frame_axes(wheel_n)              # wx_ = 人の左（+X）、wy_ = ハンドル面の上
WR = 0.185
dash_z = float(wheel_c[2]) + 0.13
dash_near = float(wheel_c[1]) - 0.16
dash_far = dash_near - 0.55
ROOF = float(cyc0[2]) + 0.27

# 座席（運転席・助手席）
for nm, x0 in (("Driver", 0.0), ("Passenger", CX * 2)):
    box(f"{nm}SeatCushion", x0 - 0.27, x0 + 0.27, float(pel.y) - 0.38, float(pel.y) + 0.14, SEAT - 0.12, SEAT, seat_m)
    sb = float(pel.y) + 0.17
    v = []
    for zz, yy in ((SEAT - 0.02, sb), (SEAT + 0.68, sb + 0.13)):
        for xx in (x0 - 0.26, x0 + 0.26):
            v += [(xx, yy, zz), (xx, yy + 0.10, zz)]
    add_mesh(f"{nm}SeatBack", [v[0], v[2], v[6], v[4], v[1], v[3], v[7], v[5]],
             [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)], seat_m)
    box(f"{nm}Headrest", x0 - 0.13, x0 + 0.13, sb + 0.12, sb + 0.20, SEAT + 0.72, SEAT + 0.92, seat_m)
box("Floor", X_R, X_L, dash_far - 0.4, float(pel.y) + 1.0, FLOOR - 0.03, FLOOR, dark)
box("CenterConsole", CX - 0.10, CX + 0.10, dash_near - 0.05, float(pel.y) + 0.05, FLOOR, SEAT + 0.05, trim)
# ダッシュボード（運転席の前は計器のひさし付き）
box("Dashboard", X_R, X_L, dash_far, dash_near, dash_z - 0.30, dash_z, dark)
box("DashLower", X_R, X_L, dash_far, dash_near - 0.10, FLOOR, dash_z - 0.30, dark)
box("InstrumentHood", -0.20, 0.20, dash_near - 0.20, dash_near - 0.02, dash_z, dash_z + 0.06, dark)
add_mesh("Instruments", [(-0.16, dash_near - 0.021, dash_z - 0.10), (0.16, dash_near - 0.021, dash_z - 0.10),
                         (0.16, dash_near - 0.021, dash_z + 0.03), (-0.16, dash_near - 0.021, dash_z + 0.03)],
         [(0, 3, 2, 1)], emission_only("InstrumentGlow", (0.05, 0.10, 0.16), 1.0))
# 中央の画面（ダッシュボード中央、運転者の方へ少し向ける）
scr_c = np.array([CX, dash_near + 0.01, dash_z + 0.02])
scr_n = np.array([0.25, 1.0, 0.18])
scr_n /= np.linalg.norm(scr_n)
sx_, sy_, _ = frame_axes(scr_n)
oriented_box("CenterScreenBody", scr_c, sx_, sy_, scr_n, 0.27, 0.17, 0.02, dark)
oriented_box("CenterScreen", scr_c + scr_n * 0.0105, sx_, sy_, scr_n, 0.25, 0.15, 0.001, emission_only("ScreenGlow", (0.10, 0.20, 0.32), 1.0))
# ハンドル（リム・スポーク・コラム）
rim = [wheel_c + WR * (math.cos(t) * wx_ + math.sin(t) * wy_) for t in np.linspace(0, 2 * math.pi, 64, endpoint=False)]
tube("WheelRim", rim, 0.016, dark, closed=True)
for t in (math.radians(0), math.radians(180), math.radians(270)):
    tube(f"WheelSpoke{int(math.degrees(t))}", [wheel_c, wheel_c + (WR - 0.01) * (math.cos(t) * wx_ + math.sin(t) * wy_)], 0.014, trim)
oriented_box("WheelHub", wheel_c - wheel_n * 0.01, wx_, wy_, wheel_n, 0.10, 0.08, 0.05, trim)
cylinder_between("SteeringColumn", wheel_c - wheel_n * 0.03, wheel_c - wheel_n * 0.30 - np.array([0, 0, 0.05]), 0.035, dark)
# フロントガラス・屋根・柱・運転席のドア
ws_bot = dash_far + 0.02
ws_top_y = float(cyc0[1]) - 0.55
add_mesh("Windshield", [(X_R, ws_bot, dash_z), (X_L, ws_bot, dash_z), (X_L, ws_top_y, ROOF), (X_R, ws_top_y, ROOF)], [(0, 1, 2, 3)], glass)
box("Roof", X_R - 0.05, X_L + 0.05, ws_top_y, float(pel.y) + 1.0, ROOF, ROOF + 0.04, principled("Headliner", (0.72, 0.70, 0.66), 0.9))
for xp in (X_R, X_L):
    tube(f"APillar{xp:+.2f}", [(xp, ws_bot, dash_z), (xp, ws_top_y, ROOF)], 0.04, trim)
box("DriverDoorLower", X_L, X_L + 0.06, dash_far, float(pel.y) + 0.6, FLOOR, dash_z - 0.05, principled("DoorTrim", (0.16, 0.16, 0.17), 0.6))
add_mesh("DriverWindow", [(X_L + 0.03, dash_near, dash_z - 0.05), (X_L + 0.03, float(pel.y) + 0.5, dash_z - 0.05),
                          (X_L + 0.03, float(pel.y) + 0.5, ROOF), (X_L + 0.03, ws_top_y, ROOF)], [(0, 1, 2, 3)], glass)
tube("BPillar", [(X_L + 0.03, float(pel.y) + 0.5, dash_z - 0.05), (X_L + 0.03, float(pel.y) + 0.5, ROOF)], 0.05, trim)
box("Hood", X_R - 0.05, X_L + 0.05, dash_far - 1.2, dash_far, dash_z - 0.16, dash_z - 0.12, principled("HoodPaint", (0.20, 0.30, 0.45), 0.35, 0.2))

# ---------------------------------------------------------------- 外：道路・草地・空
road = principled("Asphalt", (0.17, 0.17, 0.18), 0.85)
grass = principled("Grass", (0.24, 0.40, 0.18), 0.9)
paint = principled("LinePaint", (0.95, 0.95, 0.92), 0.6)
gz = FLOOR - 0.25
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

# ---------------------------------------------------------------- 手をハンドルへ（10 時と 2 時）
grips = {}
for s, sgn in (("L", 1), ("R", -1)):
    ang = math.radians(60)
    g = wheel_c + WR * (math.cos(ang) * wy_ + sgn * math.sin(ang) * wx_)
    radial = (g - wheel_c) / np.linalg.norm(g - wheel_c)
    wrist = g + radial * 0.02 + wheel_n * 0.05 - wy_ * 0.06
    grips[s] = g
    sh = bw(f"upperarm01.{s}")
    te = bpy.data.objects.new(f"HandTarget.{s}", None)
    scene.collection.objects.link(te)
    te.location = Vector(tuple(map(float, wrist)))
    pe = bpy.data.objects.new(f"ElbowPole.{s}", None)
    scene.collection.objects.link(pe)
    pe.location = Vector((float(sh.x) + sgn * 0.45, float(sh.y) + 0.10, float(sh.z) - 0.45))
    c_ = rig.pose.bones[f"lowerarm02.{s}"].constraints.new("IK")
    c_.target, c_.pole_target, c_.chain_count, c_.pole_angle, c_.iterations = te, pe, 4, math.radians(-90), 2000
upd()
for s, sgn in (("L", 1), ("R", -1)):
    pb = rig.pose.bones[f"wrist.{s}"]
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = (0.0, sgn * math.radians(80), 0.0)
upd()

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
fc = np.array([CX, ws_bot + 0.07, dash_z + 0.035])                 # フロントガラスの下の中央、道路向き
fd = np.array([0.0, -1.0, 0.0])
ax, ay, _ = frame_axes(fd)
oriented_box("FrontCamera", fc, ax, ay, fd, 0.07, 0.045, 0.05, camgrey)
cylinder_between("FrontCameraLens", fc + fd * 0.025, fc + fd * 0.035, 0.014, lens_m)
bc = scr_c + sy_ * 0.105 + scr_n * 0.02                             # 中央の画面の上、運転者向き
bd = (cyc - bc) / np.linalg.norm(cyc - bc)
ax, ay, _ = frame_axes(bd)
oriented_box("BackCamera", bc, ax, ay, bd, 0.06, 0.04, 0.045, camgrey)
cylinder_between("BackCameraLens", bc + bd * 0.022, bc + bd * 0.032, 0.012, lens_m)
gear["front_camera"] = {"location_world_m": [round(float(x), 4) for x in fc], "direction": [0, -1, 0], "where": "フロントガラスの下の中央", "facing": "道路（仮）"}
gear["back_camera"] = {"location_world_m": [round(float(x), 4) for x in bc], "direction": [round(float(x), 4) for x in bd], "where": "ダッシュボード中央の画面の上", "facing": "運転者（仮）"}
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
fo.location = (CX * 2 + 0.1, float(cyc[1]) - 0.2, ROOF - 0.05)
fo.rotation_euler = (Vector(tuple(map(float, cyc))) - fo.location).to_track_quat("-Z", "Y").to_euler()

pos = Vector((CX * 2 + 0.05, float(cyc[1]) + 0.32, float(cyc[2]) + 0.02))       # 助手席の頭のあたり、少しうしろ
look = Vector((0.02, float(cyc[1]) - 0.32, float(cyc[2]) - 0.17))
cd = bpy.data.cameras.new("PassengerView")
cd.lens, cd.sensor_width, cd.sensor_fit, cd.clip_start = 20, 36, "HORIZONTAL", 0.02
cam = bpy.data.objects.new("PassengerView", cd)
scene.collection.objects.link(cam)
cam.location = pos
cam.rotation_euler = (look - pos).to_track_quat("-Z", "Y").to_euler()
scene.camera = cam
for ob in bpy.data.objects:      # 視点の手前にある助手席の背もたれ・頭当ては描かない
    if ob.name.startswith(("PassengerSeatBack", "PassengerHeadrest")):
        ob.hide_render = True

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
bpy.ops.render.render(write_still=True)


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
    "car": {"drive": "左ハンドル（運転席は車の左、助手席は運転者の右 = -X）", "car_center_x_m": CX, "steering_wheel_center_m": vec(wheel_c),
            "steering_wheel_radius_m": WR, "center_screen_m": vec(scr_c), "world_axes": "運転者は -Y（前）を向く。運転者の右 = -X、上 = +Z"},
    "camera": {"view": "助手席側から運転者を見る", "location_world_m": vec(pos), "look_at_world_m": vec(look), "lens_mm": 20, "sensor_width_mm": 36,
               "hidden_for_view": ["助手席の背もたれと頭当て（視点の手前にあるため描いていない）"]},
    "image_px": {"eyes": px(cyc), "front_camera": px(fc), "back_camera": px(bc)},
    "render": {"engine": "CYCLES", "device": "GPU", "samples": SAMPLES, "denoise": True, "view_transform": "Standard"},
}
jp = os.path.join(OUT_DIR, f"driving_passenger_view{tag}_{RES_X}x{RES_Y}.json")
json.dump(info, open(jp, "w"), indent=1, ensure_ascii=False)
print("DRIVING_DONE", jp)
