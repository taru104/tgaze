"""スライド用「Tgaze が使われる場面」の画像（学習には使わない）。
blender --background --python render_usage_scene.py -- SCENE VIEW OUT_PNG OUT_JSON [RES_X RES_Y SAMPLES]
  SCENE = 1 (EC サイト) | 2 (オンライン授業) | 3 (視線キーボード)
  VIEW  = shoulder (人の右うしろ・肩越し) | side (横からの 3/4)
人 = generate_gaze_dataset_v3.GeneratorV3.build()（Phase 8 v3 の identity）+ 服と靴。
MacBook = create_macbook_gaze_poc.build_macbook()（GeneratorV3.build() の中で作られたもの）を机の上へ動かし、
本体と蓋を描画に出してアルミと黒い縁の材質を付ける。目は aim_with_mpfb_ik で画面上の注視点へ向ける。"""
import json
import math
import os
import sys

import bpy
import bmesh
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

PROJ = os.path.expanduser("~/mpfb2-gaze-generator-poc")
sys.path.insert(0, os.path.join(PROJ, "scripts"))
os.chdir(PROJ)
import create_macbook_gaze_poc as mbk  # noqa: E402
import create_full_face_poc as p1  # noqa: E402
import create_gaze_poc as p2  # noqa: E402
import generate_gaze_dataset_1000 as g1  # noqa: E402
import generate_gaze_dataset_v3 as v3  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, "site") if os.path.isdir(os.path.join(HERE, "site")) else HERE   # 画面画像と targets.json の置き場所
argv = sys.argv[sys.argv.index("--") + 1:]
SCENE, VIEW, OUT_PNG, OUT_JSON = int(argv[0]), argv[1], argv[2], argv[3]
RES_X = int(argv[4]) if len(argv) > 4 else 1920
RES_Y = int(argv[5]) if len(argv) > 5 else 1080
SAMPLES = int(argv[6]) if len(argv) > 6 else 128

TARGETS = json.load(open(os.path.join(SITE, "targets.json")))
SCENES = {
    1: {"name": "home_ec_site", "identity": 247, "clothes": "male_casualsuit01", "shoes": "shoes01",
        "screen": "screen_market.png", "target": "market", "hands": "keyboard", "props": ["mug", "plant"]},
    2: {"name": "online_class", "identity": 236, "clothes": "female_casualsuit01", "shoes": "shoes03",
        "screen": "screen_lecture.png", "target": "lecture", "hands": "keyboard", "props": ["notebook", "plant"]},
    3: {"name": "gaze_keyboard", "identity": 242, "clothes": "female_casualsuit02", "shoes": "shoes02",
        "screen": "screen_keyboard.png", "target": "keyboard", "hands": "lap", "props": ["plant", "mug"]},
}
SC = SCENES[SCENE]
TGT = TARGETS[SC["target"]]                     # u, v: 画面画像の左上が原点（右・下が正）、[0, 1]
DESK_H = 0.72                                   # 天板の高さ（床から）
EYE_TO_SCREEN = 0.60                            # 目（両目の中点）から画面中心まで
LID_DEG = 112.0                                 # 蓋の開き（本体に対して）

cfg = json.load(open(os.path.join(PROJ, "configs", "gaze_dataset_v2.json")))

# ---------------------------------------------------------------- 人（v3 identity + 服・靴）
_orig_build_human = p1.build_human


def build_human_with_clothes(shape_fn=None):
    p1.MHCLO_ASSETS.append(("clothes", f"{SC['clothes']}.mhclo", "Clothes"))
    p1.MHCLO_ASSETS.append(("clothes", f"{SC['shoes']}.mhclo", "Shoes"))
    return _orig_build_human(shape_fn)


p1.build_human = build_human_with_clothes
v3.GeneratorV3.set_lighting = lambda self, L: None      # 照明は部屋の照明に置き換える
gen = v3.GeneratorV3(cfg, SC["identity"], 1, os.path.join(os.path.dirname(OUT_JSON), "_scene_tmp"), "train")
gen.build()
cfg = gen.cfg
scene, rig, eyes, eye_info = gen.scene, gen.rig, gen.eyes, gen.eye_info
for n in ("KeyLight", "FillLight"):
    if n in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[n])
mbk.add_mpfb_eye_ik(rig)                         # 体のポーズより先に（IK の目標は頭に付く）
upd = bpy.context.view_layer.update
upd()


def rot_bone(name, axis, deg):
    """骨を、その付け根を中心に、アーマチュア空間の軸 axis まわりに deg 度回す（子の骨も一緒に回る）。"""
    pb = rig.pose.bones[name]
    upd()
    m = pb.matrix.copy()
    h = m.translation.copy()
    pb.matrix = Matrix.Translation(h) @ Matrix.Rotation(math.radians(deg), 4, axis) @ Matrix.Translation(-h) @ m
    upd()


# 座る：太ももを前へ、すねを下へ。上体を少し前に傾け、頭を少し下げる（人は -Y を向いている。人の右 = -X）
for s, sgn in (("L", 1), ("R", -1)):
    rot_bone(f"upperleg01.{s}", "X", -88)
    rot_bone(f"upperleg01.{s}", "Z", sgn * 4)
    rot_bone(f"lowerleg01.{s}", "X", 84)
rot_bone("spine04", "X", 5)
rot_bone("spine02", "X", 4)
rot_bone("neck01", "X", 8)
rot_bone("head", "X", 7)


def eval_world_verts(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    mw = np.array(ev.matrix_world)
    v = np.array([vv.co for vv in me.vertices])
    ev.to_mesh_clear()
    return v @ mw[:3, :3].T + mw[:3, 3]


body = bpy.data.objects["Human"]
shoes = next(o for o in bpy.data.objects if o.name.endswith("." + SC["shoes"]))
floor_z = float(eval_world_verts(shoes)[:, 2].min())
bv = eval_world_verts(body)
pel = rig.matrix_world @ rig.pose.bones["root"].head
hipz = min((rig.matrix_world @ rig.pose.bones[f"upperleg02.{s}"].head).z for s in "LR")
seat_z = float(hipz) - 0.085                     # 太ももの骨の付け根から、お尻の厚みぶん下
thigh_top = float(hipz) + 0.07
desk_top = floor_z + DESK_H


def eye_centres():
    return {s: v3.v2.f64(p2.eye_center_world(rig, i["bone"])) for s, i in eye_info.items()}


ec = eye_centres()
cyc = (ec["left"] + ec["right"]) / 2

# ---------------------------------------------------------------- MacBook を机の上へ
root = bpy.data.objects["MacBookAirM4"]
assembly = bpy.data.objects["MacBookAirM4.DisplayAssembly"]
active, webcam = gen.active, gen.webcam
assembly.rotation_euler = (math.radians(LID_DEG - 90.0), 0.0, 0.0)
upd()
o = np.array(active.matrix_world.translation) - np.array(root.location)       # 蝶番 → 画面中心
t_base = mbk.D["BASE_THICKNESS_M"]
root_z = desk_top + t_base
dz = cyc[2] - (root_z + o[2])
horiz = math.sqrt(max(EYE_TO_SCREEN ** 2 - dz ** 2, 0.15 ** 2))
center_y = cyc[1] - horiz
root.location = Vector((float(cyc[0]), float(center_y - o[1]), float(root_z)))
upd()
c, right, up, normal = mbk.display_frame(active)
DW, DH = mbk.DISPLAY_W, mbk.DISPLAY_H
base_front_y = float(root.location.y) + mbk.D["MACBOOK_DEPTH_M"] - mbk.D["LID_THICKNESS_M"]   # 手前の縁（人の側）


# ---------------------------------------------------------------- 材質
def principled(name, rgb, rough=0.5, metal=0.0, emit=0.0, alpha=1.0):
    mt = bpy.data.materials.new(name)
    mt.use_nodes = True
    b = mt.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit:
        b.inputs["Emission Color"].default_value = (*rgb, 1)
        b.inputs["Emission Strength"].default_value = emit
    if alpha < 1:
        b.inputs["Alpha"].default_value = alpha
    return mt


def assign(obj, mt):
    obj.data.materials.clear()
    obj.data.materials.append(mt)
    obj.hide_render = False


alu = principled("Aluminium", (0.80, 0.81, 0.83), 0.32, 1.0)
black = principled("BezelBlack", (0.01, 0.01, 0.012), 0.25)
keycap = principled("Keycap", (0.03, 0.03, 0.035), 0.55)
pad = principled("Trackpad", (0.62, 0.63, 0.66), 0.25, 1.0)
assign(bpy.data.objects["MacBookAirM4.Base"], alu)
assign(bpy.data.objects["MacBookAirM4.Lid"], alu)


def add_mesh(name, verts, faces, parent=None, mt=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    ob = bpy.data.objects.new(name, me)
    scene.collection.objects.link(ob)
    if parent is not None:
        ob.parent = parent
    if mt is not None:
        me.materials.append(mt)
    return ob


def box(name, x0, x1, y0, y1, z0, z1, mt, parent=None):
    v = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return add_mesh(name, v, f, parent, mt)


# 黒い縁（蓋の前面いっぱい。画面はその 0.2 mm 手前）
lid_h = mbk.D["LID_HEIGHT_M"]
W = mbk.D["MACBOOK_WIDTH_M"]
add_mesh("Bezel", [(-W / 2 + 0.002, -0.0001, 0.002), (W / 2 - 0.002, -0.0001, 0.002), (W / 2 - 0.002, -0.0001, lid_h - 0.002),
                   (-W / 2 + 0.002, -0.0001, lid_h - 0.002)], [(0, 1, 2, 3)], assembly, black)
# キーボード（本体の上面。root の +Y が人の側）とトラックパッド
bm = bmesh.new()
rows = [(14, 0.0), (14, 0.0), (13, 0.5), (12, 0.75), (11, 1.25), (9, 0.0)]
kw, gap, y0 = 0.0165, 0.0025, 0.022
for r, (n, off) in enumerate(rows):
    y = y0 + r * (kw + gap)
    width = 14 * (kw + gap) - gap
    xs = -width / 2 + off * (kw + gap) / 2
    for i in range(n):
        x = xs + i * (width - off * (kw + gap)) / n
        w_i = (width - off * (kw + gap)) / n - gap
        if r == 5 and i == 4:
            w_i = w_i * 1.0
        bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Translation((x + w_i / 2, y + kw / 2, 0.0006)) @ Matrix.Diagonal((w_i, kw, 0.0012, 1)))
km = bpy.data.meshes.new("Keyboard")
bm.to_mesh(km)
bm.free()
kb_obj = bpy.data.objects.new("Keyboard", km)
scene.collection.objects.link(kb_obj)
kb_obj.parent = root
km.materials.append(keycap)
add_mesh("Trackpad", [(-0.06, 0.135, 0.0002), (0.06, 0.135, 0.0002), (0.06, 0.200, 0.0002), (-0.06, 0.200, 0.0002)],
         [(0, 1, 2, 3)], root, pad)

# 画面：画像を貼って光らせる。ActiveDisplay の局所 x = 人から見て右、y = 上
img = bpy.data.images.load(os.path.join(SITE, SC["screen"]))
scr = bpy.data.materials.new("ScreenImage")
scr.use_nodes = True
nt = scr.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
tex = nt.nodes.new("ShaderNodeTexImage")
tex.image = img
tex.interpolation = "Cubic"
em = nt.nodes.new("ShaderNodeEmission")
em.inputs["Strength"].default_value = 1.0
outn = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(tex.outputs["Color"], em.inputs["Color"])
nt.links.new(em.outputs["Emission"], outn.inputs["Surface"])
me = active.data
uvl = me.uv_layers.new(name="UVMap")
for loop in me.loops:
    x, y, _ = me.vertices[loop.vertex_index].co
    uvl.data[loop.index].uv = (x / DW + 0.5, y / DH + 0.5)     # 左下 (0,0)、右上 (1,1)：画像の上が画面の上
assign(active, scr)

# ---------------------------------------------------------------- 注視点・目
U, V = 2 * TGT["u"] - 1, 1 - 2 * TGT["v"]                       # 画面の (-1..1)、右・上が正
a = mbk.display_point(active, U, V)
_, hist = mbk.aim_with_mpfb_ik(rig, eyes, eye_info, a)
upd()
m = mbk.measure(rig, eyes, eye_info)
r_head = v3.v2.f64(np.array(__import__("gaze_validation").bone_delta_world(rig, "head")))
pitch = {s: g1.dir_angles(r_head.T @ m[s][2])[1] - g1.dir_angles(v3.v2.f64(i["f_rest"]))[1] for s, i in eye_info.items()}
__import__("gaze_validation").apply_eyelid_follow(rig, r_head, pitch, cfg["eyelid_follow"]["upper_gain"], cfg["eyelid_follow"]["lower_gain"])
upd()
m = mbk.measure(rig, eyes, eye_info)
eye_c = {s: v3.v2.f64(m[s][0]) for s in m}
resid = {s: mbk.angle_deg(m[s][2], a - eye_c[s]) for s in m}
cyc = (eye_c["left"] + eye_c["right"]) / 2

# ---------------------------------------------------------------- 腕（IK）：手をキーボードの上 / ひざの上へ
def empty(name, loc):
    e = bpy.data.objects.new(name, None)
    scene.collection.objects.link(e)
    e.location = Vector(tuple(loc))
    return e


arm_info = {}
for s, sgn in (("L", 1), ("R", -1)):
    if SC["hands"] == "keyboard":
        wrist = (float(root.location.x) + sgn * 0.085, base_front_y + 0.01, root_z + 0.022)
    else:   # ひざの上（太ももの前半分）
        kx = rig.matrix_world @ rig.pose.bones[f"lowerleg01.{s}"].head
        wrist = (float(kx.x) * 0.8 + sgn * 0.01, float(pel.y + (kx.y - pel.y) * 0.62), thigh_top + 0.035)
    sh = rig.matrix_world @ rig.pose.bones[f"upperarm01.{s}"].head
    tgt = empty(f"HandTarget.{s}", wrist)
    pole = empty(f"ElbowPole.{s}", (sh.x + sgn * 0.35, sh.y + 0.45, sh.z - 0.35))
    c_ = rig.pose.bones[f"lowerarm02.{s}"].constraints.new("IK")
    c_.target, c_.pole_target, c_.chain_count = tgt, pole, 4
    c_.pole_angle = math.radians(-90)
    c_.iterations = 2000
    arm_info[s] = {"wrist_target": list(wrist)}
upd()
# 手のひらを下へ：前腕の軸まわりに手首を回す（rest の A ポーズでは手のひらが内向き）
for s, sgn in (("L", 1), ("R", -1)):
    pb = rig.pose.bones[f"wrist.{s}"]
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = (0.0, sgn * math.radians(70 if SC["hands"] == "keyboard" else 55), 0.0)
upd()

# ---------------------------------------------------------------- 赤い線・赤い点
red = bpy.data.materials.new("Gaze")             # 光を受けない発光だけの赤（画面の白い光で白っぽくならない）
red.use_nodes = True
_n = red.node_tree.nodes
_n.remove(_n["Principled BSDF"])
_e = _n.new("ShaderNodeEmission")
_e.inputs["Color"].default_value, _e.inputs["Strength"].default_value = (0.95, 0.03, 0.02, 1), 1.0
red.node_tree.links.new(_e.outputs["Emission"], _n["Material Output"].inputs["Surface"])
ball = mbk.make_marker("GazePoint", Vector(tuple(a + normal * 0.002)), 0.0055, (1, 1, 1, 1))
ball.data.materials.append(red)
ball.hide_render = False
for s in eye_c:
    ln = mbk.make_polyline(f"GazeLine_{s}", [Vector(tuple(eye_c[s])), Vector(tuple(a))], 0.0026, (1, 1, 1, 1))
    ln.data.materials.append(red)
    ln.hide_render = False

# ---------------------------------------------------------------- 部屋：床・壁・窓・机・いす・小物
wood = principled("DeskWood", (0.42, 0.28, 0.16), 0.45)
floor_m = principled("Floor", (0.30, 0.22, 0.15), 0.6)
wall_m = principled("Wall", (0.86, 0.85, 0.82), 0.9)
steel = principled("Steel", (0.12, 0.12, 0.13), 0.4, 0.8)
fabric = principled("ChairFabric", (0.16, 0.20, 0.30), 0.9)
desk_front = base_front_y + 0.16
desk_back = desk_front - 0.75
wall_y = desk_back - 0.25
box("Floor", -4, 4, wall_y - 0.05, 4, floor_z - 0.02, floor_z, floor_m)
box("BackWall", -4, 4, wall_y - 0.05, wall_y, floor_z, floor_z + 2.7, wall_m)
# 左の壁（人の左 = +X）と窓（光る面 + 窓から入る光）
wx = 1.55
box("SideWall", wx, wx + 0.05, wall_y - 0.05, 4, floor_z, floor_z + 2.7, wall_m)
sky = principled("WindowSky", (0.80, 0.88, 1.0), 1.0, 0.0, 1.5)
win_y0, win_y1, win_z0, win_z1 = wall_y + 0.35, wall_y + 1.65, floor_z + 0.9, floor_z + 2.2
add_mesh("Window", [(wx - 0.002, win_y0, win_z0), (wx - 0.002, win_y1, win_z0), (wx - 0.002, win_y1, win_z1), (wx - 0.002, win_y0, win_z1)],
         [(0, 3, 2, 1)], None, sky)
frame = principled("WindowFrame", (0.95, 0.95, 0.95), 0.5)
for nm, (y0_, y1_, z0_, z1_) in {"top": (win_y0 - 0.04, win_y1 + 0.04, win_z1, win_z1 + 0.04), "bot": (win_y0 - 0.04, win_y1 + 0.04, win_z0 - 0.04, win_z0),
                                  "l": (win_y0 - 0.04, win_y0, win_z0, win_z1), "r": (win_y1, win_y1 + 0.04, win_z0, win_z1),
                                  "mid": ((win_y0 + win_y1) / 2 - 0.015, (win_y0 + win_y1) / 2 + 0.015, win_z0, win_z1)}.items():
    box(f"WinFrame_{nm}", wx - 0.03, wx, y0_, y1_, z0_, z1_, frame)
ld = bpy.data.lights.new("WindowLight", "AREA")
ld.shape, ld.size, ld.size_y, ld.energy, ld.color = "RECTANGLE", 1.3, 1.3, 60.0, (1.0, 0.97, 0.92)
lo = bpy.data.objects.new("WindowLight", ld)
scene.collection.objects.link(lo)
lo.location = (wx - 0.05, (win_y0 + win_y1) / 2, (win_z0 + win_z1) / 2)
lo.rotation_euler = Vector((-1, 0, -0.25)).to_track_quat("-Z", "Y").to_euler()
cl = bpy.data.lights.new("CeilingLight", "AREA")
cl.size, cl.energy, cl.color = 1.5, 45.0, (1.0, 0.93, 0.85)
co = bpy.data.objects.new("CeilingLight", cl)
scene.collection.objects.link(co)
co.location = (0.0, float(cyc[1]) - 0.2, floor_z + 2.6)
glow = bpy.data.lights.new("DisplayGlow", "AREA")
glow.shape, glow.size, glow.size_y, glow.energy, glow.color = "RECTANGLE", DW, DH, 4.0, (0.85, 0.9, 1.0)
go = bpy.data.objects.new("DisplayGlow", glow)
scene.collection.objects.link(go)
go.matrix_world = active.matrix_world.copy()
go.location = Vector(tuple(c + normal * 0.004))
go.rotation_euler = Vector(tuple(normal)).to_track_quat("-Z", "Y").to_euler()   # 光は人の方へ（-Z が画面の前）
# 机
box("DeskTop", -0.75, 0.75, desk_back, desk_front, desk_top - 0.03, desk_top, wood)
for x_ in (-0.71, 0.67):
    for y_ in (desk_back + 0.04, desk_front - 0.08):
        box("DeskLeg", x_, x_ + 0.04, y_, y_ + 0.04, floor_z, desk_top - 0.03, steel)
# いす（座面はお尻の下、背もたれは背中のうしろ）
sx, sy = float(pel.x), float(pel.y)
seat_top = seat_z - 0.005
box("ChairSeat", sx - 0.24, sx + 0.24, sy - 0.30, sy + 0.18, seat_top - 0.06, seat_top, fabric)
bk = rig.matrix_world @ rig.pose.bones["spine02"].head
back_y = float(bk.y) + 0.16
box("ChairBack", sx - 0.22, sx + 0.22, back_y, back_y + 0.05, seat_top + 0.12, seat_top + 0.55, fabric)
box("ChairPost", sx - 0.025, sx + 0.025, sy - 0.08, sy - 0.03, floor_z + 0.06, seat_top - 0.06, steel)
for ang in range(5):
    t = math.radians(90 + ang * 72)
    box("ChairFoot", sx + 0.26 * math.cos(t) - 0.02, sx + 0.26 * math.cos(t) + 0.02, sy - 0.055 + 0.26 * math.sin(t) - 0.02,
        sy - 0.055 + 0.26 * math.sin(t) + 0.02, floor_z, floor_z + 0.05, steel)
    box("ChairArmLeg", sx + 0.13 * math.cos(t) - 0.015, sx + 0.13 * math.cos(t) + 0.015, sy - 0.055 + 0.13 * math.sin(t) - 0.015,
        sy - 0.055 + 0.13 * math.sin(t) + 0.015, floor_z + 0.04, floor_z + 0.07, steel)
box("ChairBackPost", sx - 0.02, sx + 0.02, back_y + 0.01, back_y + 0.04, seat_top - 0.05, seat_top + 0.15, steel)


def cylinder(name, x, y, z0, r, h, mt, verts=40):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=h, location=(x, y, z0 + h / 2))
    ob = bpy.context.active_object
    ob.name = name
    ob.data.materials.append(mt)
    return ob


props = []
if "mug" in SC["props"]:
    mug_m = principled("Mug", (0.93, 0.93, 0.90), 0.3)
    mx_, my_ = float(root.location.x) + 0.30, float(root.location.y) + 0.08
    cylinder("Mug", mx_, my_, desk_top, 0.042, 0.10, mug_m)
    cylinder("Coffee", mx_, my_, desk_top + 0.085, 0.038, 0.004, principled("Coffee", (0.12, 0.06, 0.03), 0.2))
    bpy.ops.mesh.primitive_torus_add(major_radius=0.028, minor_radius=0.008, location=(mx_ - 0.045, my_, desk_top + 0.055),
                                     rotation=(math.radians(90), 0, 0))
    bpy.context.active_object.data.materials.append(mug_m)
    props.append("mug")
if "notebook" in SC["props"]:
    nb = principled("Notebook", (0.25, 0.45, 0.70), 0.6)
    paper = principled("Paper", (0.95, 0.95, 0.92), 0.8)
    nx_, ny_ = float(root.location.x) - 0.33, float(root.location.y) + 0.12
    box("Notebook", nx_ - 0.09, nx_ + 0.09, ny_ - 0.12, ny_ + 0.12, desk_top, desk_top + 0.008, nb)
    box("NotebookPage", nx_ - 0.085, nx_ + 0.085, ny_ - 0.115, ny_ + 0.115, desk_top + 0.008, desk_top + 0.009, paper)
    pen = cylinder("Pen", nx_ + 0.12, ny_, desk_top + 0.004, 0.004, 0.14, principled("Pen", (0.1, 0.1, 0.12), 0.3), 12)
    pen.rotation_euler = (math.radians(90), 0, math.radians(15))
    pen.location.z = desk_top + 0.005
    props.append("notebook")
if "plant" in SC["props"]:
    pot = principled("Pot", (0.75, 0.42, 0.30), 0.7)
    leaf = principled("Leaf", (0.13, 0.38, 0.12), 0.6)
    px_, py_ = -0.55, desk_back + 0.16
    cylinder("Pot", px_, py_, desk_top, 0.07, 0.13, pot)
    rng = np.random.default_rng(7)
    for i in range(14):
        th = rng.uniform(0, 2 * math.pi)
        rr = rng.uniform(0.0, 0.07)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1, segments=16, ring_count=8,
                                             location=(px_ + rr * math.cos(th), py_ + rr * math.sin(th), desk_top + 0.16 + rng.uniform(0, 0.16)))
        lf = bpy.context.active_object
        lf.scale = (0.045, 0.018, 0.012)
        lf.rotation_euler = (rng.uniform(-0.6, 0.6), rng.uniform(-0.8, 0.8), th)
        lf.data.materials.append(leaf)
        for p_ in lf.data.polygons:
            p_.use_smooth = True
    props.append("plant")

# ---------------------------------------------------------------- 視点
cyc_v, c_v = Vector(tuple(cyc)), Vector(tuple(c))
if VIEW == "shoulder":       # 人の右うしろ（右 = -X、うしろ = +Y）、約 2 m、55 mm。画面の中身が読めるように画面へ向ける
    look = c_v + Vector((0.0, 0.0, -0.02)) + (cyc_v - c_v) * 0.10
    pos = cyc_v + Vector((-0.95, 1.55, 0.42))
    lens = 55
else:                        # 横からの 3/4（人の右前の上）：顔の向き・赤い線・画面を 1 枚に
    look = (cyc_v + c_v) / 2 + Vector((0.0, 0.0, -0.05))
    pos = look + Vector((-1.35, 0.10, 0.80))
    lens = 45
cd = bpy.data.cameras.new("SlideCam")
cd.lens, cd.sensor_width, cd.sensor_fit = lens, 36, "HORIZONTAL"
cam = bpy.data.objects.new("SlideCam", cd)
scene.collection.objects.link(cam)
cam.location = pos
cam.rotation_euler = (look - pos).to_track_quat("-Z", "Y").to_euler()
scene.camera = cam

# ---------------------------------------------------------------- 描画設定
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
for d in prefs.devices:
    d.use = d.type != "CPU"
scene.cycles.device = "GPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.cycles.use_adaptive_sampling = True
scene.view_settings.view_transform = "Standard"          # AgX だと画面が白っぽく抜ける
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.6, 0.7, 1)
scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.12
r = scene.render
r.resolution_x, r.resolution_y, r.resolution_percentage = RES_X, RES_Y, 100
r.film_transparent = False
r.image_settings.file_format = "PNG"
r.image_settings.color_mode = "RGB"
r.filepath = OUT_PNG
for ob in bpy.data.objects:          # 学習用カメラの視野などのデバッグ物は出さない
    if ob.name.startswith(("Fov", "CamMark")):
        ob.hide_render = True
upd()
bpy.ops.render.render(write_still=True)


def px(p):
    q = world_to_camera_view(scene, cam, Vector(tuple(p)))
    return [round(q.x * RES_X, 1), round((1 - q.y) * RES_Y, 1)]


vec = lambda p: [round(float(x), 4) for x in p]  # noqa: E731
out = {
    "image": os.path.basename(OUT_PNG), "size_px": [RES_X, RES_Y], "origin": "top-left, x right, y down",
    "note": "スライド用のイメージ。学習には使っていない（学習データの生成・学習はしていない）",
    "scene": {"id": SCENE, "name": SC["name"], "view": VIEW},
    "person": {"generator": "scripts/generate_gaze_dataset_v3.py GeneratorV3.build() (Phase 8 v3) + clothes + shoes; seated by rotating leg / spine / neck bones; arms by Blender IK",
               "identity": SC["identity"],
               "appearance": {k: gen.app[k] for k in ("sex_for_appearance", "race_for_appearance", "skin", "hair", "iris", "eyebrows", "eyelashes", "eyelid")},
               "hair": gen.app["hair"], "clothes": SC["clothes"], "shoes": SC["shoes"], "hands": SC["hands"]},
    "screen": {"image": SC["screen"], "image_size_px": [2560, 1664],
               "gaze_target": TGT["target"], "gaze_target_uv_image": [round(TGT["u"], 4), round(TGT["v"], 4)],
               "uv_note": "画面画像の左上が原点、右・下が正、[0,1]",
               "gaze_target_world_m": vec(a), "display_center_world_m": vec(c), "display_normal_world": vec(normal),
               "lid_angle_deg": LID_DEG, "eye_to_screen_center_m": round(float(np.linalg.norm(cyc - c)), 4)},
    "eyes": {"aim": "create_macbook_gaze_poc.aim_with_mpfb_ik", "residual_deg": {s: round(float(v), 4) for s, v in resid.items()},
             "center_world_m": {s: vec(v) for s, v in eye_c.items()}},
    "room": {"floor_z_m": round(floor_z, 4), "desk_top_z_m": round(desk_top, 4), "desk_height_m": DESK_H,
             "seat_z_m": round(seat_top, 4), "props": props, "world_axes": "人は -Y を向く。人の右 = -X、上 = +Z"},
    "camera": {"view": VIEW, "location_world_m": vec(pos), "look_at_world_m": vec(look), "lens_mm": lens, "sensor_width_mm": 36,
               "distance_to_eyes_m": round(float((pos - cyc_v).length), 3), "rotation_euler_rad": vec(cam.rotation_euler)},
    "image_px": {"gaze_point": px(a), "eye_left": px(eye_c["left"]), "eye_right": px(eye_c["right"]), "display_center": px(c)},
    "render": {"engine": "CYCLES", "device": "GPU", "samples": SAMPLES, "denoise": True, "view_transform": "Standard"},
}
json.dump(out, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
print("SCENE_DONE", json.dumps(out, ensure_ascii=False))
