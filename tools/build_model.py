"""
Build a MuJoCo MJCF model of the myCobot 280 from Elephant Robotics' official URDF.

Usage (only needed if you want to rebuild or add another variant):
    git clone --depth 1 https://github.com/elephantrobotics/mycobot_ros
    python tools/build_model.py --repo mycobot_ros --variant m5
    python tools/build_model.py --repo mycobot_ros --variant pi

It reads the URDF joint frames exactly, converts the .dae meshes to .stl
(MuJoCo cannot read .dae), and writes models/mycobot_280_<variant>.xml.
"""
import argparse, pathlib, xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R

VARIANTS = {
    "m5": "mycobot_description/urdf/mycobot_280_m5/mycobot_280_m5.urdf",
    "pi": "mycobot_description/urdf/mycobot_280_pi/mycobot_280_pi.urdf",
    "jn": "mycobot_description/urdf/mycobot_280_jn/mycobot_280_jn.urdf",
}
# The Pi's joint5.dae hides its geometry behind library-node instances that the
# converter can't place, so borrow the identical J5 servo shell from the M5 files.
OVERRIDES = {
    "pi": {"joint5": dict(mesh="../mycobot_280_m5/joint5.dae",
                          xyz=[0.0, 0.0, -0.03356], rpy=[-1.5708, 0.0, 0.0])},
}
# Colours roughly matching the real arm (white shell, dark base)
COLOR = {"g_base": "0.15 0.15 0.17 1"}
DEFAULT_COLOR = "0.92 0.92 0.92 1"


def f3(s):
    return [float(v) for v in s.split()] if s else [0.0, 0.0, 0.0]


def quat_from_rpy(rpy):
    x, y, z, w = R.from_euler("xyz", rpy).as_quat()  # URDF rpy = extrinsic XYZ
    return [w, x, y, z]


def fmt(v, nd=6):
    return " ".join(f"{x:.{nd}g}" for x in v)


def _load_dae(src):
    """Load a .dae as one triangle mesh in metres (honours <unit meter=...>)."""
    import re, trimesh
    txt = pathlib.Path(src).read_text(errors="ignore")
    m = re.search(r'<unit[^>]*meter="([0-9.eE+-]+)"', txt)
    unit = float(m.group(1)) if m else 1.0
    mesh = trimesh.load(src, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        # trimesh skips <polygons>; read the scene graph with pycollada instead
        import collada
        c = collada.Collada(str(src), ignore=[collada.common.DaeUnsupportedError,
                                              collada.common.DaeBrokenRefError])
        V, F, n = [], [], 0
        bound = list(c.scene.objects("geometry"))
        # Some files reference library nodes pycollada can't follow; use raw geometry then
        prims = ([p for bg in bound for p in bg.primitives()] if bound
                 else [p for g in c.geometries for p in g.primitives])
        if True:
            for prim in prims:
                if hasattr(prim, "triangleset"):
                    prim = prim.triangleset()
                if not hasattr(prim, "vertex_index") or prim.vertex_index is None:
                    continue
                V.append(prim.vertex); F.append(prim.vertex_index + n); n += len(prim.vertex)
        mesh = trimesh.Trimesh(np.vstack(V), np.vstack(F), process=True)
    if unit != 1.0:
        mesh.apply_scale(unit)
    return mesh


def convert_mesh(src, dst, max_faces):
    mesh = _load_dae(src)
    if max_faces and len(mesh.faces) > max_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
        except Exception as e:  # decimation is optional
            print("  (decimation skipped:", e, ")")
    mesh.export(dst)
    return len(mesh.faces), mesh.bounds


def build(repo, variant, out_dir, max_faces):
    urdf_path = pathlib.Path(repo) / VARIANTS[variant]
    root = ET.parse(urdf_path).getroot()
    mesh_out = out_dir / f"meshes_{variant}"
    mesh_out.mkdir(parents=True, exist_ok=True)

    links = {}
    for ln in root.findall("link"):
        vis = ln.find("visual")
        if vis is None:
            links[ln.get("name")] = None
            continue
        fn = vis.find("geometry/mesh").get("filename").split("/")[-1]
        org = vis.find("origin")
        links[ln.get("name")] = dict(
            mesh=fn, xyz=f3(org.get("xyz")), rpy=f3(org.get("rpy")))
    for name, ov in OVERRIDES.get(variant, {}).items():
        links[name] = dict(ov)

    joints, children = {}, set()
    for j in root.findall("joint"):
        org = j.find("origin")
        lim = j.find("limit")
        joints[j.find("parent").get("link")] = dict(
            name=j.get("name"), type=j.get("type"), child=j.find("child").get("link"),
            xyz=f3(org.get("xyz")), rpy=f3(org.get("rpy")),
            axis=f3(j.find("axis").get("xyz")) if j.find("axis") is not None else [0, 0, 1],
            lo=float(lim.get("lower")) if lim is not None else None,
            hi=float(lim.get("upper")) if lim is not None else None)
        children.add(j.find("child").get("link"))
    base = [l for l in links if l not in children][0]

    # Convert meshes
    assets = []
    for name, v in links.items():
        if v is None:
            continue
        stl = pathlib.Path(v["mesh"]).with_suffix(".stl").name
        nf, b = convert_mesh(urdf_path.parent / v["mesh"], mesh_out / stl, max_faces)
        print(f"  {v['mesh']:<16} -> {stl:<16} {nf:6d} faces, size {np.ptp(b, 0).round(3)} m")
        assets.append(f'    <mesh name="{name}" file="{stl}" inertia="convex"/>')

    # Walk the kinematic chain
    body_xml, indent, link, n_rev = [], 2, base, 0
    act_xml = []

    def geom(link_name, pad):
        v = links[link_name]
        if v is None:
            return []
        rgba = COLOR.get(link_name, DEFAULT_COLOR)
        return [f'{pad}<geom class="link" mesh="{link_name}" pos="{fmt(v["xyz"])}" '
                f'quat="{fmt(quat_from_rpy(v["rpy"]))}" rgba="{rgba}"/>']

    pad = "  " * indent
    body_xml.append(f'{pad}<body name="{base}" pos="0 0 0">')
    body_xml += geom(base, pad + "  ")
    depth = 1
    while link in joints:
        j = joints[link]
        pad = "  " * (indent + depth)
        # gravcomp mimics the servo's own position loop holding the link against gravity
        body_xml.append(f'{pad}<body name="{j["child"]}" pos="{fmt(j["xyz"])}" '
                        f'quat="{fmt(quat_from_rpy(j["rpy"]))}" gravcomp="1">')
        if j["type"] == "revolute":
            n_rev += 1
            jn = f"joint{n_rev}"
            body_xml.append(f'{pad}  <joint name="{jn}" class="servo" axis="{fmt(j["axis"])}" '
                            f'range="{j["lo"]:.4f} {j["hi"]:.4f}"/>')
            act_xml.append(f'    <position name="{jn}" joint="{jn}" class="servo" '
                           f'ctrlrange="{j["lo"]:.4f} {j["hi"]:.4f}"/>')
        body_xml += geom(j["child"], pad + "  ")
        link = j["child"]
        depth += 1
    # Tool-centre point = centre of the output flange face
    pad = "  " * (indent + depth)
    body_xml.append(f'{pad}<site name="tip" size="0.006" rgba="0 0.6 1 1"/>')
    for d in range(depth - 1, -1, -1):
        body_xml.append("  " * (indent + d) + "</body>")

    xml = f"""<!-- myCobot 280 ({variant.upper()}) for MuJoCo.
     Kinematics and meshes generated from Elephant Robotics' official URDF
     (github.com/elephantrobotics/mycobot_ros, BSD 3-Clause, see LICENSE_mycobot_ros).
     Joint angles are in radians and match the real robot's angle convention:
     degrees(qpos) is what pymycobot's send_angles() expects. -->
<mujoco model="mycobot_280_{variant}">
  <compiler angle="radian" meshdir="meshes_{variant}" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>

  <visual>
    <global offwidth="1280" offheight="960"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.6 0.6 0.6"/>
  </visual>

  <default>
    <default class="servo">
      <joint damping="0.4" armature="0.01" frictionloss="0.01"/>
      <position kp="60" kv="1.5" forcerange="-6 6"/>
    </default>
    <!-- Robot links collide with the floor and objects, but not with each other -->
    <default class="link">
      <geom type="mesh" contype="2" conaffinity="1" density="800"/>
    </default>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.84 0.87" rgb2="0.74 0.76 0.8"
             width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.05"/>
{chr(10).join(assets)}
  </asset>

  <worldbody>
    <light pos="0.4 -0.4 1.2" dir="-0.3 0.3 -1" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="1 1 0.05" material="grid" pos="0 0 -0.03"
          contype="1" conaffinity="3"/>
    <!-- Target marker: move it with data.mocap_pos (visual only) -->
    <body name="target" mocap="true" pos="0.15 -0.1 0.2">
      <geom type="sphere" size="0.012" rgba="1 0.2 0.2 0.6" contype="0" conaffinity="0"/>
    </body>
{chr(10).join(body_xml)}
  </worldbody>

  <actuator>
{chr(10).join(act_xml)}
  </actuator>

  <keyframe>
    <key name="home" qpos="0 0 0 0 0 0" ctrl="0 0 0 0 0 0"/>
  </keyframe>
</mujoco>
"""
    out = out_dir / f"mycobot_280_{variant}.xml"
    out.write_text(xml)
    print("wrote", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to a clone of elephantrobotics/mycobot_ros")
    ap.add_argument("--variant", default="m5", choices=VARIANTS)
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parents[1] / "models"))
    ap.add_argument("--max-faces", type=int, default=20000, help="decimate meshes above this")
    a = ap.parse_args()
    build(a.repo, a.variant, pathlib.Path(a.out), a.max_faces)
