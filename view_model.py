"""
Open the myCobot 280 in the MuJoCo viewer.

Use the Control panel sliders on the right to drive each servo (values in radians).
    python view_model.py                       # M5 model
    python view_model.py models/mycobot_280_pi.xml
On macOS use `mjpython view_model.py`.
"""
import sys, pathlib
import mujoco, mujoco.viewer

HERE = pathlib.Path(__file__).resolve().parent
path = sys.argv[1] if len(sys.argv) > 1 else HERE / "models" / "mycobot_280_m5.xml"
m = mujoco.MjModel.from_xml_path(str(path))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.viewer.launch(m, d)
