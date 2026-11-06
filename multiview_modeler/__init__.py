bl_info = {
    "name": "Multiview Modeler",
    "author": "ai-character-modeling",
    "version": (0, 9, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar (N) > Multiview",
    "description": "Mesh from normal maps of several views: live preview with sliders, final build at full resolution",
    "category": "Mesh",
}

import fnmatch, json, math, os, queue, re, shutil, stat, subprocess, sys, threading, time
import bpy
import numpy as np
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty,
                       StringProperty)

CORE = os.path.join(os.path.dirname(__file__), "core")
RESULT, PREVIEW, UNDERLAY, REF = "MVM_Result", "MVM_Preview", "MVM_Underlay", "MVM_Ref"
PKGS = ["numpy", "scipy", "scikit-image", "pillow"]
PATH_OPTS = {"PATH_SUPPORTS_BLEND_RELATIVE"} if bpy.app.version >= (4, 5, 0) else set()   # "//" paths are fine, no red field
NOWIN = 0x08000000 if os.name == "nt" else 0                     # CREATE_NO_WINDOW


# =============================================================================================== helpers
def deps_dir():
    return os.path.join(bpy.utils.user_resource("SCRIPTS"), "mvm_deps")


def py_env():
    env = dict(os.environ); env["PYTHONPATH"] = deps_dir(); env.pop("PYTHONHOME", None)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):    # matrix maths on all cores
        env.setdefault(k, str(os.cpu_count() or 2))
    env["PYTHONIOENCODING"] = "utf-8"; env["PYTHONUTF8"] = "1"
    return env


def neural_dir():
    return os.path.join(bpy.utils.user_resource("SCRIPTS"), "mvm_neural")


def neural_py():
    v = os.path.join(neural_dir(), "venv")
    return os.path.join(v, "Scripts", "python.exe") if os.name == "nt" else os.path.join(v, "bin", "python")


def neural_env():
    env = dict(os.environ); env.pop("PYTHONPATH", None); env.pop("PYTHONHOME", None)
    env["PYTHONIOENCODING"] = "utf-8"; env["PYTHONUTF8"] = "1"
    return env


def popen_kw(stdin=False, env=None):
    kw = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
              bufsize=1, env=env or py_env(), creationflags=NOWIN)
    if stdin:
        kw["stdin"] = subprocess.PIPE
    return kw


def default_work_dir():
    return os.path.join(bpy.app.tempdir or os.path.expanduser("~"), "mvm_work")


def work_dir_path(s):
    """the work folder without creating it: "//..." needs a saved .blend, otherwise Blender's temp folder"""
    w = s.work_dir or ""
    if not w or (w.startswith("//") and not bpy.data.filepath):
        return default_work_dir()
    p = bpy.path.abspath(w)
    return os.path.normpath(p) if os.path.isabs(p) else default_work_dir()


def work_dir(s):
    p = work_dir_path(s)
    try:
        os.makedirs(p, exist_ok=True)
    except OSError:                                              # no write access (e.g. under Program Files)
        p = default_work_dir(); os.makedirs(p, exist_ok=True)
    return p


def redraw():
    for w in bpy.context.window_manager.windows:
        for ar in w.screen.areas:
            if ar.type == "VIEW_3D":
                ar.tag_redraw()


def set_attr_shading(on):
    for w in bpy.context.window_manager.windows:
        for ar in w.screen.areas:
            if ar.type == "VIEW_3D":
                sh = ar.spaces.active.shading
                if on and sh.type == "SOLID":
                    sh.color_type = "VERTEX"
                elif not on and sh.color_type == "VERTEX":
                    sh.color_type = "MATERIAL"


def load_npz_into_object(path, name, diag="NONE"):
    with np.load(path) as d:
        V = d["V"].astype(np.float32); F = d["F"].astype(np.int32)
        err = d["err"] if "err" in d.files else None; seen = d["seen"] if "seen" in d.files else None
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V)); me.vertices.foreach_set("co", V.ravel())
    me.loops.add(F.size); me.loops.foreach_set("vertex_index", F.ravel())
    me.polygons.add(len(F)); me.polygons.foreach_set("loop_start", np.arange(0, F.size, 3, dtype=np.int32))
    try:
        me.polygons.foreach_set("loop_total", np.full(len(F), 3, np.int32))    # Blender < 4.0
    except (AttributeError, RuntimeError, TypeError):
        pass
    me.update(calc_edges=True); me.validate(clean_customdata=False)
    if diag != "NONE" and err is not None:
        if diag == "ERROR":                                      # 0 deg blue -> 30+ deg red
            t = np.clip(err / 30.0, 0, 1); col = np.c_[t, 1 - np.abs(t - 0.5) * 2, 1 - t, np.ones_like(t)]
        else:                                                    # red = no view sees it, green = 2+ views
            t = np.clip(seen / 2.0, 0, 1); col = np.c_[1 - t, t, 0.15 * np.ones_like(t), np.ones_like(t)]
        ca = me.color_attributes.new("MVM_diag", "FLOAT_COLOR", "POINT")
        ca.data.foreach_set("color", col.astype(np.float32).ravel()); me.color_attributes.active_color = ca
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(obj)
    else:
        old = obj.data; obj.data = me
        if old.users == 0:
            bpy.data.meshes.remove(old)
    obj.hide_viewport = False
    return obj


def view_list(s):
    return [{"angle": float(v.angle), "path": bpy.path.abspath(v.normal_map), "enabled": bool(v.enabled),
             "dx": int(v.dx), "dy": int(v.dy), "fmt": v.fmt.lower(),
             "depth": bpy.path.abspath(v.depth_map) if v.depth_map else "", "depth_range": float(s.depth_range),
             "depth_invert": bool(s.depth_invert)} for v in s.views if v.normal_map]


def params(s, preview):
    return {"max_carve": s.max_carve / 100.0, "carve_above": s.carve_above, "use_carve": s.use_carve, "thin": s.thin,
            "rim": 3, "soft": 0.8, "break_jump": 0.6, "smooth": s.smooth, "smooth_thr": 0.004,
            "fit_iters": (s.iters_preview if (s.start_mode != "DEPTH" or s.dn_fit) else 0) if preview else 0, "lam0": 0.3, "lam1": 0.1, "mu": 0.05,
            "agree": 25.0, "relax": s.relax, "diag": bool(preview and s.diag != "NONE"),
            "sil_iters": s.sil_iters if s.use_sil else 0, "sil_step": 0.8, "sil_spread": s.sil_spread,
            "sil_smooth": s.sil_smooth, "sil_res": int(512),
            "dn_res": int(s.dn_res_preview if preview else s.dn_res_final), "dn_maps": min(1024, 4 * int(s.dn_res_preview if preview else s.dn_res_final)),
            "dn_smooth": bool(s.dn_smooth), "dn_auto": bool(s.dn_auto), "dn_scale": bool(s.dn_scale),
            "dn_max_carve": float(s.dn_max_carve) / 100.0, "clean": bool(s.clean_floating), "trim": bool(s.trim_outside)}


def check_deps():
    try:
        r = subprocess.run([sys.executable, "-c", "import scipy, skimage, PIL"], env=py_env(), capture_output=True,
                           timeout=60, creationflags=NOWIN)
        return r.returncode == 0
    except Exception:
        return False


def _reader(proc, q):
    try:
        for line in proc.stdout:
            q.put(line.rstrip())
    except Exception as e:                                       # never leave a job waiting forever
        q.put(f"reader error: {e}")
    finally:
        q.put(None)


# =============================================================================================== batch job runner
JOB = {"proc": None, "cmds": [], "on_line": None, "on_done": None, "q": None, "cancel": False, "log": []}


def start_job(cmds, on_line=None, on_done=None):
    if JOB["proc"] is not None:
        return False
    JOB.update(cmds=list(cmds), on_line=on_line, on_done=on_done, cancel=False, log=[])
    bpy.ops.mvm.run_job("INVOKE_DEFAULT")
    return True


class MVM_OT_run_job(bpy.types.Operator):
    bl_idname = "mvm.run_job"; bl_label = "Run job"; bl_options = {"INTERNAL"}
    _timer = None

    def _next(self, context):
        if not JOB["cmds"]:
            return False
        cmd = JOB["cmds"].pop(0); JOB["q"] = queue.Queue(); env = None
        if isinstance(cmd, tuple):                                # (command, environment)
            cmd, env = cmd
        JOB["proc"] = subprocess.Popen(cmd, **popen_kw(env=env))
        threading.Thread(target=_reader, args=(JOB["proc"], JOB["q"]), daemon=True).start()
        context.scene.mvm.status = "running: " + os.path.basename(cmd[1] if len(cmd) > 1 else cmd[0])
        return True

    def invoke(self, context, event):
        context.scene.mvm.running = True
        self._next(context)
        self._timer = context.window_manager.event_timer_add(0.4, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _finish(self, context, ok, msg):
        s = context.scene.mvm; s.running = False; s.status = msg
        context.window_manager.event_timer_remove(self._timer); JOB["proc"] = None
        if JOB["on_done"]:
            try:
                JOB["on_done"](context, ok)
            except Exception as e:
                s.status = f"error after job: {e}"
        redraw()
        return {"FINISHED"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        s = context.scene.mvm; p = JOB["proc"]
        if JOB["cancel"] and p is not None:
            p.kill(); JOB["cmds"].clear(); return self._finish(context, False, "cancelled")
        ended = False
        while True:
            try:
                line = JOB["q"].get_nowait()
            except queue.Empty:
                break
            if line is None:
                ended = True; break
            JOB["log"].append(line); JOB["log"] = JOB["log"][-200:]
            if line.startswith("PROGRESS "):
                _, i, n, *txt = line.split(" ")
                s.progress = int(i) / max(1, int(n)); s.status = " ".join(txt)
            elif line.startswith("STATUS "):                       # stage text; "... NN %" also moves the bar
                s.status = line[7:]; mt = re.search(r"(\d+) %$", line)
                if mt:
                    s.progress = int(mt.group(1)) / 100.0
            if JOB["on_line"]:
                try:
                    JOB["on_line"](context, line)
                except Exception as e:                            # a preview problem must not stop the job
                    s.status = f"preview skipped: {e}"[:120]
        if ended:
            rc = p.wait()
            if rc != 0:
                tail = " | ".join(JOB["log"][-3:])
                JOB["cmds"].clear(); return self._finish(context, False, f"failed ({rc}): {tail[-300:]}")
            if not self._next(context):
                return self._finish(context, True, "done")
        redraw()
        return {"PASS_THROUGH"}


class MVM_OT_cancel(bpy.types.Operator):
    bl_idname = "mvm.cancel"; bl_label = "Cancel"
    def execute(self, context):
        JOB["cancel"] = True; return {"FINISHED"}


# =============================================================================================== live mode
LIVE = {"proc": None, "q": None, "dirty": 0.0, "sent": 0, "stop": False}


def mark_dirty(self=None, context=None):
    if LIVE["proc"] is not None:
        LIVE["dirty"] = time.time()


def live_progress(txt):
    """rough share of a preview rebuild from the worker's stage text"""
    m = re.search(r"(\d+) %$", txt)
    if m:
        return int(m.group(1)) / 100.0
    m = re.search(r"(\d+)/(\d+)", txt); f = int(m.group(1)) / max(1, int(m.group(2))) if m else 0.0
    for key, a, b in (("aligning", 0.02, 0.05), ("integrating", 0.05, 0.25), ("fusing", 0.3, 0.35), ("silhouette fit", 0.35, 0.7),
                      ("silhouette IoU", 0.7, 0.7), ("fit ", 0.7, 0.98)):
        if txt.startswith(key) or key in txt[:16]:
            return a + (b - a) * f
    return None


class MVM_OT_live_start(bpy.types.Operator):
    bl_idname = "mvm.live_start"; bl_label = "Start live preview"
    bl_description = "A background process keeps the maps in memory; every slider change rebuilds a low-resolution preview"
    _timer = None

    def invoke(self, context, event):
        if LIVE["proc"] is not None:
            return {"CANCELLED"}
        s = context.scene.mvm
        if len([v for v in view_list(s) if v["enabled"]]) < 2:
            self.report({"ERROR"}, "need normal maps for at least 2 enabled views"); return {"CANCELLED"}
        if not check_deps():
            self.report({"ERROR"}, "install the dependencies first"); return {"CANCELLED"}
        LIVE.update(proc=subprocess.Popen([sys.executable, os.path.join(CORE, "live_worker.py"), "--live"], **popen_kw(stdin=True)),
                    q=queue.Queue(), stop=False, dirty=time.time() - 10)
        threading.Thread(target=_reader, args=(LIVE["proc"], LIVE["q"]), daemon=True).start()
        s.live_on = True; s.live_status = "first build: loading and integrating all views"
        self._timer = context.window_manager.event_timer_add(0.2, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _send(self, s):
        LIVE["sent"] += 1; wd = work_dir(s); s.live_progress = 0.0
        req = {"id": LIVE["sent"], "out": os.path.join(wd, "preview.npz"), "height": s.height, "res": s.res_preview,
               "fit_res": int(s.fit_res_preview), "views": view_list(s), "p": params(s, True),
               "aligned_dir": os.path.join(wd, "aligned_preview")}
        st = os.path.join(wd, "start.npz")
        if s.start_mode == "DEPTH":
            req["depth_mode"] = True
        elif s.start_mode != "VOLUME" and os.path.exists(st):
            req["start"] = st
        try:
            LIVE["proc"].stdin.write(json.dumps(req) + "\n"); LIVE["proc"].stdin.flush()
        except OSError:
            LIVE["stop"] = True

    def _end(self, context):
        context.window_manager.event_timer_remove(self._timer); p = LIVE["proc"]
        try:
            p.stdin.write('{"cmd": "quit"}\n'); p.stdin.flush(); p.wait(timeout=3)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        LIVE["proc"] = None; s = context.scene.mvm; s.live_on = False
        if not s.live_status.startswith("worker"):
            s.live_status = "stopped"
        redraw()
        return {"FINISHED"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        s = context.scene.mvm
        if LIVE["stop"]:
            return self._end(context)
        if LIVE["dirty"] and time.time() - LIVE["dirty"] > s.delay:
            LIVE["dirty"] = 0.0; self._send(s)
        tail = []
        while True:
            try:
                line = LIVE["q"].get_nowait()
            except queue.Empty:
                break
            if line is None:
                s.live_status = "worker stopped: " + " | ".join(tail[-2:])[:120]; LIVE["stop"] = True; break
            tail.append(line)
            if line.startswith("STATUS "):
                s.live_status = line[7:]; pr = live_progress(line[7:])
                if pr is not None:
                    s.live_progress = max(s.live_progress, pr)
            elif line.startswith("MESH "):
                st = json.loads(line.split(" ", 2)[2])
                try:
                    load_npz_into_object(os.path.join(work_dir(s), "preview.npz"), PREVIEW, s.diag)
                    set_attr_shading(s.diag != "NONE")
                except Exception as e:
                    s.live_status = f"preview load failed: {e}"[:120]; continue
                txt = f"{st['triangles']} tris, {st['total_s']} s"
                for k, lab in (("silhouette_iou", "outline IoU {}"), ("faces_vs_maps_deg", "vs maps {} deg"), ("folded_pct", "folded {} %"), ("unseen_pct", "unseen {} %")):
                    if k in st:
                        txt += ", " + lab.format(st[k])
                s.live_status = txt; s.live_progress = 1.0; update_overlay(s)
            elif line.startswith("ERR "):
                s.live_status = line.split(" ", 2)[-1][:150]
        redraw()
        return {"PASS_THROUGH"}


class MVM_OT_live_stop(bpy.types.Operator):
    bl_idname = "mvm.live_stop"; bl_label = "Stop live preview"
    def execute(self, context):
        LIVE["stop"] = True
        if LIVE["proc"] is None:                                   # flag left over (reinstall, reopened file): nothing to stop
            s = context.scene.mvm; s.live_on = False; s.live_status = "stopped"; redraw()
        return {"FINISHED"}


# =============================================================================================== overlay
def update_overlay(s):
    obj = bpy.data.objects.get(REF)
    if not s.overlay:
        if obj is not None:
            obj.hide_viewport = True
        return
    if not s.views or not (0 <= s.view_index < len(s.views)):
        return
    v = s.views[s.view_index]; p = os.path.join(work_dir(s), "aligned_preview", f"n_{int(v.angle)}.png")
    if not os.path.exists(p):
        return
    img = bpy.data.images.load(p, check_existing=True); img.reload()
    if obj is None:
        obj = bpy.data.objects.new(REF, None); bpy.context.scene.collection.objects.link(obj)
        obj.empty_display_type = "IMAGE"
    obj.data = img; obj.hide_viewport = False
    obj.empty_display_size = s.height / (1 - 2 * 0.045)          # aligned frame: the height is 91 % of the image
    obj.location = (0, 0, s.height / 2)
    obj.rotation_euler = (math.pi / 2, 0, math.radians(v.angle))
    obj.use_empty_image_alpha = True; obj.color[3] = s.overlay_opacity
    obj.empty_image_depth = "FRONT"; obj.empty_image_side = "FRONT"


def _ov(self, context):
    try:
        update_overlay(context.scene.mvm)
    except Exception:
        pass


def _show_underlay(self, context=None):
    o = bpy.data.objects.get(UNDERLAY)
    if o is not None:
        o.hide_viewport = not self.show_underlay


# =============================================================================================== data
class MVM_View(bpy.types.PropertyGroup):
    angle: FloatProperty(name="Angle", default=0.0, min=0, max=359, description="0 = front, 90 = character's left side, 180 = back", update=mark_dirty)
    normal_map: StringProperty(name="Normal map", subtype="FILE_PATH", options=PATH_OPTS, description="Normal map of this view (made by the tool, never drawn)", update=mark_dirty)
    measure: StringProperty(name="Measurement view", subtype="FILE_PATH", options=PATH_OPTS,
                            description="White clay under red-left / green-top / blue-right lights -> normal map")
    enabled: BoolProperty(name="Use", default=True, description="Exclude a view that contradicts the others", update=mark_dirty)
    dx: IntProperty(name="Shift X", default=0, min=-20, max=20, description="Manual horizontal correction, px", update=mark_dirty)
    dy: IntProperty(name="Shift Y", default=0, min=-20, max=20, description="Manual vertical correction, px", update=mark_dirty)
    fmt: EnumProperty(name="Format", items=[("AUTO", "Auto", ""), ("OPENGL", "OpenGL", "green up"), ("DIRECTX", "DirectX", "green down")],
                      default="AUTO", update=mark_dirty)
    depth_map: StringProperty(name="Depth map", subtype="FILE_PATH", options=PATH_OPTS, update=mark_dirty,
                              description="Depth of this view, same framing as its normal map: .npy (world units toward the camera, NaN = background) or 16-bit PNG (0 = background)")


class MVM_Settings(bpy.types.PropertyGroup):
    views: CollectionProperty(type=MVM_View)
    view_index: IntProperty(default=0, update=_ov)
    # common
    height: FloatProperty(name="Height", default=2.2, min=0.01, max=100, description="Character height in scene units", update=mark_dirty)
    work_dir: StringProperty(name="Work folder", subtype="DIR_PATH", options=PATH_OPTS, default="//mvm_work")
    delay: FloatProperty(name="Update delay, s", default=0.5, min=0.2, max=2.0)
    # volume
    res_preview: IntProperty(name="Preview res", default=320, min=200, max=500, update=mark_dirty)
    res_final: IntProperty(name="Final res", default=640, min=400, max=1200)
    max_carve: FloatProperty(name="Max carve depth, % height", default=2.5, min=0.0, max=6.0, precision=2,
                             description="How deep the surface may go inside the silhouette hull (0 = no limit). Low keeps arms, tail and torso whole; high gives deeper ear cups and gaps", update=mark_dirty)
    use_carve: BoolProperty(name="Carve thin above", default=True, description="Ears, buns, strands carve by depth above this height; below - only silhouettes", update=mark_dirty)
    carve_above: FloatProperty(name="", default=0.56, min=0.0, max=1.0, subtype="FACTOR", update=mark_dirty)
    thin: FloatProperty(name="Thin part threshold, px", default=9.0, min=0.0, max=30.0, description="Parts thinner than ~2x this (px at 800) do not carve", update=mark_dirty)
    smooth: IntProperty(name="Smoothing iterations", default=12, min=0, max=30, update=mark_dirty)
    # fit
    fit_res_preview: EnumProperty(name="Maps (preview)", items=[("512", "512", ""), ("1024", "1024", "")], default="512", update=mark_dirty)
    fit_res_final: EnumProperty(name="Maps (final)", items=[("1024", "1024", ""), ("2048", "2048", "")], default="1024")
    iters_preview: IntProperty(name="Iterations (preview)", default=1, min=0, max=12, update=mark_dirty)
    iters_final: IntProperty(name="Iterations (final)", default=3, min=0, max=12)
    relax: FloatProperty(name="Remesh (relax)", default=0.3, min=0.0, max=0.6, description="Evens out triangles, removes streaks", update=mark_dirty)
    use_detail: BoolProperty(name="Finer mesh above", default=False, description="Final build: 1 -> 4 split above this height")
    detail_above: FloatProperty(name="", default=0.56, min=0.0, max=1.0, subtype="FACTOR")
    # start mesh
    start_mode: EnumProperty(name="Start mesh", items=[("VOLUME", "Volume from views", "silhouettes + depth of all views"),
                                                       ("HUNYUAN", "Hunyuan3D 2.0", "shape neural net from one colour image (or front + left + back); needs ~6 GB video memory"),
                                                       ("DEPTH", "Depth + normals", "method C2-C: depth map + normal map of every view, sub-voxel surface")],
                             default="VOLUME", update=mark_dirty)
    color_image: StringProperty(name="Front colour image", subtype="FILE_PATH", options=PATH_OPTS, description="Colour view used by the neural net (front is best)")
    fg_ratio: FloatProperty(name="Object size in frame", default=0.85, min=0.6, max=0.95, description="How much of the square input the character takes")
    hy_variant: EnumProperty(name="Model", items=[("mini", "Hunyuan3D-2 mini", "lightest, fastest"),
                                                  ("std", "Hunyuan3D-2", "standard, 6 GB for the shape"),
                                                  ("mv", "Hunyuan3D-2 multi-view", "front + left + back images (experimental)")], default="mini")
    color_left: StringProperty(name="Left colour image", subtype="FILE_PATH", options=PATH_OPTS, description="Multi-view model: view of the character's left side (90 deg)")
    color_back: StringProperty(name="Back colour image", subtype="FILE_PATH", options=PATH_OPTS, description="Multi-view model: view from behind (180 deg)")
    hy_octree: IntProperty(name="Detail grid", default=256, min=128, max=512, description="Resolution of the surface grid (detail vs memory)")
    hy_steps: IntProperty(name="Steps", default=30, min=10, max=60, description="Denoising steps (more = slower, slightly cleaner)")
    use_sil: BoolProperty(name="Fit to silhouettes", default=True, description="Push / pull the neural start so its outline matches every view", update=mark_dirty)
    sil_iters: IntProperty(name="Silhouette passes", default=10, min=0, max=60, description="More = closer to the outlines (about 0.7 s per pass)", update=mark_dirty)
    sil_spread: IntProperty(name="Move whole parts", default=40, min=0, max=80, description="How far a correction spreads over the mesh: 0 = only the rim moves, high = whole arms / back move", update=mark_dirty)
    sil_smooth: IntProperty(name="Silhouette smoothing", default=2, min=0, max=10, description="Smoothing after each pass", update=mark_dirty)
    depth_range: FloatProperty(name="PNG depth range", default=1.0, min=0.1, max=4.0, update=mark_dirty,
                               description="16-bit PNG: the full value range covers this many character heights, mid-grey = the vertical axis")
    depth_invert: BoolProperty(name="Bright = far", default=False, update=mark_dirty, description="Tick if the depth maps are brighter further from the camera")
    dn_auto: BoolProperty(name="Auto depth offset", default=True, update=mark_dirty,
                          description="Shift each view's depth so it touches the silhouettes from inside (needed for generated depth)")
    dn_scale: BoolProperty(name="Auto depth scale", default=True, update=mark_dirty,
                           description="Find each view's depth scale from its normal map (relative depth). Off for exact depth and for maps from Bake maps (already scaled)")
    dn_smooth: BoolProperty(name="Smoothing (off = C2-B)", default=True, update=mark_dirty,
                            description="On: C2-C, feature-preserving smoothing along the normal, at most 0.3 voxel (mean error 0.214 %). Off: C2-B, raw sub-voxel surface (normals 4.8 deg)")
    dn_max_carve: FloatProperty(name="Max carve depth, % height", default=3.0, min=0.0, max=20.0, precision=1, update=mark_dirty,
                                description="Generated depth: a view may not carve deeper than this under the silhouette hull (tested: 3.1 -> 1.08 % on baked maps). "
                                            "0 = no limit, for exact depth from a 3D model (the limit costs 0.21 -> 0.71 % there)")
    dn_res_preview: EnumProperty(name="Grid preview", items=[("96", "96", ""), ("128", "128", ""), ("192", "192", ""), ("256", "256", "")], default="128", update=mark_dirty)
    dn_res_final: EnumProperty(name="Grid final", items=[("192", "192", ""), ("256", "256", "tested: 0.21 % of height"),
                                                         ("384", "384", "thin parts (strands, ribbons) survive; ~3x time"),
                                                         ("512", "512", "finest: ahoge, laces; ~8x time, ~3 GB RAM")], default="256")
    dn_fit: BoolProperty(name="Also fit to normal maps", default=False, update=mark_dirty,
                         description="Run the normal-map fit on top (usually not needed: the sub-voxel surface is already more accurate)")
    # bake maps from colour views
    bake_dir: StringProperty(name="Views folder", subtype="DIR_PATH", options=PATH_OPTS,
                             description="Folder with view_000.png ... view_315.png - the 3D-render views made with the 8-view skill")
    bake_normals: EnumProperty(name="Normals", items=[("MARIGOLD", "Marigold", "separate normal net (better detail, ~2 GB)"),
                                                     ("DEPTH", "From depth", "normals from the depth gradient (faster, softer)")], default="MARIGOLD")
    bake_quality: EnumProperty(name="Quality", items=[("FAST", "Fast", "Marigold 1 step"), ("NORMAL", "Normal", "4 steps x 3"),
                                                     ("BEST", "Best", "10 steps x 5, slow")], default="NORMAL")
    relief_inverted: BoolProperty(default=False)
    trim_outside: BoolProperty(name="Trim outside silhouettes", default=True, update=mark_dirty,
                               description="All methods: cut triangles that stick out of the silhouette of any view by more than 2 px "
                                           "(thin slivers attached to the body). The real surface is inside all silhouettes")
    clean_floating: BoolProperty(name="Remove floating parts", default=True, update=mark_dirty,
                                 description="All methods: delete pieces of the mesh that are not connected to the body (slivers, crumbs in the air). "
                                             "Separate parts at least 1 % of the main body are kept")
    bake_clay: BoolProperty(name="Convert to clay before baking", default=False,
                            description="Remove the colours of every view (albedo of each material divided out) so the generators see only shading, like a white clay render. "
                                        "Saved as clay_XXX.png. Painted lines stay; dark materials give noisier shading")
    bake_delight: BoolProperty(name="Remove drawn lighting", default=False,
                               description="Even out highlights and shadows painted into the views before baking: the generators read painted light as shape. "
                                           "Can also flatten useful relief - check the maps sheet")
    # game mesh
    poly_limit: IntProperty(name="Polygon limit", default=70000, min=1000, max=2000000)
    keep_copy: BoolProperty(name="Keep full copy", default=True, description="Copy before decimation, for normal-map baking")
    # display
    show_underlay: BoolProperty(name="Show start mesh", default=False, update=_show_underlay)
    diag: EnumProperty(name="Highlight", items=[("NONE", "None", ""), ("ERROR", "Disagreement with views", "blue = agrees, red = 30+ deg"),
                                               ("UNSEEN", "Guessed areas", "red = no view sees it")], default="NONE", update=mark_dirty)
    overlay: BoolProperty(name="Overlay selected view", default=False, update=_ov)
    overlay_opacity: FloatProperty(name="Opacity", default=0.5, min=0.05, max=1.0, subtype="FACTOR", update=_ov)
    # state
    live_on: BoolProperty(default=False)
    live_status: StringProperty(default="")
    live_progress: FloatProperty(default=0.0, subtype="FACTOR", min=0, max=1)
    running: BoolProperty(default=False)
    progress: FloatProperty(default=0.0, subtype="FACTOR", min=0, max=1)
    status: StringProperty(default="")
    report: StringProperty(default="")


# =============================================================================================== operators
class MVM_OT_standard_views(bpy.types.Operator):
    bl_idname = "mvm.standard_views"; bl_label = "8 standard views"; bl_description = "0, 45, ... 315 degrees"
    def execute(self, context):
        s = context.scene.mvm; s.views.clear()
        for a in range(0, 360, 45):
            s.views.add().angle = a
        return {"FINISHED"}


class MVM_OT_view_add(bpy.types.Operator):
    bl_idname = "mvm.view_add"; bl_label = "Add view"
    def execute(self, context):
        s = context.scene.mvm; v = s.views.add(); v.angle = (s.views[-2].angle + 45) % 360 if len(s.views) > 1 else 0
        s.view_index = len(s.views) - 1; return {"FINISHED"}


class MVM_OT_view_remove(bpy.types.Operator):
    bl_idname = "mvm.view_remove"; bl_label = "Remove view"
    def execute(self, context):
        s = context.scene.mvm
        if s.views:
            s.views.remove(s.view_index); s.view_index = max(0, s.view_index - 1); mark_dirty()
        return {"FINISHED"}


class MVM_OT_reset(bpy.types.Operator):
    bl_idname = "mvm.reset"; bl_label = "Reset"; bl_description = "All sliders back to the tested defaults"
    def execute(self, context):
        s = context.scene.mvm
        keep = {"views", "view_index", "height", "work_dir", "live_on", "live_status", "running", "progress", "status", "report", "rna_type", "name"}
        for p in s.bl_rna.properties:
            if p.identifier not in keep and not p.is_readonly:
                s.property_unset(p.identifier)
        mark_dirty(); return {"FINISHED"}


_FIT = ("use_sil", "sil_iters", "sil_spread", "sil_smooth", "iters_preview", "iters_final", "relax", "use_detail", "detail_above",
        "fit_res_preview", "fit_res_final")
METHOD_PROPS = {
    "VOLUME": ("res_preview", "res_final", "max_carve", "use_carve", "carve_above", "thin", "smooth") + _FIT,
    "HUNYUAN": ("fg_ratio", "hy_variant", "hy_octree", "hy_steps", "color_left", "color_back") + _FIT,
    "DEPTH": ("dn_max_carve", "dn_smooth", "dn_scale", "dn_auto", "depth_range", "depth_invert", "dn_res_preview", "dn_res_final", "dn_fit"),
}


class MVM_OT_reset_method(bpy.types.Operator):
    bl_idname = "mvm.reset_method"; bl_label = "Reset method settings"
    bl_description = "Settings of this method back to the tested defaults (silhouette and normal-map fit are shared by methods 1 and 2)"
    mode: StringProperty(default="VOLUME")
    def execute(self, context):
        s = context.scene.mvm
        for k in METHOD_PROPS.get(self.mode, ()):
            s.property_unset(k)
        mark_dirty(); self.report({"INFO"}, "method settings reset"); return {"FINISHED"}


class MVM_OT_install_deps(bpy.types.Operator):
    bl_idname = "mvm.install_deps"; bl_label = "Install dependencies"
    bl_description = "numpy, scipy, scikit-image, pillow into a separate folder used only by the background process (~150 MB, once)"
    def execute(self, context):
        os.makedirs(deps_dir(), exist_ok=True); py = sys.executable
        ensure = ("import subprocess, sys\n"
                  "if subprocess.run([sys.executable, '-m', 'pip', '--version']).returncode:\n"
                  "    subprocess.run([sys.executable, '-m', 'ensurepip', '--user'])")
        cmds = [[py, "-c", ensure], [py, "-m", "pip", "install", "--upgrade", "--target", deps_dir()] + PKGS]
        def done(ctx, ok):
            if ok:
                ctx.scene.mvm.status = "dependencies installed"
        start_job(cmds, on_done=done); return {"FINISHED"}


class MVM_OT_make_normals(bpy.types.Operator):
    bl_idname = "mvm.make_normals"; bl_label = "Normals from measurement views"
    bl_description = "shading_to_normals --rgb-lights for every view that has a measurement image and no normal map"
    def execute(self, context):
        s = context.scene.mvm; wd = work_dir(s); cmds, todo = [], []
        for i, v in enumerate(s.views):
            if v.measure and not v.normal_map:
                out = os.path.join(wd, f"normal_{int(v.angle)}.png")
                cmds.append([sys.executable, os.path.join(CORE, "shading_to_normals.py"), bpy.path.abspath(v.measure), out, "--rgb-lights"])
                todo.append((i, out))
        if not cmds:
            self.report({"WARNING"}, "no view has a measurement image without a normal map"); return {"CANCELLED"}
        def done(ctx, ok):
            for i, out in todo:
                if os.path.exists(out):
                    ctx.scene.mvm.views[i].normal_map = out
        start_job(cmds, on_done=done); return {"FINISHED"}


class MVM_OT_build(bpy.types.Operator):
    bl_idname = "mvm.build"; bl_label = "Final build"
    bl_description = "The current parameters at full resolution: volume, then the fit to all views (background)"
    def execute(self, context):
        s = context.scene.mvm; views = view_list(s)
        missing = [v["path"] for v in views if not os.path.exists(v["path"])]
        if len([v for v in views if v["enabled"]]) < 2 or missing:
            self.report({"ERROR"}, f"need at least 2 enabled views with normal maps; missing: {missing[:2]}"); return {"CANCELLED"}
        if not check_deps():
            self.report({"ERROR"}, "install the dependencies first"); return {"CANCELLED"}
        if LIVE["proc"] is not None:
            LIVE["stop"] = True                                   # both are CPU-heavy
        wd = work_dir(s); py = sys.executable; al = os.path.join(wd, "aligned")
        job = {"work": wd, "height": s.height, "res": s.res_final, "fit_res": int(s.fit_res_final), "views": views, "p": params(s, False)}
        if s.start_mode == "DEPTH":
            nod = [v["angle"] for v in views if v["enabled"] and not (v["depth"] and os.path.exists(v["depth"]))]
            if nod:
                self.report({"ERROR"}, f"depth maps missing for views {nod}"); return {"CANCELLED"}
            job["depth_mode"] = True
            if not s.dn_fit:
                job["no_fit_maps"] = True                         # aligned 2048 maps are only for the normal-map fit
        elif s.start_mode != "VOLUME":
            if not os.path.exists(os.path.join(wd, "start.npz")):
                self.report({"ERROR"}, "run the neural start first (Start mesh panel)"); return {"CANCELLED"}
            job["start"] = os.path.join(wd, "start.npz")
        jp = os.path.join(wd, "job.json")
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(job, f, indent=1)
        cmds = [[py, os.path.join(CORE, "live_worker.py"), "--final", jp]]
        n = s.iters_final if (s.start_mode != "DEPTH" or s.dn_fit) else 0
        if n > 0:
            lams = ",".join(f"{0.3 + (0.1 - 0.3) * (k / max(1, n - 1)):.3f}" for k in range(n))
            fit = [py, os.path.join(CORE, "multiview_fit.py"), os.path.join(wd, "underlay.npz"), os.path.join(wd, "fit.npz"),
                   "--height", str(s.height), "--iters", str(n), "--lam", lams, "--mu", str(0.05), "--agree", str(25.0),
                   "--relax", str(s.relax), "--report", os.path.join(wd, "fit.json")]
            for v in views:
                if v["enabled"]:
                    fit += ["--view", f"{v['angle']:g}:{os.path.join(al, 'n_%d.png' % int(v['angle']))}"]
            if s.use_detail:
                fit += ["--subdivide-above", str(s.detail_above)]
            cmds.append(fit)
        prev = os.path.join(wd, "fit.npz.preview.npz"); s.progress = 0.0; s.report = ""

        def on_line(ctx, line):
            if line.startswith("underlay.obj:"):
                o = load_npz_into_object(os.path.join(wd, "underlay.npz"), UNDERLAY); o.hide_viewport = not ctx.scene.mvm.show_underlay
                if n == 0:
                    load_npz_into_object(os.path.join(wd, "underlay.npz"), RESULT)
            if line.startswith("PROGRESS ") and not line.startswith("PROGRESS 0 ") and n > 0 and os.path.exists(prev):
                load_npz_into_object(prev, RESULT)

        def on_done(ctx, ok):
            if ok and n > 0 and os.path.exists(os.path.join(wd, "fit.npz")):
                load_npz_into_object(os.path.join(wd, "fit.npz"), RESULT)
                try:
                    with open(os.path.join(wd, "fit.json")) as f:
                        r = json.load(f)
                    last = r["iterations"][-1]
                    ctx.scene.mvm.report = (f"{r['triangles']} tris, folded {100 * r['folded_pairs_share']:.2f} %, "
                                            f"faces vs maps {last['median_face_vs_maps_deg']} deg")
                except Exception:
                    pass
            pv = bpy.data.objects.get(PREVIEW)
            if ok and pv is not None:
                pv.hide_viewport = True
        start_job(cmds, on_line=on_line, on_done=on_done); return {"FINISHED"}


class MVM_OT_install_neural(bpy.types.Operator):
    bl_idname = "mvm.install_neural"; bl_label = "Install neural start"
    bl_description = "Separate environment with PyTorch (CUDA) + Hunyuan3D code. About 4 GB, once; the model weights download on the first run"
    def execute(self, context):
        cmd = [sys.executable, os.path.join(CORE, "install_neural.py"), neural_dir()]
        env = neural_env()
        def done(ctx, ok):
            if ok:
                ctx.scene.mvm.status = "neural start installed: " + " ".join(l for l in JOB["log"] if l.startswith("DONE"))[:80]
        start_job([(cmd, env)], on_done=done); return {"FINISHED"}


class MVM_OT_neural_start(bpy.types.Operator):
    bl_idname = "mvm.neural_start"; bl_label = "Generate start mesh"
    bl_description = "Neural net from the front colour image, then automatic orientation / scale / offset from the view silhouettes"
    def execute(self, context):
        s = context.scene.mvm; wd = work_dir(s)
        if s.start_mode == "VOLUME":
            self.report({"ERROR"}, "choose Hunyuan3D 2.0 as the start mesh"); return {"CANCELLED"}
        img = bpy.path.abspath(s.color_image)
        if not img or not os.path.exists(img):
            self.report({"ERROR"}, "set the front colour image"); return {"CANCELLED"}
        repo = os.path.join(neural_dir(), "Hunyuan3D-2")
        if not os.path.exists(neural_py()) or not os.path.isdir(repo):
            self.report({"ERROR"}, "install the neural start first"); return {"CANCELLED"}
        if len([v for v in view_list(s) if v["enabled"]]) < 2 or not check_deps():
            self.report({"ERROR"}, "need dependencies and normal maps of at least 2 views (for the alignment)"); return {"CANCELLED"}
        raw = os.path.join(wd, "start_raw.npz")
        gen = [neural_py(), os.path.join(CORE, "neural_start.py"), "--image", img, "--out", raw,
               "--repo", repo, "--fg-ratio", str(s.fg_ratio), "--octree", str(s.hy_octree), "--steps", str(s.hy_steps), "--variant", s.hy_variant]
        if s.hy_variant == "mv":
            left, back = bpy.path.abspath(s.color_left), bpy.path.abspath(s.color_back)
            if not (os.path.exists(left) and os.path.exists(back)):
                self.report({"ERROR"}, "the multi-view model needs the left and back colour images"); return {"CANCELLED"}
            gen += ["--image-left", left, "--image-back", back]
        jp = os.path.join(wd, "align.json")
        with open(jp, "w", encoding="utf-8") as f:
            json.dump({"work": wd, "height": s.height, "views": view_list(s), "p": params(s, False),
                       "align_start": raw, "start_out": os.path.join(wd, "start.npz")}, f)
        al = [sys.executable, os.path.join(CORE, "live_worker.py"), "--final", jp]
        def done(ctx, ok):
            if ok and os.path.exists(os.path.join(wd, "start.npz")):
                o = load_npz_into_object(os.path.join(wd, "start.npz"), "MVM_Start"); o.hide_viewport = True
                info = [l for l in JOB["log"] if l.startswith("START ")]
                if info:
                    ctx.scene.mvm.report = "start mesh aligned, silhouette IoU " + str(json.loads(info[-1][6:]).get("silhouette_iou"))
                mark_dirty()
        start_job([(gen, neural_env()), al], on_done=done); return {"FINISHED"}


class MVM_OT_install_bake(bpy.types.Operator):
    bl_idname = "mvm.install_bake"; bl_label = "Install map bakers"
    bl_description = "The neural environment (PyTorch CUDA, shared with Hunyuan3D) + Depth Anything V2 and Marigold packages. Models download on the first bake"
    def execute(self, context):
        cmd = [sys.executable, os.path.join(CORE, "install_neural.py"), neural_dir(), "--maps"]
        def done(ctx, ok):
            if ok:
                ctx.scene.mvm.status = "map bakers installed: " + " ".join(l for l in JOB["log"] if l.startswith("DONE"))[:80]
        start_job([(cmd, neural_env())], on_done=done); return {"FINISHED"}


class MVM_OT_bake_maps(bpy.types.Operator):
    bl_idname = "mvm.bake_maps"; bl_label = "Bake maps"
    bl_description = ("Depth + normal map of every view_XXX.png in the folder, then sets the view list to the baked views (one per file). The chosen method is kept. "
                      "Axes, depth sign and scale are checked against the silhouettes and normals")
    def execute(self, context):
        s = context.scene.mvm; src = bpy.path.abspath(s.bake_dir)
        if not src or not os.path.isdir(src):
            self.report({"ERROR"}, "choose the folder with view_XXX.png"); return {"CANCELLED"}
        if not os.path.exists(neural_py()):
            self.report({"ERROR"}, "install the map bakers first"); return {"CANCELLED"}
        out = os.path.join(work_dir(s), "maps")
        cmd = [neural_py(), os.path.join(CORE, "bake_maps.py"), src, out, "--normals", s.bake_normals.lower(), "--quality", s.bake_quality.lower()]
        if s.bake_clay:
            cmd.append("--clay")
        elif s.bake_delight:
            cmd.append("--delight")
        def done(ctx, ok):
            if not ok:
                return
            st = ctx.scene.mvm; found = []
            for l in JOB["log"]:
                if l.startswith("BAKED "):
                    found = json.loads(l[6:])
            if not found:
                return
            st.views.clear()                          # as many views as baked files in the folder
            for a in sorted(found):
                v = st.views.add(); v.angle = a
                v.normal_map = os.path.join(out, f"normal_{a:03d}.png"); v.depth_map = os.path.join(out, f"depth_{a:03d}.png")
                v.fmt = "OPENGL"; v.enabled = True
            st.depth_range = 1.0; st.depth_invert = False; st.dn_auto = True; st.relief_inverted = False
            if st.dn_max_carve == 0:
                st.dn_max_carve = 3.0                 # generated depth: limit the carving (method 3)
            st.dn_scale = False                       # the baker already fitted the scale; a second estimate made it worse (0.46 vs 0.29 %)
            sheet = os.path.join(out, "maps_sheet.png")
            if os.path.exists(sheet):
                im = bpy.data.images.get("MVM_maps_sheet")
                if im:
                    im.filepath = sheet; im.reload()
                else:
                    bpy.data.images.load(sheet).name = "MVM_maps_sheet"
            st.status = f"baked {len(found)} views; check MVM_maps_sheet in the Image Editor"
            mark_dirty()
        start_job([(cmd, neural_env())], on_done=done); return {"FINISHED"}


class MVM_OT_invert_relief(bpy.types.Operator):
    bl_idname = "mvm.invert_relief"; bl_label = "Invert relief"
    bl_description = ("Turn the relief inside out in the normal and depth maps of every view (raised <-> recessed). "
                      "For when the generator read the picture backwards. Rewrites the files; press again to undo")
    def execute(self, context):
        s = context.scene.mvm; files = []
        for v in s.views:
            for fp in (v.normal_map, v.depth_map):
                fp = bpy.path.abspath(fp) if fp else ""
                if fp and os.path.exists(fp) and fp not in files:
                    files.append(fp)
        if not files:
            self.report({"ERROR"}, "no map files in the views"); return {"CANCELLED"}
        if not check_deps():
            self.report({"ERROR"}, "install the dependencies first"); return {"CANCELLED"}
        def done(ctx, ok):
            if ok:
                for im in bpy.data.images:
                    if im.filepath and bpy.path.abspath(im.filepath) in files:
                        im.reload()
                st = ctx.scene.mvm; st.relief_inverted = not st.relief_inverted
                st.status = f"relief inverted in {len(files)} files" + (" (now inverted)" if st.relief_inverted else " (back to original)")
                mark_dirty()
        start_job([[sys.executable, os.path.join(CORE, "invert_maps.py")] + files], on_done=done); return {"FINISHED"}


class MVM_OT_reduce(bpy.types.Operator):
    bl_idname = "mvm.reduce"; bl_label = "Reduce to limit"
    bl_description = "Decimate modifier on the result to the polygon limit (bake the detail from the full copy afterwards)"
    def execute(self, context):
        s = context.scene.mvm; obj = bpy.data.objects.get(RESULT) or bpy.data.objects.get(PREVIEW)
        if obj is None:
            self.report({"ERROR"}, "build the mesh first"); return {"CANCELLED"}
        if s.keep_copy and bpy.data.objects.get(obj.name + "_full") is None:
            c = obj.copy(); c.data = obj.data.copy(); c.name = obj.name + "_full"; context.scene.collection.objects.link(c); c.hide_viewport = True
        tris = len(obj.data.polygons); mod = obj.modifiers.get("MVM_Limit") or obj.modifiers.new("MVM_Limit", "DECIMATE")
        mod.decimate_type = "COLLAPSE"; mod.ratio = min(1.0, s.poly_limit / max(1, tris)); mod.use_collapse_triangulate = True
        self.report({"INFO"}, f"{tris} -> ~{int(tris * mod.ratio)} triangles"); return {"FINISHED"}


# =============================================================================================== clear cache
HF_MODELS = ("tencent/Hunyuan3D-2", "tencent/Hunyuan3D-2mini", "tencent/Hunyuan3D-2mv", "depth-anything/Depth-Anything-V2-Large-hf", "prs-eth/marigold-normals-v1-1")
WORK_DIRS = ("aligned", "aligned_preview", "maps")
WORK_FILES = ("*.npz", "*.json", "*.obj", "normal_*.png", "*.tmp")


def hf_cache():
    e = os.environ
    if e.get("HF_HUB_CACHE"):
        return e["HF_HUB_CACHE"]
    home = e.get("HF_HOME") or os.path.join(e.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "huggingface")
    return os.path.join(home, "hub")


def _size(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    t = 0
    for r, _, fs in os.walk(p):
        for f in fs:
            try:
                t += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return t


def _work_dirs(s):
    """the work folder of this scene + the default one (unsaved .blend) - only files the plugin writes are touched"""
    out = []
    for d in (work_dir_path(s), default_work_dir()):
        d = os.path.normpath(d)
        if os.path.isdir(d) and d not in out:
            out.append(d)
    return out


def cache_items(s, envs):
    """(label, path) of everything Clear cache removes"""
    items = []
    hub = hf_cache()
    for rid in HF_MODELS:
        name = "models--" + rid.replace("/", "--")
        for p in (os.path.join(hub, name), os.path.join(hub, ".locks", name)):
            if os.path.exists(p):
                items.append(("model " + rid if ".locks" not in p else "lock " + rid, p))
    for d in _work_dirs(s):
        for sub in WORK_DIRS:
            if os.path.isdir(os.path.join(d, sub)):
                items.append(("temp " + sub, os.path.join(d, sub)))
        for f in os.listdir(d):
            fp = os.path.join(d, f)
            if os.path.isfile(fp) and any(fnmatch.fnmatch(f, pat) for pat in WORK_FILES):
                items.append(("temp " + f, fp))
    if envs:
        for label, p in (("environment PyTorch + Hunyuan3D + map bakers", neural_dir()), ("plugin dependencies", deps_dir())):
            if os.path.isdir(p):
                items.append((label, p))
    return items


def _rm(p):
    def onerr(func, path, exc):                                      # read-only files on Windows
        try:
            os.chmod(path, stat.S_IWRITE); func(path)
        except Exception:
            pass
    if os.path.isdir(p) and not os.path.islink(p):
        shutil.rmtree(p, onerror=onerr)
    elif os.path.exists(p):
        try:
            os.remove(p)
        except OSError:
            os.chmod(p, stat.S_IWRITE); os.remove(p)


def _gb(b):
    return f"{b / 1e9:.2f} GB" if b >= 1e8 else f"{b / 1e6:.1f} MB"


class MVM_OT_clear_cache(bpy.types.Operator):
    bl_idname = "mvm.clear_cache"; bl_label = "Clear cache"
    bl_description = ("Delete the downloaded models (Hunyuan3D, Depth Anything V2, Marigold) and the plugin's temporary files. "
                      "Meshes in the scene stay")
    envs: BoolProperty(name="Also installed environments", default=False,
                       description="PyTorch / Hunyuan3D / map bakers environment (~6 GB) and the plugin dependencies, plus pip's download cache. "
                                   "Install them again before the next use")
    _items = []

    def invoke(self, context, event):
        if JOB["proc"] is not None:
            self.report({"ERROR"}, "wait for the running job or cancel it"); return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        L = self.layout; items = cache_items(context.scene.mvm, self.envs)
        L.prop(self, "envs")
        if not items:
            L.label(text="Nothing to delete"); return
        tot = 0; c = L.column(align=True)
        for label, p in items:
            b = _size(p); tot += b
            if not label.startswith("lock "):
                c.label(text=f"{label}: {_gb(b)}")
        r = L.row(); r.alert = True; r.label(text=f"Will be deleted: {_gb(tot)}. This cannot be undone.", icon="ERROR")

    def execute(self, context):
        s = context.scene.mvm
        if JOB["proc"] is not None:
            self.report({"ERROR"}, "wait for the running job or cancel it"); return {"CANCELLED"}
        if LIVE["proc"] is not None:                                 # the worker keeps files open
            try:
                LIVE["proc"].kill(); LIVE["proc"].wait(timeout=5)
            except Exception:
                pass
            LIVE["proc"] = None; s.live_on = False; s.live_status = ""
        items = cache_items(s, self.envs); wds = _work_dirs(s)
        for im in list(bpy.data.images):                             # images loaded from the work folder (maps sheet, overlay)
            fp = os.path.normpath(bpy.path.abspath(im.filepath)) if im.filepath else ""
            if im.name == "MVM_maps_sheet" or any(fp.startswith(d + os.sep) for d in wds):
                bpy.data.images.remove(im)
        ref = bpy.data.objects.get(REF)
        if ref is not None:
            bpy.data.objects.remove(ref)
        if self.envs:                                                # pip's own cache (torch wheels ~2.5 GB)
            py = neural_py() if os.path.exists(neural_py()) else sys.executable
            try:
                subprocess.run([py, "-m", "pip", "cache", "purge"], capture_output=True, timeout=300, creationflags=NOWIN,
                               env=neural_env() if py != sys.executable else py_env())
            except Exception:
                pass
        freed, failed = 0, []
        for label, p in items:
            b = _size(p)
            try:
                _rm(p)
            except Exception:
                pass
            if os.path.exists(p):
                failed.append(label)
            else:
                freed += b
        for d in wds:                                                # the folder itself if nothing else is in it
            try:
                if not os.listdir(d):
                    os.rmdir(d)
            except OSError:
                pass
        for v in s.views:                                            # map paths that pointed into the deleted temp folders
            for attr in ("normal_map", "depth_map"):
                fp = getattr(v, attr)
                if fp and not os.path.exists(bpy.path.abspath(fp)):
                    setattr(v, attr, "")
        s.report = ""
        s.status = f"cache cleared: {_gb(freed)}" + (f"; could not delete: {', '.join(failed[:3])}" if failed else "")
        self.report({"WARNING"} if failed else {"INFO"}, s.status)
        redraw(); return {"FINISHED"}


# =============================================================================================== UI
class MVM_UL_views(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        r = layout.row(align=True); r.prop(item, "enabled", text=""); r.prop(item, "angle", text="", emboss=False)
        r.label(text=os.path.basename(item.normal_map) or ("measure: " + os.path.basename(item.measure) if item.measure else "-"))


class _P:
    bl_space_type = "VIEW_3D"; bl_region_type = "UI"; bl_category = "Multiview"


class MVM_PT_main(_P, bpy.types.Panel):
    bl_label = "Multiview Modeler"
    def draw(self, context):
        s = context.scene.mvm; L = self.layout
        r = L.row(align=True); r.operator("mvm.install_deps", icon="IMPORT"); r.operator("mvm.reset", icon="LOOP_BACK")
        L.prop(s, "height"); L.prop(s, "work_dir")
        L.prop(s, "start_mode", text="Method")
        r = L.row(align=True); r.prop(s, "trim_outside", text="Trim outside"); r.prop(s, "clean_floating", text="Remove floating")
        b = L.box()
        if s.live_on and LIVE["proc"] is not None:
            b.operator("mvm.live_stop", icon="PAUSE")
        else:
            r = b.row(align=True); r.operator("mvm.live_start", icon="PLAY"); r.prop(s, "delay")
        if s.live_on and LIVE["proc"] is not None:
            b.prop(s, "live_progress", text=(s.live_status or "waiting")[:60], slider=True)
        elif s.live_status:
            b.label(text=s.live_status[:90], icon="TIME")
        if s.running:
            L.prop(s, "progress", text=s.status[:60] or "running", slider=True); L.operator("mvm.cancel", icon="CANCEL")
        else:
            L.operator("mvm.build", icon="MESH_MONKEY")
            if s.status:
                L.label(text=s.status[:90])
        if s.report:
            L.label(text=s.report, icon="INFO")


class MVM_PT_views(_P, bpy.types.Panel):
    bl_label = "Views"; bl_parent_id = "MVM_PT_main"
    def draw(self, context):
        s = context.scene.mvm; L = self.layout
        r = L.row(); r.template_list("MVM_UL_views", "", s, "views", s, "view_index", rows=4)
        c = r.column(align=True); c.operator("mvm.view_add", text="", icon="ADD"); c.operator("mvm.view_remove", text="", icon="REMOVE")
        L.operator("mvm.standard_views", icon="SPHERE")
        if s.views and 0 <= s.view_index < len(s.views):
            v = s.views[s.view_index]
            L.prop(v, "angle"); L.prop(v, "normal_map"); L.prop(v, "depth_map"); L.prop(v, "measure")
            r = L.row(align=True); r.prop(v, "dx"); r.prop(v, "dy"); L.prop(v, "fmt")
        L.operator("mvm.make_normals", icon="NODE_MATERIAL")
        r = L.row(); r.operator("mvm.invert_relief", text="Invert relief (all views)", icon="ARROW_LEFTRIGHT")
        if s.relief_inverted:
            r.label(text="inverted", icon="CHECKMARK")


class MVM_PT_bake(_P, bpy.types.Panel):
    bl_label = "Bake maps from views"; bl_parent_id = "MVM_PT_main"; bl_options = {"DEFAULT_CLOSED"}
    def draw(self, context):
        s = context.scene.mvm; L = self.layout
        L.label(text="Colour 3D-render views -> depth + normal maps", icon="INFO")
        L.operator("mvm.install_bake", icon="IMPORT")
        L.prop(s, "bake_dir"); r = L.row(align=True); r.prop(s, "bake_normals", text=""); r.prop(s, "bake_quality", text="")
        L.prop(s, "bake_clay")
        r = L.row(); r.enabled = not s.bake_clay; r.prop(s, "bake_delight")
        L.operator("mvm.bake_maps", icon="RENDER_STILL")


def _method_header(panel, context, mode):
    on = context.scene.mvm.start_mode == mode
    panel.layout.label(text="", icon="RADIOBUT_ON" if on else "RADIOBUT_OFF")


def _sil_box(L, s):
    b = L.box(); b.prop(s, "use_sil", text="Fit to silhouettes")
    c = b.column(); c.enabled = s.use_sil
    c.prop(s, "sil_iters"); c.prop(s, "sil_spread"); c.prop(s, "sil_smooth")


def _fit_box(L, s):
    b = L.box(); b.label(text="Fit to normal maps")
    r = b.row(align=True); r.prop(s, "iters_preview", text="Preview"); r.prop(s, "iters_final", text="Final")
    b.prop(s, "relax", slider=True)
    r = b.row(); r.prop(s, "use_detail"); r.prop(s, "detail_above", slider=True)


class MVM_PT_m_volume(_P, bpy.types.Panel):
    bl_label = "1. Volume from views"; bl_parent_id = "MVM_PT_main"; bl_options = {"DEFAULT_CLOSED"}
    def draw_header(self, context):
        _method_header(self, context, "VOLUME")
    def draw(self, context):
        s = context.scene.mvm; L = self.layout
        L.prop_enum(s, "start_mode", "VOLUME", text="Use this method")
        L.label(text="Only normal maps. Tested: 0.78 % of height", icon="INFO")
        r = L.row(align=True); r.prop(s, "res_preview", text="Preview"); r.prop(s, "res_final", text="Final")
        L.prop(s, "max_carve", slider=True)
        r = L.row(); r.prop(s, "use_carve"); r.prop(s, "carve_above", slider=True)
        L.prop(s, "thin", slider=True); L.prop(s, "smooth", slider=True)
        _sil_box(L, s); _fit_box(L, s)
        L.operator("mvm.reset_method", text="Reset method settings", icon="LOOP_BACK").mode = "VOLUME"


class MVM_PT_m_hunyuan(_P, bpy.types.Panel):
    bl_label = "2. Hunyuan3D start"; bl_parent_id = "MVM_PT_main"; bl_options = {"DEFAULT_CLOSED"}
    def draw_header(self, context):
        _method_header(self, context, "HUNYUAN")
    def draw(self, context):
        s = context.scene.mvm; L = self.layout
        L.prop_enum(s, "start_mode", "HUNYUAN", text="Use this method")
        L.label(text="Colour image + normal maps. Clean shape, closed parts are completed.", icon="INFO")
        L.label(text="Licence: Tencent Hunyuan, see its terms before commercial use", icon="ERROR")
        L.operator("mvm.install_neural", text="Install Hunyuan3D", icon="IMPORT")
        L.prop(s, "hy_variant")
        L.prop(s, "color_image")
        if s.hy_variant == "mv":
            L.prop(s, "color_left"); L.prop(s, "color_back")
        L.prop(s, "fg_ratio", slider=True); L.prop(s, "hy_octree"); L.prop(s, "hy_steps")
        L.operator("mvm.neural_start", icon="SHADERFX")
        _sil_box(L, s); _fit_box(L, s)
        L.operator("mvm.reset_method", text="Reset method settings", icon="LOOP_BACK").mode = "HUNYUAN"


class MVM_PT_m_depth(_P, bpy.types.Panel):
    bl_label = "3. Depth + normals (C2)"; bl_parent_id = "MVM_PT_main"; bl_options = {"DEFAULT_CLOSED"}
    def draw_header(self, context):
        _method_header(self, context, "DEPTH")
    def draw(self, context):
        s = context.scene.mvm; L = self.layout
        L.prop_enum(s, "start_mode", "DEPTH", text="Use this method")
        L.label(text="Depth map for every view (Views panel). Tested: 0.21 %", icon="INFO")
        r = L.row(align=True); r.prop(s, "dn_res_preview", text="Preview"); r.prop(s, "dn_res_final", text="Final")
        L.prop(s, "dn_max_carve", slider=True)
        L.prop(s, "dn_smooth"); r = L.row(align=True); r.prop(s, "dn_scale"); r.prop(s, "dn_auto")
        r = L.row(align=True); r.prop(s, "depth_range"); r.prop(s, "depth_invert")
        L.prop(s, "dn_fit")
        if s.dn_fit:
            _fit_box(L, s)
        L.operator("mvm.reset_method", text="Reset method settings", icon="LOOP_BACK").mode = "DEPTH"


class MVM_PT_display(_P, bpy.types.Panel):
    bl_label = "Display"; bl_parent_id = "MVM_PT_main"; bl_options = {"DEFAULT_CLOSED"}
    def draw(self, context):
        s = context.scene.mvm; L = self.layout
        L.prop(s, "diag"); L.prop(s, "show_underlay")
        r = L.row(); r.prop(s, "overlay"); r.prop(s, "overlay_opacity", slider=True)


class MVM_PT_game(_P, bpy.types.Panel):
    bl_label = "Game mesh"; bl_parent_id = "MVM_PT_main"; bl_options = {"DEFAULT_CLOSED"}
    def draw(self, context):
        s = context.scene.mvm; L = self.layout
        L.prop(s, "poly_limit"); L.prop(s, "keep_copy"); L.operator("mvm.reduce", icon="MOD_DECIM")


class MVM_PT_clear(_P, bpy.types.Panel):
    bl_label = "Clear cache"; bl_parent_id = "MVM_PT_main"; bl_options = {"HIDE_HEADER"}
    def draw(self, context):
        L = self.layout; L.separator(); r = L.row(); r.alert = True; r.scale_y = 1.2
        r.operator("mvm.clear_cache", text="Clear cache", icon="TRASH")


classes = (MVM_View, MVM_Settings, MVM_OT_run_job, MVM_OT_cancel, MVM_OT_live_start, MVM_OT_live_stop, MVM_OT_standard_views,
           MVM_OT_view_add, MVM_OT_view_remove, MVM_OT_reset, MVM_OT_install_deps, MVM_OT_make_normals, MVM_OT_build,
           MVM_OT_reduce, MVM_OT_reset_method, MVM_OT_install_neural, MVM_OT_neural_start, MVM_OT_install_bake, MVM_OT_bake_maps, MVM_OT_invert_relief, MVM_UL_views,
           MVM_PT_main, MVM_PT_views, MVM_PT_bake,
           MVM_PT_m_volume, MVM_PT_m_hunyuan, MVM_PT_m_depth, MVM_PT_display, MVM_PT_game, MVM_OT_clear_cache, MVM_PT_clear)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.mvm = PointerProperty(type=MVM_Settings)


def unregister():
    if LIVE["proc"] is not None:
        try:
            LIVE["proc"].kill()
        except Exception:
            pass
    del bpy.types.Scene.mvm
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
