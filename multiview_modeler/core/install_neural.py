"""install_neural.py - separate venv for the Hunyuan3D start mesh and the map bakers (PyTorch with CUDA). Run with Blender's Python.

  python install_neural.py DIR [--cuda cu126] [--maps]

DIR/venv           the virtual environment (python.exe there runs neural_start.py)
DIR/Hunyuan3D-2    code of Tencent-Hunyuan/Hunyuan3D-2 (main branch); only the shape part is used, nothing is compiled
rembg is replaced by a stub: the background is cut by neural_start.py itself.
"""
import argparse, io, os, shutil, subprocess, sys, urllib.request, zipfile

REPOS = {"hunyuan": ("https://github.com/Tencent-Hunyuan/Hunyuan3D-2/archive/refs/heads/main.zip", "Hunyuan3D-2")}
REMBG_STUB = '''"""stub: neural_start.py cuts the background itself"""
def new_session(*a, **k):
    return None


def remove(img, *a, **k):
    return img
'''


def say(*a):
    print(*a, flush=True)


def run(cmd, step, n, text):
    say(f"PROGRESS {step} {n} {text}")
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    tail = p.stdout.strip().splitlines()[-6:]
    for l in tail:
        say("  " + l)
    if p.returncode != 0:
        sys.exit(f"failed: {text}")


def fetch(url, dst):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    data = urllib.request.urlopen(url, timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        root = z.namelist()[0].split("/")[0]; z.extractall(os.path.dirname(dst))
    os.replace(os.path.join(os.path.dirname(dst), root), dst)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--hunyuan", action="store_true", help="ignored, always installed")
    ap.add_argument("--cuda", default="cu126")
    ap.add_argument("--maps", action="store_true", help="also the map bakers: Depth Anything V2 (transformers) + Marigold (diffusers)")
    a = ap.parse_args(); os.makedirs(a.dir, exist_ok=True); venv = os.path.join(a.dir, "venv")
    py = os.path.join(venv, "Scripts", "python.exe") if os.name == "nt" else os.path.join(venv, "bin", "python")
    n = 6 if a.maps else 5; k = 0
    if not os.path.exists(py):
        k += 1; run([sys.executable, "-m", "venv", venv], k, n, "creating the environment")
    else:
        k += 1; say(f"PROGRESS {k} {n} environment exists")
    k += 1; run([py, "-m", "pip", "install", "--upgrade", "pip", "setuptools==69.5.1", "wheel"], k, n, "pip")
    k += 1; run([py, "-m", "pip", "install", "torch", "torchvision", "--index-url", f"https://download.pytorch.org/whl/{a.cuda}"],
                k, n, "PyTorch with CUDA (~2.5 GB, several minutes)")
    site = subprocess.run([py, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], capture_output=True, text=True).stdout.strip()
    repo = os.path.join(a.dir, "Hunyuan3D-2")
    if os.path.exists(os.path.join(repo, "hy3dgen", "shapegen", "__init__.py")):     # already there (e.g. downloaded by hand) - no GitHub needed
        k += 1; say(f"PROGRESS {k} {n} Hunyuan3D code found, download skipped")
    else:
        k += 1; say(f"PROGRESS {k} {n} Hunyuan3D code (GitHub)")
        try:
            fetch(REPOS["hunyuan"][0], repo)
        except Exception as e:
            sys.exit("Hunyuan3D code download failed (" + str(e)[:120] + "). Download " + REPOS["hunyuan"][0] +
                     ", unpack, rename Hunyuan3D-2-main to Hunyuan3D-2 and put it into " + a.dir + ", then press Install again")
    k += 1; run([py, "-m", "pip", "install", "diffusers>=0.33,<0.36", "accelerate", "transformers>=4.45,<4.50", "huggingface-hub<1.0", "safetensors",
                 "omegaconf", "einops", "trimesh", "pymeshlab", "opencv-python-headless", "pyyaml", "tqdm", "Pillow", "scikit-image", "scipy"],
                k, n, "Hunyuan3D packages")
    if subprocess.run([py, "-c", "import rembg"], capture_output=True).returncode != 0:
        os.makedirs(os.path.join(site, "rembg"), exist_ok=True); open(os.path.join(site, "rembg", "__init__.py"), "w").write(REMBG_STUB)
    if a.maps:
        k += 1; say(f"PROGRESS {k} {n} map bakers use the same packages")
    chk = subprocess.run([py, "-c", "import sys; sys.path.insert(0, r'" + repo + "'); import torch; "
                          "from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline; "
                          "print('CUDA', torch.cuda.is_available(), torch.version.cuda, 'Hunyuan3D import OK')"], capture_output=True, text=True)
    say("DONE " + (chk.stdout.strip() or "Hunyuan3D import FAILED: " + chk.stderr.strip()[-300:]))


if __name__ == "__main__":
    main()
