"""Mesh helpers. One job: slice an STL into convex pieces, so a part with holes or teeth collides properly."""
import hashlib
import os
import shutil

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mesh_cache")


def convex_parts(stl: str, threshold: float = 0.015) -> list:
    """Paths of the convex pieces (CoACD) of `stl`, in the STL's own units. The first call takes ~10 s per
    mesh; the pieces are cached in .mesh_cache/, keyed by the file's CONTENT, so editing an STL re-slices it
    and nothing stale is ever reused. A lower threshold = more, tighter pieces = slower contacts."""
    with open(stl, "rb") as fh:
        key = hashlib.sha1(fh.read()).hexdigest()[:10]
    folder = os.path.join(CACHE, f"{os.path.splitext(os.path.basename(stl))[0]}_{key}_{threshold}")
    if not os.path.isdir(folder):
        import coacd
        import trimesh
        coacd.set_log_level("error")
        mesh = trimesh.load(stl)
        tmp = folder + ".part"          # build aside, rename at the end: a killed run never leaves a half-cache
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        for i, (v, f) in enumerate(coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=threshold, seed=0)):
            trimesh.Trimesh(v, f).export(os.path.join(tmp, f"{i}.stl"))
        os.rename(tmp, folder)
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder), key=lambda f: int(f[:-4]))]
