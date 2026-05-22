#!/usr/bin/env python3
"""
Visualize WHAM output using pytorch3d — renders the SMPL mesh per frame and saves a video.

Usage:
    python scripts/visualize_wham.py \
        --input output/wham/4/4/wham_output.pkl \
        --smpl SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl \
        --output output/wham/4/render.mp4 \
        --track -1
"""

import argparse
import pickle
import joblib
import numpy as np
import torch
import cv2

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesVertex,
)


def load_smpl_faces(smpl_path):
    """Load SMPL faces without requiring chumpy."""
    import sys

    # Replace every chumpy class with a plain stub during unpickling
    class ChumStub:
        def __init__(self, *args, **kwargs): pass
        def __setstate__(self, state): pass

    class SmplUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith('chumpy'):
                return ChumStub
            return super().find_class(module, name)

    with open(smpl_path, 'rb') as f:
        smpl = SmplUnpickler(f, encoding='latin1').load()

    faces = np.array(smpl['f'], dtype=np.int64)
    return torch.tensor(faces, dtype=torch.long)


def pick_track(data):
    """Pick the track with the most frames (= main person = Ronaldo)."""
    return max(data.values(), key=lambda t: len(t['frame_ids']))


def build_renderer(image_size, device):
    from pytorch3d.renderer import FoVPerspectiveCameras, AmbientLights
    cameras = FoVPerspectiveCameras(
        znear=0.1,
        zfar=100.0,
        fov=60.0,
        device=device,
    )
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
    )
    lights = PointLights(
        ambient_color=((0.5, 0.5, 0.5),),
        diffuse_color=((0.7, 0.7, 0.7),),
        specular_color=((0.2, 0.2, 0.2),),
        location=[[1.0, 2.0, -2.0]],
        device=device,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )
    return renderer


def render_track(track, faces, output_path, image_size=512, device_str="auto"):
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    verts_all = track['verts']           # (T, 6890, 3)
    T = verts_all.shape[0]
    fps = 30

    renderer = build_renderer(image_size, device)
    faces_t = faces.to(device)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (image_size, image_size))

    print(f"Rendering {T} frames...")
    for i in range(T):
        verts = torch.tensor(verts_all[i], dtype=torch.float32).unsqueeze(0).to(device)

        # Fix coordinate system: SMPL has Y-up, pytorch3d renderer has Y-down
        verts[..., 1] *= -1

        # Center and scale
        center = verts.mean(dim=1, keepdim=True)
        verts = verts - center
        verts[..., 2] += 3.0   # push back so camera can see it

        # Vertex colors (light blue)
        colors = torch.ones_like(verts) * torch.tensor([0.6, 0.8, 1.0], device=device)
        textures = TexturesVertex(verts_features=colors)

        mesh = Meshes(
            verts=verts,
            faces=faces_t.unsqueeze(0),
            textures=textures,
        )

        image = renderer(mesh)                          # (1, H, W, 4)
        image_np = (image[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        writer.write(image_bgr)

        if i % 30 == 0:
            print(f"  Frame {i}/{T}")

    writer.release()
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Path to wham_output.pkl")
    parser.add_argument("--smpl",   required=True, help="Path to SMPL .pkl model file")
    parser.add_argument("--output", required=True, help="Output video path (.mp4)")
    parser.add_argument("--track",  type=int, default=-1,
                        help="Track ID to render (-1 = pick longest track = main person)")
    parser.add_argument("--size",   type=int, default=512, help="Output image size")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu (default: auto)")
    args = parser.parse_args()

    print("Loading WHAM output...")
    data = joblib.load(args.input)
    print(f"Tracks found: {list(data.keys())}")

    if args.track == -1:
        track = pick_track(data)
        print(f"Auto-selected track with {len(track['frame_ids'])} frames")
    else:
        track = data[args.track]

    print("Loading SMPL faces...")
    faces = load_smpl_faces(args.smpl)

    render_track(track, faces, args.output, image_size=args.size, device_str=args.device)


if __name__ == "__main__":
    main()
