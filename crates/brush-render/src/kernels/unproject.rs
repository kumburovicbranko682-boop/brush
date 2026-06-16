//! Per-pixel undistorted camera-ray grid.
//!
//! Writes the true z=1 camera ray for each pixel center to `[H*W*3]` by
//! inverting the lens model through the shared in-kernel `unproject_ray` (the
//! same undistort the GOF rasterizer uses on-the-fly). The depth-normal
//! consistency loss multiplies this grid by the GOF median depth to recover
//! camera-space surface points, so the finite-difference normal uses true
//! undistorted rays for any lens (incl. fisheye), not the pinhole approximation.
//!
//! Stays in brush-render (not brush-mesh) because the depth-normal training
//! loss uses this ray grid via `burn_glue::unproject_ray_grid`; brush-mesh
//! depends on brush-render, so moving it there would invert that edge.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

use super::camera_model::{CameraModel, unproject_ray};

pub const WG_SIZE: u32 = 256;

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn unproject_ray_grid_kernel(
    out: &mut Tensor<f32>,
    img_w: u32,
    img_h: u32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    #[comptime] camera_model: CameraModel,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx >= img_w * img_h {
        terminate!();
    }
    let px = idx % img_w;
    let py = idx / img_w;
    let d_x = (px as f32 + 0.5f32 - cx) / fx;
    let d_y = (py as f32 + 0.5f32 - cy) / fy;
    let ray = unproject_ray(d_x, d_y, camera_model);
    let base = (idx * 3u32) as usize;
    out[base] = ray.x();
    out[base + 1] = ray.y();
    out[base + 2] = ray.z();
}
