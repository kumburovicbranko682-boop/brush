//! Viewer-only colormap + RGBA8 packing for the GOF geometry channels.
//! Takes the burn-computed depth / normal / alpha maps and writes a packed
//! `u32` image in the same layout the splat-backbuffer display shader reads.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

use crate::burn_glue::GeoColormapMode;

pub const WG_SIZE: u32 = 256;

/// Magma ramp (black→purple→orange→pale-yellow) over `t in [0, 1]`.
/// Polynomial fit of the matplotlib magma colormap.
#[cube]
fn magma(t: f32) -> (f32, f32, f32) {
    let t = clamp(t, 0.0f32, 1.0f32);
    let r = clamp(
        -0.002136f32
            + t * (0.2516f32 + t * (8.353f32 + t * (-27.66f32 + t * (28.32f32 + t * -8.40f32)))),
        0.0f32,
        1.0f32,
    );
    let g = clamp(
        0.000949f32
            + t * (0.6739f32 + t * (-3.276f32 + t * (10.31f32 + t * (-12.07f32 + t * 4.864f32)))),
        0.0f32,
        1.0f32,
    );
    let b = clamp(
        -0.005774f32
            + t * (2.494f32 + t * (-9.621f32 + t * (21.27f32 + t * (-23.05f32 + t * 9.110f32)))),
        0.0f32,
        1.0f32,
    );
    (r, g, b)
}

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn colormap_pack_kernel(
    depth: &Tensor<f32>,
    normal: &Tensor<f32>,
    alpha: &Tensor<f32>,
    out: &mut Tensor<u32>,
    dmin: f32,
    dinv_range: f32,
    num_pixels: u32,
    #[comptime] mode: GeoColormapMode,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx >= num_pixels {
        terminate!();
    }

    let mut r = 0.0f32;
    let mut g = 0.0f32;
    let mut b = 0.0f32;

    if comptime![mode == GeoColormapMode::Alpha] {
        let a = clamp(alpha[idx as usize], 0.0f32, 1.0f32);
        r = a;
        g = a;
        b = a;
    } else if alpha[idx as usize] > 0.5f32 {
        // Empty pixels (no surface) stay black.
        if comptime![mode == GeoColormapMode::Normal] {
            let nb = (idx * 3u32) as usize;
            r = clamp(normal[nb] * 0.5f32 + 0.5f32, 0.0f32, 1.0f32);
            g = clamp(normal[nb + 1] * 0.5f32 + 0.5f32, 0.0f32, 1.0f32);
            b = clamp(normal[nb + 2] * 0.5f32 + 0.5f32, 0.0f32, 1.0f32);
        } else {
            let t = clamp((depth[idx as usize] - dmin) * dinv_range, 0.0f32, 1.0f32);
            let (mr, mg, mb) = magma(t);
            r = mr;
            g = mg;
            b = mb;
        }
    }

    let ri = clamp(r * 255.0f32, 0.0f32, 255.0f32) as u32;
    let gi = clamp(g * 255.0f32, 0.0f32, 255.0f32) as u32;
    let bi = clamp(b * 255.0f32, 0.0f32, 255.0f32) as u32;
    out[idx as usize] = ri | (gi << 8u32) | (bi << 16u32) | (255u32 << 24u32);
}
