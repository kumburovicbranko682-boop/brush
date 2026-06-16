//! GOF geometry helpers.
//!
//! The renderer emits, per pixel, an alpha-blended view-space surface normal
//! `N` (the GOF quadric gradient `-normalize(Σ_c⁻¹·ray)`) and the GOF median
//! depth: the camera-space z depth of the splat at the 0.5 transmittance
//! crossing, raw (not alpha-normalized). These helpers take the `[H, W, 7]`
//! geometry slice `(Nx, Ny, Nz, depth, distortion, S1, S2)` sliced from the
//! full render as `out_img[.., .., 4..11]`, with depth at local index 3 and
//! distortion at 4. They are pure burn tensor ops so they compose with autodiff
//! (depth/normal losses) and also back the viewer's depth/normal channels. The
//! viewer-side colormap + RGBA8 packing lives in the app crate
//! (`ui::geo_view`).

use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use burn::tensor::{Tensor, s};

/// Geometry render channels: `rgba` + `Nx,Ny,Nz,depth` + normalized GOF
/// distortion + its two mapped-depth moments (backward-only).
pub const GEO_CHANNELS: usize = 11;

/// Minimum coverage alpha for a pixel's GOF median depth to be trusted in the
/// depth losses; below this the single-sample depth is unreliable.
const COVERAGE_FLOOR: f32 = 0.8;

// All helpers below take the 7-channel *geometry* tensor `[H, W, 7]` =
// `(Nx, Ny, Nz, depth, distortion, S1, S2)` (the blended GOF normal + GOF
// median depth + distortion moments), not the full rgba+geo render. Callers
// slice it out (`out_img[.., .., 4..11]`) and pass coverage alpha separately.

/// Per-pixel GOF median depth (camera-space z), `[H, W, 1]`: the depth of the
/// splat at the 0.5 transmittance crossing. Raw, not alpha-normalized (it is a
/// single surface sample, not a blend); empty pixels read 0 and should be
/// masked by coverage downstream.
pub fn rendered_depth(geo: Tensor<3>) -> Tensor<3> {
    geo.slice(s![.., .., 3..4])
}

/// Rendered unit normal map `[H, W, 3]` from the geometry tensor.
pub fn rendered_normal(geo: Tensor<3>) -> Tensor<3> {
    normalize3(geo.slice(s![.., .., 0..3]))
}

fn normalize3(v: Tensor<3>) -> Tensor<3> {
    // Epsilon *inside* the sqrt so the gradient stays finite for near-zero
    // vectors (blended normal ≈ 0 at uncovered pixels). `sqrt'(0) = ∞`,
    // which would otherwise NaN the rotation gradients.
    let len = (v.clone().powf_scalar(2.0).sum_dim(2) + 1e-8).sqrt();
    v / len
}

/// Per-pixel cross product of two `[H, W, 3]` vector fields.
fn cross3(a: Tensor<3>, b: Tensor<3>) -> Tensor<3> {
    let ax = a.clone().slice(s![.., .., 0..1]);
    let ay = a.clone().slice(s![.., .., 1..2]);
    let az = a.slice(s![.., .., 2..3]);
    let bx = b.clone().slice(s![.., .., 0..1]);
    let by = b.clone().slice(s![.., .., 1..2]);
    let bz = b.slice(s![.., .., 2..3]);
    let cx = ay.clone() * bz.clone() - az.clone() * by.clone();
    let cy = az * bx.clone() - ax.clone() * bz;
    let cz = ax * by - ay * bx;
    Tensor::cat(vec![cx, cy, cz], 2)
}

/// Single-view depth-normal consistency: cosine error `1 − ⟨N_render, N_depth⟩`
/// against the central-difference normal of the unprojected depth map, in camera
/// space. Masked to interior pixels
/// whose whole finite-difference stencil is covered (`alpha >= 0.8`): the GOF
/// median depth is unreliable below that. Returns the scalar masked mean.
///
/// `ray_grid` is the precomputed `[H, W, 3]` true undistorted z=1 camera-ray
/// grid (see `burn_glue::unproject_ray_grid`): a splat-independent constant, so
/// the FD points unproject against the real lens rays, not a pinhole guess.
pub fn depth_normal_consistency(
    geo: Tensor<3>,
    alpha: Tensor<3>,
    ray_grid: Tensor<3>,
) -> Tensor<1> {
    let [h, w, _] = geo.dims();
    let n_render = rendered_normal(geo.clone());
    let depth = rendered_depth(geo);
    // Camera-space surface points: depth is camera-space z, so the point is
    // `z * rp` with rp the z=1 camera ray (not the unit ray).
    let points = ray_grid * depth;

    // Central differences over the interior (GOF/2DGS `depth_to_normal`).
    let right = points.clone().slice(s![1..h - 1, 2..w, ..]);
    let left = points.clone().slice(s![1..h - 1, 0..w - 2, ..]);
    let bottom = points.clone().slice(s![2..h, 1..w - 1, ..]);
    let top = points.slice(s![0..h - 2, 1..w - 1, ..]);
    let n_fd = normalize3(cross3(right - left, top - bottom));

    let n_r = n_render.slice(s![1..h - 1, 1..w - 1, ..]);
    // Cosine error 1 - <N_render, N_depth>. The cross-product stencil already
    // faces the camera (matching GOF/2DGS), so no orientation flip is applied.
    let err = -(n_r * n_fd).sum_dim(2) + 1.0;

    // Mask to interior pixels whose full stencil (center + 4 neighbours) is
    // covered, so uncovered pixels' unstable depth never enters the loss.
    let cov = alpha.detach().greater_equal_elem(COVERAGE_FLOOR).float();
    let mask = cov.clone().slice(s![1..h - 1, 1..w - 1, ..])
        * cov.clone().slice(s![1..h - 1, 2..w, ..])
        * cov.clone().slice(s![1..h - 1, 0..w - 2, ..])
        * cov.clone().slice(s![2..h, 1..w - 1, ..])
        * cov.slice(s![0..h - 2, 1..w - 1, ..]);

    (err * mask.clone()).sum() / mask.sum().clamp_min(1.0)
}

/// Depth-distortion loss (GOF): per pixel `Σᵢ>ⱼ wᵢwⱼ(mᵢ−mⱼ)² / ((1−T)²+ε)`
/// over NDC-mapped depths, accumulated and normalized in the rasterizer with
/// a full backward. Reads that channel off the `[H, W, 7]` geometry slice.
pub fn depth_distortion(geo: Tensor<3>) -> Tensor<3> {
    geo.slice(s![.., .., 4..5]).clamp_min(0.0)
}

/// Nearest-neighbour resize of a `[Hd, Wd, 1]` map to `[h, w, 1]`. Nearest
/// avoids fabricating depth across discontinuities when upsampling.
fn resize_nearest(x: Tensor<3>, h: usize, w: usize) -> Tensor<3> {
    let [hd, wd, _] = x.dims();
    let out = interpolate(
        x.reshape([1, 1, hd, wd]),
        [h, w],
        InterpolateOptions::new(InterpolateMode::Nearest),
    );
    out.reshape([h, w, 1])
}

/// Metric depth supervision: L1 between rendered depth (GOF median camera-space
/// `z`) and the per-view `LiDAR` z-depth, evaluated sparsely at the `LiDAR` grid
/// (the GT is a point sample; the render is downscaled to it, not the GT
/// upscaled). Supervised where `ARKit` confidence >= `min_conf` and the GT is
/// within `max_depth` (no return is encoded as +inf). Returns the scalar
/// weighted mean.
pub fn depth_l1_loss(
    geo: Tensor<3>,
    alpha: Tensor<3>,
    gt_z: Tensor<3>,
    gt_conf: Tensor<3>,
    min_conf: u8,
    max_depth: f32,
) -> Tensor<1> {
    let [hd, wd, _] = gt_z.dims();
    let rendered_z = rendered_depth(geo);
    let [h, w, _] = rendered_z.dims();
    let gt_z = gt_z.detach();

    // Trust only returns within `max_depth` (drops +inf no-returns) and at
    // confidence >= min_conf, the same floor the init uses.
    let near = gt_z.clone().lower_equal_elem(max_depth).float();
    let conf_ok = gt_conf.detach().greater_equal_elem(min_conf as f32).float();

    // Sanitize inf before the subtract: `render - inf = -inf`, and `-inf * 0`
    // (the weight there) is NaN. Clamp inf -> 1e9 (real depths are < 1e9, so
    // they pass through); the weight already masks them out.
    let gt_safe = gt_z.clamp_max(1e9);

    // "No depth" where coverage < 0.8: the GOF median depth is unreliable at low
    // alpha, so such pixels neither supervise nor count.
    let covered = alpha.detach().greater_equal_elem(COVERAGE_FLOOR).float();

    // Credit the best-matching covered pixel in each LiDAR cell: min LogL1
    // `log(1 + |render - gt|)` over them. A LiDAR ray can miss a thin feature
    // that only one render pixel covers, so requiring *some* surface in the cell
    // to match the GT (not all of them) is the robust semantics. LogL1 saturates
    // on large errors so a far/noisy return can't dominate. A cell with no
    // covered pixel is dropped. Needs an integer render:LiDAR ratio; else a
    // single nearest sample.
    let kh = h / hd;
    let kw = w / wd;
    let (err, cell_covered) = if kh * hd == h && kw * wd == w {
        // [H,W,1] -> [hd,kh,wd,kw] groups each cell's source pixels on dims 1,3.
        let blk = rendered_z.reshape([hd, kh, wd, kw]);
        let cov = covered.reshape([hd, kh, wd, kw]);
        let gt_b = gt_safe.reshape([hd, 1, wd, 1]);
        let logl1 = (blk - gt_b).abs().add_scalar(1.0).log();
        // Inflate uncovered pixels so they are never the per-cell min.
        let masked = logl1 + (cov.clone().neg() + 1.0) * 1e9;
        let err = masked.min_dim(3).min_dim(1).reshape([hd, wd, 1]);
        let cell_covered = cov.max_dim(3).max_dim(1).reshape([hd, wd, 1]);
        (err, cell_covered)
    } else {
        let down_z = resize_nearest(rendered_z, hd, wd);
        let down_cov = resize_nearest(covered, hd, wd);
        ((down_z - gt_safe).abs().add_scalar(1.0).log(), down_cov)
    };

    let weight = (conf_ok * near * cell_covered).detach();
    (err * weight.clone()).sum() / weight.sum().clamp_min(1.0)
}
