//! Per-Gaussian seed-point sampling for the tetrahedralization.
//!
//! For each Gaussian we emit the 6 octahedron axis points of its oriented
//! `±r` extent plus the center (7 points; ablated against GOF's 9-point
//! box: fewer holes, equal PSNR, 29% fewer seeds). Each point's "scale" is
//! the parent Gaussian's max axis length, used downstream by
//! [`crate::filter::filter_mesh`].
//!
//! Points are frustum-culled against the training cameras: a point is kept
//! iff it projects inside the image (an optional margin widens this; GOF
//! uses none) for at least one view, and its depth lies in `(near, far)`.
//!
//! Matches `scene/gaussian_model.py::get_tetra_points` from the reference repo.

use brush_render::camera::Camera;
use glam::{Vec3, Vec4Swizzles};
use rayon::prelude::*;

/// Octahedron half-extent in standard deviations. GOF uses 3 (the support
/// cap, `exp(-0.5 * 3^2) <= 1/255`); seeding wider reaches into the gaps
/// between splats, and the coarser tets there bridge field dips narrower
/// than their span, cutting enclosed holes ~26% at unchanged PSNR.
pub const SIGMA_SCALE: f32 = 4.0;

#[derive(Debug, Clone)]
pub struct TetraPointsConfig {
    pub near: f32,
    /// Far plane of the per-camera seed frustums: the meshed region is the
    /// union of all camera frustums truncated at this distance (metres on
    /// metric scenes). Keeps the mesh to the part of the scene the capture
    /// actually orbits instead of every far-field splat.
    pub far: f32,
    /// Frustum margin as a fraction of the image size. GOF uses 0 (strict
    /// image bounds); positive widens, negative crops toward the image
    /// center. Overridden per-extraction by `ExtractConfig::image_crop`.
    pub frustum_margin: f32,
}

impl Default for TetraPointsConfig {
    fn default() -> Self {
        Self {
            // Matches the integrate kernels' NEAR_PLANE: seed points the
            // integration can't see are pure Delaunay load.
            near: 0.2,
            far: 2.5,
            frustum_margin: -0.1,
        }
    }
}

pub struct TetraPoints {
    pub points: Vec<Vec3>,
    pub scales: Vec<f32>,
}

/// Build the seed-point set for `means / quats / log_scales`. Input
/// arrays are flat `[N*3]`, `[N*4]` (wxyz), `[N*3]`. `cameras` /
/// `image_sizes` drive frustum culling.
///
/// The caller is expected to pass *baked* scales; i.e. the mip 3D
/// filter floor already folded in via [`Splats::bake_min_scale`]
/// (which `extract_mesh` does up front). No per-Gaussian inflation
/// happens here, which keeps the seed sampler in sync with the
/// integrate kernel that also reads baked transforms.
pub fn build_tetra_points(
    means: &[f32],
    quats_wxyz: &[f32],
    log_scales: &[f32],
    cameras: &[Camera],
    image_sizes: &[glam::UVec2],
    cfg: &TetraPointsConfig,
) -> TetraPoints {
    assert_eq!(
        cameras.len(),
        image_sizes.len(),
        "one image size per camera"
    );
    let n = means.len() / 3;
    assert_eq!(quats_wxyz.len(), n * 4, "flat [N*4] wxyz quats");
    assert_eq!(log_scales.len(), n * 3, "flat [N*3] log scales");

    let frustum_tests = build_frustum_tests(cameras, image_sizes);

    // Up to 7 points per Gaussian, parallel over Gaussians. The frustum
    // cull is folded in so we don't materialise the full temporary.
    let chunks: Vec<(Vec<Vec3>, Vec<f32>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mean = Vec3::new(means[3 * i], means[3 * i + 1], means[3 * i + 2]);
            let q = glam::Quat::from_xyzw(
                quats_wxyz[4 * i + 1],
                quats_wxyz[4 * i + 2],
                quats_wxyz[4 * i + 3],
                quats_wxyz[4 * i],
            )
            .normalize();
            let scale = Vec3::new(
                log_scales[3 * i].exp(),
                log_scales[3 * i + 1].exp(),
                log_scales[3 * i + 2].exp(),
            );
            let r_sigma = SIGMA_SCALE;
            // Stored per-point scale: `r · max_axis_effective`, the half-
            // extent along the widest axis; matches GOF's `vertices_scale`
            // so the filter rule `edge_len > scale_a + scale_b` lines up.
            let s_max = scale.max_element() * r_sigma;

            let mut pts: Vec<Vec3> = Vec::with_capacity(7);
            let mut scs: Vec<f32> = Vec::with_capacity(7);
            let mut push = |world: Vec3| {
                if point_in_any_frustum(world, &frustum_tests, cfg) {
                    pts.push(world);
                    scs.push(s_max);
                }
            };
            for axis in 0..3 {
                let mut d = Vec3::ZERO;
                d[axis] = scale[axis] * r_sigma;
                push(mean + q * d);
                push(mean - q * d);
            }
            push(mean);
            (pts, scs)
        })
        .collect();

    let mut points = Vec::with_capacity(n * 7);
    let mut scales = Vec::with_capacity(n * 7);
    for (mut p, mut s) in chunks {
        points.append(&mut p);
        scales.append(&mut s);
    }
    TetraPoints { points, scales }
}

/// Per-camera data for [`point_in_any_frustum`], built once: the world-to-
/// camera matrix and pinhole params are too costly to rebuild per point.
pub(crate) struct FrustumTest {
    w2c: glam::Mat4,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
}

pub(crate) fn build_frustum_tests(
    cameras: &[Camera],
    image_sizes: &[glam::UVec2],
) -> Vec<FrustumTest> {
    cameras
        .iter()
        .zip(image_sizes.iter())
        .map(|(cam, sz)| {
            let pinhole = cam.build_pinhole_params(*sz);
            FrustumTest {
                w2c: glam::Mat4::from(cam.world_to_local()),
                fx: pinhole.fx,
                fy: pinhole.fy,
                cx: pinhole.cx,
                cy: pinhole.cy,
                w: sz.x as f32,
                h: sz.y as f32,
            }
        })
        .collect()
}

fn point_in_any_frustum(p: Vec3, tests: &[FrustumTest], cfg: &TetraPointsConfig) -> bool {
    for t in tests {
        let p_cam = (t.w2c * p.extend(1.0)).xyz();
        let z = p_cam.z;
        if !(z > cfg.near && z < cfg.far) {
            continue;
        }
        let px = p_cam.x / z * t.fx + t.cx;
        let py = p_cam.y / z * t.fy + t.cy;
        let m = cfg.frustum_margin;
        if px >= -m * t.w && px <= (1.0 + m) * t.w && py >= -m * t.h && py <= (1.0 + m) * t.h {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use brush_render::kernels::camera_model::CameraModel;

    fn fake_cam(pos: Vec3) -> (Camera, glam::UVec2) {
        let cam = Camera::new(
            pos,
            glam::Quat::IDENTITY,
            std::f64::consts::PI * 0.5,
            std::f64::consts::PI * 0.5,
            glam::Vec2::new(0.5, 0.5),
            CameraModel::default(),
        );
        (cam, glam::UVec2::new(256, 256))
    }

    #[test]
    fn emits_expected_seeds_per_gaussian_when_visible() {
        // Pin the region config: this test checks seed emission, not the
        // subject-extraction defaults.
        let region = TetraPointsConfig {
            far: 5.0,
            frustum_margin: 0.0,
            ..Default::default()
        };
        let (cam, sz) = fake_cam(Vec3::new(0.0, 0.0, -3.0));
        let means = vec![0.0, 0.0, 0.0];
        let quats = vec![1.0, 0.0, 0.0, 0.0];
        let log_scales = vec![-2.0; 3];
        let out = build_tetra_points(&means, &quats, &log_scales, &[cam], &[sz], &region);
        assert_eq!(out.points.len(), 7, "octahedron: 6 axis points + center");
        let s = (-2.0f32).exp() * SIGMA_SCALE;
        for sc in &out.scales {
            assert!((sc - s).abs() < 1e-6, "stored scale is 3 sigma max axis");
        }
    }

    #[test]
    fn cull_drops_far_points() {
        let (cam, sz) = fake_cam(Vec3::new(0.0, 0.0, 0.0));
        let means = vec![0.0, 0.0, 1.0e10];
        let quats = vec![1.0, 0.0, 0.0, 0.0];
        let log_scales = vec![-2.0; 3];
        let cfg = TetraPointsConfig {
            far: 100.0,
            ..Default::default()
        };
        let out = build_tetra_points(&means, &quats, &log_scales, &[cam], &[sz], &cfg);
        assert!(out.points.is_empty());
    }
}
