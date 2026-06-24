//! Splat initialization from camera rays at random depths.
//!
//! The image-colour analogue of the `LiDAR` init: for each view we cast rays
//! through random pixels, place a point at a random depth bracketing the scene
//! centre, and take its colour from the image. Scale is one constant from the
//! scene scale and point density. No real geometry, just a better-than-noise
//! starting cloud when there is no point cloud or depth to seed from.

use brush_render::{camera::Camera, gaussian_splats::inverse_sigmoid, sh::rgb_to_sh};
use brush_serde::SplatData;
use glam::{UVec2, Vec3};
use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::scene::SceneView;

/// Average nearest-neighbour distance between cameras (min 1m), used as a scene
/// scale fallback for degenerate / forward-facing rigs where the cameras don't
/// orbit a centre.
fn nn_camera_spacing(cameras: &[Camera]) -> f32 {
    if cameras.len() < 2 {
        return 1.0;
    }
    let mut total = 0.0f32;
    for (i, cam) in cameras.iter().enumerate() {
        let mut min_dist = f32::INFINITY;
        for (j, other) in cameras.iter().enumerate() {
            if i != j {
                min_dist = min_dist.min(cam.position.distance(other.position));
            }
        }
        total += min_dist;
    }
    (total / cameras.len() as f32 * 3.0).max(1.0)
}

/// Scene centre and scale from the camera rig. For object-centric rigs the
/// centroid is ~the scene centre and the mean camera-to-centroid distance is
/// ~the scene radius; fall back to camera spacing for near-coincident rigs.
fn scene_center_scale(cameras: &[Camera], scale_override: Option<f32>) -> (Vec3, f32) {
    let n = cameras.len().max(1) as f32;
    let center = cameras.iter().fold(Vec3::ZERO, |a, c| a + c.position) / n;
    let mean_radius = cameras
        .iter()
        .map(|c| c.position.distance(center))
        .sum::<f32>()
        / n;
    let scale = scale_override.unwrap_or_else(|| {
        if mean_radius > 1.0 {
            mean_radius
        } else {
            nn_camera_spacing(cameras)
        }
    });
    (center, scale)
}

/// Cast `count` rays for one view: random pixel, random depth bracketing the
/// scene centre, returning each ray's world position and source pixel for
/// colour lookup. Pure geometry; the caller maps the pixel to a colour.
fn sample_view_rays(
    cam: &Camera,
    img_size: UVec2,
    count: usize,
    center: Vec3,
    scene_scale: f32,
    min_standoff: f32,
    seed: u64,
) -> Vec<(Vec3, u32, u32)> {
    let (iw, ih) = (img_size.x, img_size.y);
    let pin = cam.build_pinhole_params(img_size);
    let l2w = cam.local_to_world();

    // Depth range brackets the scene centre as seen from this camera.
    let d_center = cam.position.distance(center).max(scene_scale);
    let near = (d_center - scene_scale).max(min_standoff);
    let far = (d_center + scene_scale).max(near + min_standoff);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let u = rng.random_range(0.0..iw as f32);
        let v = rng.random_range(0.0..ih as f32);
        let depth = rng.random_range(near..far);

        let xc = (u - pin.cx) / pin.fx * depth;
        let yc = (v - pin.cy) / pin.fy * depth;
        let p = l2w.transform_point3(Vec3::new(xc, yc, depth));
        if !p.is_finite() {
            continue;
        }

        let px = (u as u32).min(iw - 1);
        let py = (v as u32).min(ih - 1);
        out.push((p, px, py));
    }
    out
}

/// Build init `SplatData` by casting `init_count` camera rays through random
/// pixels at random depths. `scene_scale_override` fixes the depth range;
/// otherwise it's estimated from the cameras. Per-view image-load failures are
/// skipped; errors only if no view yields a usable image.
pub async fn random_init_splats(
    views: &[SceneView],
    init_count: usize,
    scene_scale_override: Option<f32>,
    seed: u64,
) -> anyhow::Result<SplatData> {
    let cameras: Vec<Camera> = views.iter().map(|v| v.camera).collect();
    let (center, scene_scale) = scene_center_scale(&cameras, scene_scale_override);
    let min_standoff = scene_scale * 0.2;

    // Per-view point quota: split `init_count` evenly, remainder to the first
    // views. Each view carries its own seed so chunks stay independent.
    let n_views = views.len().max(1);
    let base = init_count / n_views;
    let rem = init_count % n_views;
    let jobs: Vec<(SceneView, usize, u64)> = views
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let count = base + usize::from(i < rem);
            (v.clone(), count, seed.wrapping_add(i as u64 * 0x9E3779B9))
        })
        .collect();

    async fn sample_chunk(
        jobs: Vec<(SceneView, usize, u64)>,
        center: Vec3,
        scene_scale: f32,
        min_standoff: f32,
    ) -> anyhow::Result<Vec<(Vec3, Vec3)>> {
        let mut out = Vec::new();
        for (view, count, seed) in &jobs {
            if *count == 0 {
                continue;
            }
            let img = match view.image.load().await {
                Ok(img) => img.to_rgb8(),
                Err(_) => continue,
            };
            let (iw, ih) = (img.width(), img.height());
            if iw == 0 || ih == 0 {
                continue;
            }
            let raw = img.as_raw();

            let rays = sample_view_rays(
                &view.camera,
                UVec2::new(iw, ih),
                *count,
                center,
                scene_scale,
                min_standoff,
                *seed,
            );
            for (p, px, py) in rays {
                let pb = (py as usize * iw as usize + px as usize) * 3;
                let col = Vec3::new(raw[pb] as f32, raw[pb + 1] as f32, raw[pb + 2] as f32) / 255.0;
                out.push((p, col));
            }
        }
        Ok(out)
    }

    let workers = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(jobs.len().max(1));
    let chunk_size = jobs.len().div_ceil(workers).max(1);
    let mut actors = Vec::new();
    let mut handles = Vec::new();
    for chunk in jobs.chunks(chunk_size) {
        let chunk = chunk.to_vec();
        let actor = brush_async::Actor::new("random-init");
        handles.push(actor.run(move || sample_chunk(chunk, center, scene_scale, min_standoff)));
        actors.push(actor);
    }

    let mut pts: Vec<(Vec3, Vec3)> = Vec::new();
    for handle in handles {
        pts.extend(handle.await?);
    }
    drop(actors);

    if pts.is_empty() {
        anyhow::bail!("random init could not load any image to sample from");
    }

    // One isotropic scale for all seeds, from the scene scale and point
    // density, shrunk so seeds start small and grow into the scene.
    let log_s = (0.3 * scene_scale / (pts.len() as f32).cbrt())
        .max(1e-4)
        .ln();

    let n = pts.len();
    let mut means = Vec::with_capacity(n * 3);
    let mut log_scales = Vec::with_capacity(n * 3);
    let mut sh = Vec::with_capacity(n * 3);
    for (p, col) in pts {
        let dc = rgb_to_sh(col);
        means.extend_from_slice(&[p.x, p.y, p.z]);
        log_scales.extend_from_slice(&[log_s, log_s, log_s]);
        sh.extend_from_slice(&[dc.x, dc.y, dc.z]);
    }

    // Start translucent so seeds fade in rather than blocking the scene.
    let raw_opacities = vec![inverse_sigmoid(0.1); n];

    Ok(SplatData {
        means,
        rotations: None,
        log_scales: Some(log_scales),
        sh_coeffs: Some(sh),
        raw_opacities: Some(raw_opacities),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use brush_render::kernels::camera_model::CameraModel;
    use glam::Quat;

    fn orbit_cams(r: f32, n: usize) -> Vec<Camera> {
        (0..n)
            .map(|i| {
                let a = i as f32 / n as f32 * std::f32::consts::TAU;
                let pos = Vec3::new(r * a.cos(), r * a.sin(), 0.0);
                // brush cameras look along +Z in local space.
                let rot = Quat::from_rotation_arc(Vec3::Z, (-pos).normalize());
                Camera::new(
                    pos,
                    rot,
                    0.8,
                    0.8,
                    glam::vec2(0.5, 0.5),
                    CameraModel::Pinhole,
                )
            })
            .collect()
    }

    #[test]
    fn scene_scale_matches_orbit_radius() {
        let cams = orbit_cams(4.0, 12);
        let (center, scale) = scene_center_scale(&cams, None);
        assert!(center.length() < 0.5, "center {center:?} not near origin");
        assert!((scale - 4.0).abs() < 1.0, "scale {scale} should be ~4");
    }

    #[test]
    fn scale_override_wins() {
        let cams = orbit_cams(4.0, 12);
        let (_, scale) = scene_center_scale(&cams, Some(9.0));
        assert_eq!(scale, 9.0);
    }

    #[test]
    fn rays_bracket_scene_center() {
        let cams = orbit_cams(4.0, 12);
        let (center, scale) = scene_center_scale(&cams, None);
        let min_standoff = scale * 0.2;

        let mut all = Vec::new();
        for (i, cam) in cams.iter().enumerate() {
            all.extend(sample_view_rays(
                cam,
                UVec2::new(640, 480),
                400,
                center,
                scale,
                min_standoff,
                i as u64,
            ));
        }

        assert!(all.iter().all(|(p, ..)| p.is_finite()), "non-finite point");
        // Points should cluster around the scene centre, not pile up at the
        // cameras (mean distance well inside the orbit radius).
        let mean_d = all.iter().map(|(p, ..)| p.distance(center)).sum::<f32>() / all.len() as f32;
        assert!(
            mean_d < 0.75 * scale,
            "mean dist {mean_d} too large; not centered"
        );
    }
}
