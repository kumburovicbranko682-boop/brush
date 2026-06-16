//! Splat initialization from per-view `LiDAR` depth.
//!
//! Back-projects each view's metric `LiDAR` depth into a world point cloud
//! (confidence-filtered), samples per-point color, then collapses the heavy
//! cross-view overlap by keeping one isotropic splat per occupied metric cell.
//! Cells are metric, so density is fixed regardless of object size.

use brush_render::sh::rgb_to_sh;
use brush_serde::SplatData;
use glam::{UVec2, Vec3};
use hashbrown::HashMap;

use crate::scene::SceneView;

/// Build init `SplatData` from the views' `LiDAR` depth. `voxel_size` is the
/// metric cell size in metres (`<= 0` falls back to 2cm). `min_conf` is the
/// `ARKit` confidence floor (0/1/2). `max_dist` drops returns farther than that
/// (metres). Returns `None` when no view has usable depth.
pub async fn lidar_init_splats(
    views: &[SceneView],
    voxel_size: f32,
    min_conf: u8,
    max_dist: f32,
) -> anyhow::Result<Option<SplatData>> {
    // Pass 1: back-project every confident depth pixel into a world point and a
    // sampled color. Runs on a small pool of actor threads; each owns a chunk
    // of views.
    async fn project_chunk(
        views: Vec<SceneView>,
        min_conf: u8,
        max_dist: f32,
    ) -> anyhow::Result<Vec<(Vec3, Vec3)>> {
        let mut pts: Vec<(Vec3, Vec3)> = Vec::new();
        for view in &views {
            let Some(depth_loader) = &view.depth else {
                continue;
            };
            let depth = depth_loader.load().await?;
            let (w, h) = (depth.width, depth.height);
            if w == 0 || h == 0 {
                continue;
            }

            let img = view.image.load().await?.to_rgb8();
            let (iw, ih) = (img.width() as f32, img.height() as f32);
            let img_raw = img.as_raw();

            // Intrinsics at the depth map's native resolution (LiDAR is aligned
            // to the RGB camera's fov). Pinhole back-projection ignores lens
            // distortion, which is fine for an init.
            let pin = view
                .camera
                .build_pinhole_params(UVec2::new(w as u32, h as u32));
            let l2w = view.camera.local_to_world();

            for v in 0..h {
                for u in 0..w {
                    let j = v * w + u;
                    if depth.conf[j] < min_conf {
                        continue;
                    }
                    let z = depth.depth[j];
                    if !(z.is_finite() && z > 0.0) || z > max_dist {
                        continue;
                    }
                    let xc = (u as f32 + 0.5 - pin.cx) / pin.fx * z;
                    let yc = (v as f32 + 0.5 - pin.cy) / pin.fy * z;
                    let p = l2w.transform_point3(Vec3::new(xc, yc, z));
                    if !p.is_finite() {
                        continue;
                    }

                    // Color from the matching image pixel (depth and RGB share
                    // the camera fov, different resolutions).
                    let ix = (((u as f32 + 0.5) / w as f32) * iw).clamp(0.0, iw - 1.0) as usize;
                    let iy = (((v as f32 + 0.5) / h as f32) * ih).clamp(0.0, ih - 1.0) as usize;
                    let pb = (iy * iw as usize + ix) * 3;
                    let col = Vec3::new(
                        img_raw[pb] as f32,
                        img_raw[pb + 1] as f32,
                        img_raw[pb + 2] as f32,
                    ) / 255.0;

                    pts.push((p, col));
                }
            }
        }
        Ok(pts)
    }

    let workers = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(views.len().max(1));
    let chunk_size = views.len().div_ceil(workers).max(1);
    // Keep the actors alive until their handles resolve (dropping an actor
    // tears down its worker thread).
    let mut actors = Vec::new();
    let mut handles = Vec::new();
    for chunk in views.chunks(chunk_size) {
        let chunk = chunk.to_vec();
        let actor = brush_async::Actor::new("lidar-init");
        handles.push(actor.run(move || project_chunk(chunk, min_conf, max_dist)));
        actors.push(actor);
    }

    let mut pts: Vec<(Vec3, Vec3)> = Vec::new();
    for handle in handles {
        pts.extend(handle.await?);
    }
    drop(actors);

    if pts.is_empty() {
        return Ok(None);
    }

    let voxel = if voxel_size > 0.0 { voxel_size } else { 0.02 };
    let inv = 1.0 / voxel;

    // Pass 2: keep one splat per metric cell. The first sample in a slot wins
    // outright; averaging would invent blurred data and costs a
    // read-modify-write per point.
    let mut acc: HashMap<(i64, i64, i64), (Vec3, Vec3)> = HashMap::new();
    for (p, col) in pts {
        let key = (
            (p.x * inv).floor() as i64,
            (p.y * inv).floor() as i64,
            (p.z * inv).floor() as i64,
        );
        acc.entry(key).or_insert((p, col));
    }

    // Isotropic splat: a sphere ~half the cell so seeds tile.
    let log_s = (voxel * 0.4).max(1e-4).ln();

    let n_out = acc.len();
    let mut means = Vec::with_capacity(n_out * 3);
    let mut log_scales = Vec::with_capacity(n_out * 3);
    let mut sh = Vec::with_capacity(n_out * 3);
    for (p, col) in acc.into_values() {
        let dc = rgb_to_sh(col);
        means.extend_from_slice(&[p.x, p.y, p.z]);
        log_scales.extend_from_slice(&[log_s, log_s, log_s]);
        sh.extend_from_slice(&[dc.x, dc.y, dc.z]);
    }

    Ok(Some(SplatData {
        means,
        rotations: None,
        log_scales: Some(log_scales),
        sh_coeffs: Some(sh),
        raw_opacities: None,
    }))
}
