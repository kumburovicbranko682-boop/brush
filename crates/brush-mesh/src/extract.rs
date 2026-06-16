//! Top-level mesh-extraction driver. Glues tetra-point sampling, CPU
//! Delaunay, per-view GPU opacity integration, marching tets, and binary
//! search refinement.

use brush_cube::{MainBackendBase, calc_cube_count_1d};
use brush_render::SplatOps;
use brush_render::burn_glue::resolve_to_cube_float;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{RenderOptions, SplatRenderMode, Splats};
use burn::backend::ops::FloatTensorOps;
use burn::backend::tensor::FloatTensor;
use burn::tensor::s;
use burn_cubecl::cubecl::CubeDim;
use burn_wgpu::{WgpuDevice, WgpuRuntime};
use glam::{UVec2, Vec3};

use crate::Mesh;
use crate::filter::{filter_mesh_with_keep, filter_small_components};
use crate::marching_tet::marching_tets;
use crate::refine::{N_STEPS, RefineState, read_back_f32};
use crate::tetra_points::{TetraPointsConfig, build_tetra_points};

/// User-facing extraction config.
#[derive(Debug, Clone)]
pub struct ExtractConfig {
    pub tetra_points: TetraPointsConfig,
    /// Iso-value for the level set. Carves the surface where transmittance
    /// has dropped to this fraction. Lower = fatter solid: micro cracks
    /// need a deeper field dip to open (0.5 cut enclosed-hole count 31%
    /// vs 0.6 on the 1M-splat scene at equal PSNR). GOF uses 0.5.
    pub iso_value: f32,
    /// Taubin smoothing iterations (λ|μ pairs) applied before baking and
    /// simplification: a gentle band-limit so marching-tets noise near the
    /// vertex-spacing frequency doesn't alias into the decimated surface.
    /// Trades measured fidelity for visual smoothness (every iteration costs
    /// PSNR), so it defaults off; 0 disables.
    pub smooth_iters: u32,
    /// Multiplier on GOF's edge-scale filter: crossings on Delaunay edges
    /// longer than `edge_filter_scale x (sum of endpoint splat scales)` are
    /// dropped. 1.0 matches GOF; larger keeps more long-edge faces (fewer
    /// cracks in sparse regions, more speckle webbing); very large
    /// effectively disables the filter.
    pub edge_filter_scale: f32,
    /// Drop connected components with fewer faces than this (speckle blobs
    /// from isolated iso-crossings). 0/1 disables.
    pub min_component_faces: usize,
    /// Also drop components below this fraction of the largest one:
    /// disconnected debris scales with the mesh, an absolute floor alone
    /// does not.
    pub min_component_frac: f32,
    /// Simplify the mesh to roughly this many faces (quadric collapse)
    /// before any texturing: the geometry-rate dial. 0 disables.
    pub target_faces: u32,
    /// Bake a UV-atlased color texture with this atlas side length in
    /// texels: decouples color resolution from vertex density. 0 keeps
    /// vertex colors only.
    pub texture_size: u32,
    /// Cap on faces entering charting and baking, as a multiple of
    /// `target_faces`: dense extractions pre-simplify (no UVs, quadric
    /// placement free to average noise) so chart seams form on coarser
    /// geometry and the final UV-preserving pass can actually reach the
    /// target. 0 disables.
    pub pre_simplify_factor: usize,
    /// Fraction of input views the bake blends (most-used first);
    /// rendering is batched so this is a time dial, not a memory one.
    pub bake_view_frac: f32,
    /// Distance falloff exponent in the blend weight (cos^4 / z^p):
    /// 0 = none, 2 = favor close views. Measured monotonic on the test
    /// scene (masked PSNR 18.08 / 18.19 / 18.27 for p = 0 / 1 / 2);
    /// closer views carry more detail per texel and the self-consistent
    /// splat renders make favoring them ghost-free.
    pub texture_dist_power: f32,
    /// Resolution scale for the carve's cached renders. The alpha
    /// integrate only uses the per-tile gaussian lists (never the image),
    /// so reduced resolution shrinks render time, integrate time, and VRAM
    /// roughly quadratically; the bake re-renders its kept views at full
    /// resolution regardless. 0.5 measured PSNR-neutral (0.25 close).
    pub carve_scale: f32,
    /// Central image fraction kept across the whole pipeline: seed frustum
    /// cull, bake view-selection, and texel color blend all reject samples
    /// projecting outside the central `image_crop` box. Single source of
    /// truth (overrides `tetra_points.frustum_margin` in `extract_mesh`).
    /// Every meshed vertex is central in at least one view (the seed cull
    /// guarantees it), so cropping the bake only drops distorted-edge
    /// samples; it never creates untextured regions.
    pub image_crop: f32,
}

impl Default for ExtractConfig {
    fn default() -> Self {
        Self {
            tetra_points: TetraPointsConfig::default(),
            iso_value: 0.5,
            smooth_iters: 1,
            edge_filter_scale: 1.0,
            min_component_faces: 500,
            min_component_frac: 0.01,
            target_faces: 750_000,
            texture_size: 4096,
            pre_simplify_factor: 2,
            bake_view_frac: 0.8,
            texture_dist_power: 2.0,
            carve_scale: 0.5,
            image_crop: 0.8,
        }
    }
}

/// Extract a triangle mesh from `splats`. `views` carries the per-view
/// `(camera, image_size)` pairs used for both frustum-culling the seed
/// points and integrating the opacity along rays.
/// Extraction output: the mesh plus the optional baked color atlas.
pub struct ExtractOutput {
    pub mesh: Mesh,
    pub texture: Option<crate::texture::Texture>,
}

pub async fn extract_mesh(
    splats: Splats,
    views: &[(Camera, UVec2)],
    cfg: &ExtractConfig,
) -> ExtractOutput {
    // Bake the min_scale floor once so the seed sampler and integrate
    // kernels see the same effective splats as the renderer.
    let splats = splats.bake_min_scale();

    // Central crop frac -> per-side margin, the single source of truth for
    // the seed frustum cull, bake view-selection, and texel color blend.
    let crop_margin = ((1.0 - cfg.image_crop) * 0.5).clamp(0.0, 0.49);

    let n_splats = splats.num_splats() as usize;
    log::info!(
        "Extracting mesh from {n_splats} splats across {} views",
        views.len()
    );

    // Per-phase profile; each phase ends with a sync so queued GPU work
    // gets billed to the phase that launched it.
    let sync_client = resolve_to_cube_float(splats.transforms.val())
        .client
        .clone();
    let mut phases: Vec<(&'static str, std::time::Duration)> = Vec::new();
    let mut phase_start = std::time::Instant::now();

    // Pull splat tensors back to host for seed-point sampling.
    let means_t = splats.transforms.val().slice(s![.., 0..3]);
    let quats_t = splats.transforms.val().slice(s![.., 3..7]);
    let log_scales_t = splats.transforms.val().slice(s![.., 7..10]);
    let means: Vec<f32> = means_t
        .into_data_async()
        .await
        .expect("read means")
        .into_vec::<f32>()
        .expect("means f32");
    let quats: Vec<f32> = quats_t
        .into_data_async()
        .await
        .expect("read quats")
        .into_vec::<f32>()
        .expect("quats f32");
    let log_scales: Vec<f32> = log_scales_t
        .into_data_async()
        .await
        .expect("read scales")
        .into_vec::<f32>()
        .expect("scales f32");

    sync_client.sync().await.expect("sync");
    phases.push(("load_tensors", phase_start.elapsed()));
    phase_start = std::time::Instant::now();

    let cams: Vec<Camera> = views.iter().map(|(c, _)| *c).collect();
    let img_sizes: Vec<UVec2> = views.iter().map(|(_, s)| *s).collect();
    let splat_bbox = bbox(means.chunks_exact(3).map(|c| Vec3::new(c[0], c[1], c[2])));

    // The render cache feeds seed selection, the initial alpha eval, every
    // binary-search iter, and the colour eval.
    let render_mode = if splats.render_mip {
        SplatRenderMode::Mip
    } else {
        SplatRenderMode::Default
    };
    let carve_views: Vec<(Camera, UVec2)> = views
        .iter()
        .map(|(c, s)| {
            let scaled = (s.as_vec2() * cfg.carve_scale.clamp(0.05, 1.0)).round();
            (*c, scaled.as_uvec2().max(UVec2::ONE))
        })
        .collect();
    let view_cache = pre_render_views(&splats, &carve_views, render_mode).await;
    sync_client.sync().await.expect("sync");
    phases.push(("pre_render", phase_start.elapsed()));
    phase_start = std::time::Instant::now();

    // image_crop drives the seed frustum: override frustum_margin so the
    // crop has one source of truth (negative margin = inset).
    let mut tetra_cfg = cfg.tetra_points.clone();
    tetra_cfg.frustum_margin = -crop_margin;
    let pts = build_tetra_points(&means, &quats, &log_scales, &cams, &img_sizes, &tetra_cfg);
    log::info!("Seed points (frustum-culled): {}", pts.points.len());
    if pts.points.len() < 4 {
        log::warn!("Too few seed points to triangulate; returning empty mesh");
        return ExtractOutput {
            mesh: Mesh::default(),
            texture: None,
        };
    }
    phases.push(("select_and_build_seeds", phase_start.elapsed()));
    phase_start = std::time::Instant::now();

    // CPU Delaunay overlaps the GPU initial alpha eval over the seeds.
    let pts_for_delaunay = pts.points.clone();
    let delaunay_handle = tokio::task::spawn_blocking(move || {
        let span = tracing::trace_span!("delaunay_3d").entered();
        let tets = crate::delaunay::delaunay_3d(&pts_for_delaunay);
        drop(span);
        tets
    });
    let alpha_fut = evaluate_alpha(&splats, &pts.points, &view_cache);
    let (alpha, tets_result) = tokio::join!(alpha_fut, delaunay_handle);
    let tets = tets_result.expect("delaunay task panicked");
    log::info!("Delaunay tets: {}", tets.len());
    sync_client.sync().await.expect("sync");
    phases.push(("delaunay_and_initial_alpha", phase_start.elapsed()));
    phase_start = std::time::Instant::now();
    let sdf: Vec<f32> = alpha.iter().map(|a| a - cfg.iso_value).collect();

    let mt = marching_tets(&tets, &sdf);
    log::info!(
        "Crossings: {}, faces: {}",
        mt.crossings.len(),
        mt.faces.len()
    );
    if mt.crossings.is_empty() {
        log::warn!("No iso-surface crossings; nothing to refine. Returning empty mesh.");
        return ExtractOutput {
            mesh: Mesh::default(),
            texture: None,
        };
    }

    phases.push(("marching_tets", phase_start.elapsed()));
    phase_start = std::time::Instant::now();

    // Binary-search refinement with GPU-resident bracket state: N_STEPS
    // fixed steps over all crossings, no CPU traffic until the readback.
    let device = resolve_to_cube_float(splats.transforms.val()).device;
    let mut state = RefineState::new(&mt.crossings, &pts.points, &sdf, &device);
    for _ in 0..N_STEPS {
        let mid_pos_t = state.compute_midpoints_t();
        let min_alpha_t = integrate_alpha(&splats, &mid_pos_t, state.n_crossings(), &view_cache);
        state.update_bracket_t(min_alpha_t, cfg.iso_value);
    }
    sync_client.sync().await.expect("sync");
    phases.push(("binary_search", phase_start.elapsed()));
    phase_start = std::time::Instant::now();
    let refined = state.finish().await;

    // GOF's filter_mesh: keep a crossing iff its original Delaunay edge is
    // shorter than the two endpoint Gaussian scales combined (refined
    // positions sit between the endpoints, so test the original edge).
    let keep_crossing: Vec<bool> = mt
        .crossings
        .iter()
        .map(|c| {
            let pa = pts.points[c.a as usize];
            let pb = pts.points[c.b as usize];
            let d = (pa - pb).length();
            let scale_sum = pts.scales[c.a as usize] + pts.scales[c.b as usize];
            d <= scale_sum * cfg.edge_filter_scale
        })
        .collect();

    let mut mesh = Mesh {
        vertices: refined,
        faces: mt.faces,
    };

    mesh = filter_mesh_with_keep(&mesh, &keep_crossing);

    // Crop outliers from edges spanning the empty far field (huge billboard
    // splats defeat the edge-length filter since their scales are huge too).
    let margin = 0.1 * (splat_bbox.1 - splat_bbox.0).length();
    let crop_lo = splat_bbox.0 - Vec3::splat(margin);
    let crop_hi = splat_bbox.1 + Vec3::splat(margin);
    let inside: Vec<bool> = mesh
        .vertices
        .iter()
        .map(|v| {
            v.x >= crop_lo.x
                && v.x <= crop_hi.x
                && v.y >= crop_lo.y
                && v.y <= crop_hi.y
                && v.z >= crop_lo.z
                && v.z <= crop_hi.z
        })
        .collect();
    let n_dropped = inside.iter().filter(|&&k| !k).count();
    if n_dropped > 0 {
        log::info!(
            "Cropping {n_dropped} verts outside splat-bbox+10% margin ({:.2} of {})",
            n_dropped as f32 / inside.len() as f32 * 100.0,
            inside.len()
        );
        mesh = filter_mesh_with_keep(&mesh, &inside);
    }

    mesh = filter_small_components(&mesh, cfg.min_component_faces, cfg.min_component_frac);

    if cfg.smooth_iters > 0 {
        let t_smooth = std::time::Instant::now();
        crate::smooth::taubin_smooth(&mut mesh, cfg.smooth_iters);
        log::info!(
            "Taubin smoothing: {} iters in {:.2}s",
            cfg.smooth_iters,
            t_smooth.elapsed().as_secs_f64()
        );
    }

    let mesh_bbox = bbox(mesh.vertices.iter().copied());
    log::info!(
        "Final mesh: {} verts, {} faces, bbox extent ({:.2}x{:.2}x{:.2})",
        mesh.vertices.len(),
        mesh.faces.len(),
        mesh_bbox.1.x - mesh_bbox.0.x,
        mesh_bbox.1.y - mesh_bbox.0.y,
        mesh_bbox.1.z - mesh_bbox.0.z,
    );

    phases.push(("filter_and_crop", phase_start.elapsed()));
    phase_start = std::time::Instant::now();

    // With a texture the bake happens on the full-resolution mesh and the
    // simplify runs afterwards, UV-preserving, underneath the finished
    // atlas: one geometry for charting/visibility/baking, so artifacts
    // can't come from a pre-bake decimation.
    let texturing = cfg.texture_size > 0 && !mesh.faces.is_empty();
    if cfg.target_faces > 0 && !texturing {
        let t_simp = std::time::Instant::now();
        mesh = crate::simplify::simplify_mesh(&mesh, cfg.target_faces as usize);
        log::info!("Simplify: {:.2}s", t_simp.elapsed().as_secs_f64());
        phases.push(("simplify", phase_start.elapsed()));
        phase_start = std::time::Instant::now();
    }

    // Dense extractions: cap the faces entering charting and baking so
    // seam-vertex counts (which lock the final UV-preserving pass) stay
    // proportional to the target rather than the raw extraction.
    if texturing
        && cfg.target_faces > 0
        && cfg.pre_simplify_factor > 0
        && mesh.faces.len() > cfg.target_faces as usize * cfg.pre_simplify_factor
    {
        let t_pre = std::time::Instant::now();
        mesh = crate::simplify::simplify_mesh(
            &mesh,
            cfg.target_faces as usize * cfg.pre_simplify_factor,
        );
        log::info!("Pre-simplify: {:.2}s", t_pre.elapsed().as_secs_f64());
        phases.push(("pre_simplify", t_pre.elapsed()));
        phase_start = std::time::Instant::now();
    }

    // Color atlas bake: pick views per face via mesh-depth visibility (to
    // bound which renders the blend reads), chart + pack UVs with density
    // following observed view resolution (the charter splits seam vertices,
    // so the mesh is re-indexed), then blend every kept view into each
    // texel sample on the GPU.
    let texture = if texturing {
        let size = cfg.texture_size;
        let mut rgba = vec![0u8; (size * size * 4) as usize];
        let atlas_uvs = {
            use crate::kernels::bake;
            use brush_cube::{create_tensor, create_tensor_from_slice};
            use burn::tensor::DType;

            let t_depth = std::time::Instant::now();
            let transforms_p = resolve_to_cube_float(splats.transforms.val());
            let device = transforms_p.device.clone();
            let client = transforms_p.client.clone();

            let centers: Vec<Vec3> = mesh
                .faces
                .iter()
                .map(|f| {
                    (mesh.vertices[f[0] as usize]
                        + mesh.vertices[f[1] as usize]
                        + mesh.vertices[f[2] as usize])
                        / 3.0
                })
                .collect();
            let fnormals: Vec<f32> = mesh
                .faces
                .iter()
                .flat_map(|f| {
                    let a = mesh.vertices[f[0] as usize];
                    let n = (mesh.vertices[f[1] as usize] - a)
                        .cross(mesh.vertices[f[2] as usize] - a)
                        .normalize_or_zero();
                    [n.x, n.y, n.z]
                })
                .collect();

            let n_views = views.len();
            let md = raster_mesh_depth(&device, &mesh, views);
            let n_faces = mesh.faces.len() as u32;

            let centers_flat: Vec<f32> = centers.iter().flat_map(|c| [c.x, c.y, c.z]).collect();
            let centers_t = create_tensor_from_slice(&centers_flat, &device, DType::F32);
            let normals_t = create_tensor_from_slice(&fnormals, &device, DType::F32);
            let select = |keep: Vec<u32>| {
                let keep_t = create_tensor_from_slice(&keep, &device, DType::U32);
                let best_t = create_tensor::<1>([n_faces as usize], &device, DType::F32);
                bake::select_best_view_kernel::launch::<WgpuRuntime>(
                    &client,
                    calc_cube_count_1d(n_faces, bake::BAKE_WG),
                    CubeDim::new_1d(bake::BAKE_WG),
                    centers_t.clone().into_tensor_arg(),
                    normals_t.clone().into_tensor_arg(),
                    md.view_data_t.clone().into_tensor_arg(),
                    md.offsets_t.clone().into_tensor_arg(),
                    md.depth_t.clone().into_tensor_arg(),
                    keep_t.into_tensor_arg(),
                    best_t.clone().into_tensor_arg(),
                    n_faces,
                    n_views as u32,
                    crop_margin,
                );
                best_t
            };
            let best1 = read_back_f32(select(vec![1u32; n_views])).await;
            log::info!(
                "depth raster + select: {:.2}s",
                t_depth.elapsed().as_secs_f64()
            );

            // Keep only the most-used views so the sampling and color
            // phases touch a bounded set of cached images (the full
            // 1012-view cache pages on small machines), then re-select
            // within that set for full coverage of what remains.
            let t_prune = std::time::Instant::now();
            let bake_views =
                ((n_views as f32 * cfg.bake_view_frac.clamp(0.0, 1.0)).round() as usize).max(1);
            let best: Vec<Option<u32>> = if bake_views < n_views {
                let mut counts = vec![0u32; n_views];
                for &b in &best1 {
                    if b > 0.0 {
                        counts[b as usize - 1] += 1;
                    }
                }
                let mut order: Vec<usize> = (0..n_views).collect();
                order.sort_by_key(|&v| std::cmp::Reverse(counts[v]));
                let mut keep = vec![0u32; n_views];
                for &v in order.iter().take(bake_views) {
                    keep[v] = 1;
                }
                let best2 = read_back_f32(select(keep)).await;
                log::info!(
                    "view prune: {} -> {} views ({:.2}s)",
                    n_views,
                    bake_views,
                    t_prune.elapsed().as_secs_f64()
                );
                best2
                    .iter()
                    .map(|&b| if b > 0.0 { Some(b as u32 - 1) } else { None })
                    .collect()
            } else {
                best1
                    .iter()
                    .map(|&b| if b > 0.0 { Some(b as u32 - 1) } else { None })
                    .collect()
            };

            // Observed resolution per face (pixels per world unit at
            // the chosen view) steers per-chart texel density.
            let importance: Vec<f32> = best
                .iter()
                .zip(&centers)
                .map(|(b, c)| match b {
                    Some(v) => {
                        let (cam, sz) = &views[*v as usize];
                        let z = (glam::Mat4::from(cam.world_to_local()) * c.extend(1.0)).z;
                        if z > 0.0 {
                            (cam.build_pinhole_params(*sz).fx / z).max(0.0)
                        } else {
                            0.0
                        }
                    }
                    None => 0.0,
                })
                .collect();

            if let Some((atlased, uvs, face_src)) =
                crate::texture::atlas_mesh(&mesh, size, &importance)
            {
                let best: Vec<Option<u32>> = face_src.iter().map(|&f| best[f as usize]).collect();
                mesh = atlased;
                log::info!(
                    "Baking {size}x{size} atlas ({} faces, {} verts after seam splits)",
                    mesh.faces.len(),
                    mesh.vertices.len()
                );

                // Render the views the bake actually uses at full
                // resolution, on demand; the carve cache may be low-res
                // and the other ~80% of views are never needed again.
                let mut used_idx: Vec<usize> = best.iter().flatten().map(|&v| v as usize).collect();
                used_idx.sort_unstable();
                used_idx.dedup();

                let mut unseen = 0usize;
                for b in &best {
                    if b.is_none() {
                        unseen += 1;
                    }
                }
                if unseen > 0 {
                    log::info!("{unseen} faces seen by no view");
                }

                // Smooth vertex normals: face normals oriented toward the
                // face's chosen view's camera (winding from marching tets
                // is arbitrary), then angle-free accumulation. The blend
                // weights use |cos| so residual orientation noise is
                // harmless; smoothness across faces is what matters.
                let mut vnormals = vec![Vec3::ZERO; mesh.vertices.len()];
                for (fi, f) in mesh.faces.iter().enumerate() {
                    let a = mesh.vertices[f[0] as usize];
                    let mut n =
                        (mesh.vertices[f[1] as usize] - a).cross(mesh.vertices[f[2] as usize] - a);
                    if let Some(v) = best[fi] {
                        let cam_pos = views[v as usize].0.position;
                        let centroid =
                            (a + mesh.vertices[f[1] as usize] + mesh.vertices[f[2] as usize]) / 3.0;
                        if n.dot(cam_pos - centroid) < 0.0 {
                            n = -n;
                        }
                    }
                    for &vi in f {
                        vnormals[vi as usize] += n;
                    }
                }
                for n in &mut vnormals {
                    *n = n.normalize_or_zero();
                }
                let pack_n = |n: Vec3| -> f32 {
                    let q = |x: f32| ((x * 127.0 + 128.0).clamp(0.0, 255.0)) as u32;
                    f32::from_bits(q(n.x) | (q(n.y) << 8) | (q(n.z) << 16))
                };

                // Rasterize every face once at the supersampled grid:
                // interleaved (pos.xyz, packed normal) per texel sample.
                const BAKE_SS: u32 = 2;
                let ss = size * BAKE_SS;
                let t_raster = std::time::Instant::now();
                use rayon::prelude::*;
                let per_face: Vec<(Vec<f32>, Vec<u32>)> = (0..mesh.faces.len())
                    .into_par_iter()
                    .map(|fi| {
                        let f = mesh.faces[fi];
                        let (pa, pb, pc) = (
                            mesh.vertices[f[0] as usize],
                            mesh.vertices[f[1] as usize],
                            mesh.vertices[f[2] as usize],
                        );
                        let (na, nb, nc) = (
                            vnormals[f[0] as usize],
                            vnormals[f[1] as usize],
                            vnormals[f[2] as usize],
                        );
                        let mut t = Vec::new();
                        let mut ti = Vec::new();
                        crate::texture::for_each_face_texel(
                            &mesh,
                            &uvs,
                            ss,
                            fi,
                            |idx, wa, wb, wc| {
                                let p = pa * wa + pb * wb + pc * wc;
                                let n = (na * wa + nb * wb + nc * wc).normalize_or_zero();
                                t.extend_from_slice(&[p.x, p.y, p.z, pack_n(n)]);
                                ti.push(idx);
                            },
                        );
                        (t, ti)
                    })
                    .collect();
                let mut texels: Vec<f32> = Vec::new();
                let mut texel_idx: Vec<u32> = Vec::new();
                for (t, ti) in per_face {
                    texels.extend_from_slice(&t);
                    texel_idx.extend_from_slice(&ti);
                }
                let n_texels = texel_idx.len();
                log::info!(
                    "rasterized {n_texels} texel samples ({:.2}s)",
                    t_raster.elapsed().as_secs_f64()
                );

                // Blend every kept view into per-sample accumulators:
                // weights are smooth over the surface (interpolated
                // normals, continuous visibility), so no face or chart
                // shows as a discontinuity.
                // Render and blend in small batches so only a handful of
                // full-res images are ever resident: view count stops
                // being a memory limit (only a time dial).
                let t_blend = std::time::Instant::now();
                let texels_t = create_tensor_from_slice(&texels, &device, DType::F32);
                let color_sum_t: FloatTensor<MainBackendBase> = MainBackendBase::float_from_data(
                    burn::tensor::TensorData::zeros::<f32, _>([n_texels, 3]),
                    &device,
                );
                let weight_sum_t: FloatTensor<MainBackendBase> = MainBackendBase::float_from_data(
                    burn::tensor::TensorData::zeros::<f32, _>([n_texels]),
                    &device,
                );
                {
                    type B = MainBackendBase;
                    let transforms_b = resolve_to_cube_float(splats.transforms.val());
                    let sh_coeffs_b = resolve_to_cube_float(splats.sh_coeffs.val());
                    let raw_opacities_b = resolve_to_cube_float(splats.raw_opacities.val());
                    const BAKE_VIEW_BATCH: usize = 32;
                    for batch in used_idx.chunks(BAKE_VIEW_BATCH) {
                        let mut images = Vec::with_capacity(batch.len());
                        for &v in batch {
                            let (cam, sz) = &views[v];
                            let out = <B as SplatOps>::render(
                                cam,
                                *sz,
                                transforms_b.clone(),
                                sh_coeffs_b.clone(),
                                raw_opacities_b.clone(),
                                RenderOptions::color().with_render_mode(render_mode),
                            )
                            .await;
                            images.push((v, out.out_img));
                        }
                        for (v, img) in images {
                            let (_, sz) = &views[v];
                            bake::blend_texel_colors_kernel::launch::<WgpuRuntime>(
                                &client,
                                calc_cube_count_1d(n_texels as u32, bake::BAKE_WG),
                                CubeDim::new_1d(bake::BAKE_WG),
                                texels_t.clone().into_tensor_arg(),
                                md.view_data_t.clone().into_tensor_arg(),
                                img.into_tensor_arg(),
                                md.depth_t.clone().into_tensor_arg(),
                                color_sum_t.clone().into_tensor_arg(),
                                weight_sum_t.clone().into_tensor_arg(),
                                v as u32,
                                md.offsets[v * 3],
                                md.offsets[v * 3 + 1],
                                md.offsets[v * 3 + 2],
                                sz.x,
                                sz.y,
                                n_texels as u32,
                                cfg.texture_dist_power,
                                crop_margin,
                            );
                        }
                    }
                }
                let color_sum = read_back_f32(color_sum_t).await;
                let weight_sum = read_back_f32(weight_sum_t).await;
                log::info!(
                    "rendered + blended {} views ({:.2}s)",
                    used_idx.len(),
                    t_blend.elapsed().as_secs_f64()
                );

                // Scatter samples to the supersampled grid, then box-
                // downsample covered samples into the final atlas.
                let mut ss_rgb = vec![0f32; (ss * ss) as usize * 3];
                let mut ss_w = vec![0f32; (ss * ss) as usize];
                for (si, &idx) in texel_idx.iter().enumerate() {
                    let w = weight_sum[si];
                    if w > 0.0 {
                        let o = idx as usize * 3;
                        ss_rgb[o] += color_sum[si * 3] / w;
                        ss_rgb[o + 1] += color_sum[si * 3 + 1] / w;
                        ss_rgb[o + 2] += color_sum[si * 3 + 2] / w;
                        ss_w[idx as usize] += 1.0;
                    }
                }
                rgba.par_chunks_mut((size * 4) as usize)
                    .enumerate()
                    .for_each(|(y, row)| {
                        let y = y as u32;
                        for x in 0..size {
                            let mut acc = [0f32; 3];
                            let mut cnt = 0f32;
                            for sy in 0..BAKE_SS {
                                for sx in 0..BAKE_SS {
                                    let si = ((y * BAKE_SS + sy) * ss + x * BAKE_SS + sx) as usize;
                                    if ss_w[si] > 0.0 {
                                        for (c, a) in acc.iter_mut().enumerate() {
                                            *a += ss_rgb[si * 3 + c] / ss_w[si];
                                        }
                                        cnt += 1.0;
                                    }
                                }
                            }
                            if cnt == 0.0 {
                                continue;
                            }
                            let o = (x * 4) as usize;
                            for (c, a) in acc.iter().enumerate() {
                                row[o + c] = (a / cnt).clamp(0.0, 255.0) as u8;
                            }
                            row[o + 3] = 255;
                        }
                    });
                Some(uvs)
            } else {
                log::warn!("UV atlas generation failed; skipping texture");
                None
            }
        };
        if let Some(uvs) = atlas_uvs {
            phases.push(("texture_bake", phase_start.elapsed()));
            Some(crate::texture::Texture {
                width: size,
                height: size,
                rgba,
                uvs,
            })
        } else {
            None
        }
    } else {
        None
    };

    let mut texture = texture;
    if cfg.target_faces > 0 {
        let t_simp = std::time::Instant::now();
        if let Some(t) = texture.as_mut() {
            let (m2, uv2) =
                crate::simplify::simplify_mesh_with_uvs(&mesh, &t.uvs, cfg.target_faces as usize);
            mesh = m2;
            t.uvs = uv2;
        } else if texturing {
            // Atlas failure: still honor the face target.
            mesh = crate::simplify::simplify_mesh(&mesh, cfg.target_faces as usize);
        }
        log::info!("UV simplify: {:.2}s", t_simp.elapsed().as_secs_f64());
        phases.push(("uv_simplify", t_simp.elapsed()));
    }

    let total: std::time::Duration = phases.iter().map(|(_, d)| *d).sum();
    log::info!("=== EXTRACT PROFILE ===");
    for (name, d) in &phases {
        let pct = if total.is_zero() {
            0.0
        } else {
            d.as_secs_f64() / total.as_secs_f64() * 100.0
        };
        log::info!("  {:>28}: {:>7.2}s ({:>4.1}%)", name, d.as_secs_f64(), pct);
    }
    log::info!("  {:>28}: {:>7.2}s", "TOTAL", total.as_secs_f64());

    ExtractOutput { mesh, texture }
}

fn bbox(points: impl Iterator<Item = Vec3>) -> (Vec3, Vec3) {
    points.fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(mn, mx), p| (mn.min(p), mx.max(p)),
    )
}

/// Cached per-view forward render, shared by the initial alpha eval, the
/// binary-search iters, and the colour eval (the render dominates per-view
/// cost and is identical across them).
/// The slice of a render the alpha integrate needs: per-tile gaussian
/// lists and projection uniforms. Holding full `RenderOutput`s would pin
/// the packed image and backward-pass tensors of every view.
#[derive(Clone)]
pub struct ViewRender {
    pub compact_gid_from_isect: burn::backend::tensor::IntTensor<MainBackendBase>,
    pub tile_offsets: burn::backend::tensor::IntTensor<MainBackendBase>,
    pub global_from_compact_gid: burn::backend::tensor::IntTensor<MainBackendBase>,
    pub project_uniforms: brush_render::shaders::helpers::ProjectUniforms,
    pub camera_model: brush_render::kernels::camera_model::CameraModel,
}

/// Render every training view once and stash the output. Caller hands
/// the resulting slice to each `evaluate_alpha` call below.
async fn pre_render_views(
    splats: &Splats,
    views: &[(Camera, UVec2)],
    render_mode: SplatRenderMode,
) -> Vec<ViewRender> {
    type B = MainBackendBase;
    let transforms_p = resolve_to_cube_float(splats.transforms.val());
    let sh_coeffs_p = resolve_to_cube_float(splats.sh_coeffs.val());
    let raw_opacities_p = resolve_to_cube_float(splats.raw_opacities.val());
    let mut cache = Vec::with_capacity(views.len());
    for (view_idx, (cam, sz)) in views.iter().enumerate() {
        let out = <B as SplatOps>::render(
            cam,
            *sz,
            transforms_p.clone(),
            sh_coeffs_p.clone(),
            raw_opacities_p.clone(),
            RenderOptions::color().with_render_mode(render_mode),
        )
        .await;
        cache.push(ViewRender {
            compact_gid_from_isect: out.compact_gid_from_isect,
            tile_offsets: out.aux.tile_offsets,
            global_from_compact_gid: out.global_from_compact_gid,
            project_uniforms: out.project_uniforms,
            camera_model: cam.camera_model,
        });
        if (view_idx + 1).is_multiple_of(32) || view_idx + 1 == views.len() {
            log::info!("pre-rendered view {}/{}", view_idx + 1, views.len());
        }
    }
    cache
}

/// Per-view tile-cooperative integrate over all views: project vertices to
/// tiles, histogram + prefix-sum + scatter into per-tile slices, then one
/// workgroup per tile streams gaussians through shared memory. Returns the
/// running min-alpha aggregator.
fn integrate_alpha(
    splats: &Splats,
    pts_tensor: &FloatTensor<MainBackendBase>,
    n: usize,
    view_renders: &[ViewRender],
) -> FloatTensor<MainBackendBase> {
    type B = MainBackendBase;
    use crate::kernels::integrate;
    use brush_cube::{create_tensor, create_tensor_from_slice};
    use burn::tensor::DType;

    let transforms_p = resolve_to_cube_float(splats.transforms.val());
    let raw_opacities_p = resolve_to_cube_float(splats.raw_opacities.val());
    let device = transforms_p.device.clone();
    let client = transforms_p.client.clone();

    // Aggregators: live across all per-view kernels.
    let min_alpha_t: FloatTensor<B> = B::float_from_data(
        burn::tensor::TensorData::new(vec![f32::INFINITY; n], [n]),
        &device,
    );

    // Per-view scratch, overwritten each view; only the aggregators carry.
    let tile_ids_t = create_tensor::<1>([n], &device, DType::U32);
    let depths_t = create_tensor::<1>([n], &device, DType::F32);
    let ray_dir_xy_t = create_tensor::<2>([n, 2], &device, DType::F32);
    let sorted_indices_t = create_tensor::<1>([n], &device, DType::U32);

    for view_render in view_renders {
        let out = view_render.clone();
        let camera_model = view_render.camera_model;
        let tile_bw = out.project_uniforms.tile_bounds[0];
        let tile_bh = out.project_uniforms.tile_bounds[1];
        let n_tiles = (tile_bw * tile_bh) as usize;

        integrate::project_vertices_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(n as u32, integrate::TILE_SIZE),
            CubeDim::new_1d(integrate::TILE_SIZE),
            pts_tensor.clone().into_tensor_arg(),
            tile_ids_t.clone().into_tensor_arg(),
            depths_t.clone().into_tensor_arg(),
            ray_dir_xy_t.clone().into_tensor_arg(),
            n as u32,
            out.project_uniforms.to_launch_object(),
            camera_model,
        );

        let counts_t = create_tensor_from_slice(&vec![0u32; n_tiles + 1], &device, DType::U32);
        integrate::histogram_tile_ids_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(n as u32, integrate::TILE_SIZE),
            CubeDim::new_1d(integrate::TILE_SIZE),
            tile_ids_t.clone().into_tensor_arg(),
            counts_t.clone().into_tensor_arg(),
            n as u32,
            tile_bw * tile_bh,
        );
        let vertex_tile_offsets_t = brush_prefix_sum::prefix_sum(counts_t);

        let write_counters_t = create_tensor_from_slice(&vec![0u32; n_tiles], &device, DType::U32);
        integrate::scatter_vertices_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(n as u32, integrate::TILE_SIZE),
            CubeDim::new_1d(integrate::TILE_SIZE),
            tile_ids_t.clone().into_tensor_arg(),
            vertex_tile_offsets_t.clone().into_tensor_arg(),
            write_counters_t.clone().into_tensor_arg(),
            sorted_indices_t.clone().into_tensor_arg(),
            n as u32,
            tile_bw * tile_bh,
        );

        // One workgroup per tile (CUBE_POS = tile id, like rasterize).
        let num_tiles_u32 = tile_bw * tile_bh;
        integrate::integrate_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(num_tiles_u32 * integrate::TILE_SIZE, integrate::TILE_SIZE),
            CubeDim::new_1d(integrate::TILE_SIZE),
            transforms_p.clone().into_tensor_arg(),
            raw_opacities_p.clone().into_tensor_arg(),
            out.compact_gid_from_isect.into_tensor_arg(),
            out.tile_offsets.into_tensor_arg(),
            out.global_from_compact_gid.into_tensor_arg(),
            sorted_indices_t.clone().into_tensor_arg(),
            vertex_tile_offsets_t.into_tensor_arg(),
            depths_t.clone().into_tensor_arg(),
            ray_dir_xy_t.clone().into_tensor_arg(),
            min_alpha_t.clone().into_tensor_arg(),
            out.project_uniforms.to_launch_object(),
        );
    }

    min_alpha_t
}

/// Per-view camera tensors plus shadow-map style mesh depth buffers (one
/// concatenated buffer at 1/4 view resolution): the visibility oracle for
/// bake view selection.
struct MeshDepth {
    view_data_t: brush_cube::CubeTensor<WgpuRuntime>,
    offsets_t: brush_cube::CubeTensor<WgpuRuntime>,
    offsets: Vec<u32>,
    depth_t: brush_cube::CubeTensor<WgpuRuntime>,
}

use crate::kernels::bake::DEPTH_DOWNSCALE;

fn raster_mesh_depth(device: &WgpuDevice, mesh: &Mesh, views: &[(Camera, UVec2)]) -> MeshDepth {
    use crate::kernels::bake;
    use brush_cube::create_tensor_from_slice;
    use burn::tensor::DType;
    use burn_cubecl::cubecl::Runtime;
    let client = WgpuRuntime::client(device);

    let n_views = views.len();
    let mut view_data: Vec<f32> = Vec::with_capacity(n_views * 16);
    let mut offsets: Vec<u32> = Vec::with_capacity(n_views * 3);
    let mut depth_total = 0u32;
    let inv_ds = 1.0 / DEPTH_DOWNSCALE as f32;
    for (cam, sz) in views {
        let w2c = glam::Mat4::from(cam.world_to_local());
        for r in 0..3 {
            let row = w2c.row(r);
            view_data.extend_from_slice(&[row.x, row.y, row.z, row.w]);
        }
        let pin = cam.build_pinhole_params(*sz);
        view_data.extend_from_slice(&[
            pin.fx * inv_ds,
            pin.fy * inv_ds,
            pin.cx * inv_ds,
            pin.cy * inv_ds,
        ]);
        let qw = sz.x.div_ceil(DEPTH_DOWNSCALE);
        let qh = sz.y.div_ceil(DEPTH_DOWNSCALE);
        offsets.extend_from_slice(&[depth_total, qw, qh]);
        depth_total += qw * qh;
    }
    let view_data_t = create_tensor_from_slice(&view_data, device, DType::F32);
    let offsets_t = create_tensor_from_slice(&offsets, device, DType::U32);
    let depth_t = create_tensor_from_slice(
        &vec![f32::INFINITY.to_bits(); depth_total as usize],
        device,
        DType::U32,
    );
    let verts_flat: Vec<f32> = mesh.vertices.iter().flat_map(|v| [v.x, v.y, v.z]).collect();
    let verts_t = create_tensor_from_slice(&verts_flat, device, DType::F32);
    let faces_flat: Vec<u32> = mesh.faces.iter().flatten().copied().collect();
    let faces_t = create_tensor_from_slice(&faces_flat, device, DType::U32);
    let n_faces = mesh.faces.len() as u32;
    for v in 0..n_views {
        bake::raster_depth_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(n_faces, bake::BAKE_WG),
            CubeDim::new_1d(bake::BAKE_WG),
            verts_t.clone().into_tensor_arg(),
            faces_t.clone().into_tensor_arg(),
            view_data_t.clone().into_tensor_arg(),
            depth_t.clone().into_tensor_arg(),
            v as u32,
            offsets[v * 3],
            offsets[v * 3 + 1],
            offsets[v * 3 + 2],
            n_faces,
        );
    }
    MeshDepth {
        view_data_t,
        offsets_t,
        offsets,
        depth_t,
    }
}

fn points_tensor(points: &[Vec3], splats: &Splats) -> FloatTensor<MainBackendBase> {
    let device = resolve_to_cube_float(splats.transforms.val()).device;
    let pts_flat: Vec<f32> = points.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
    MainBackendBase::float_from_data(
        burn::tensor::TensorData::new(pts_flat, [points.len(), 3]),
        &device,
    )
}

/// Per-point alpha over all views (GOF carving rule): `alpha(p) =
/// max_views(T_view(p))`, so a point is open as soon as any view sees
/// through to it and solid only when every view is blocked. Never-seen
/// points (no view in frustum) default to open space (alpha = 1).
async fn evaluate_alpha(splats: &Splats, points: &[Vec3], view_renders: &[ViewRender]) -> Vec<f32> {
    if points.is_empty() {
        return Vec::new();
    }
    let pts_tensor = points_tensor(points, splats);
    let min_alpha_t = integrate_alpha(splats, &pts_tensor, points.len(), view_renders);
    read_back_f32(min_alpha_t)
        .await
        .iter()
        .map(|&a| if a.is_finite() { 1.0 - a } else { 1.0 })
        .collect()
}
