//! Extraction-parameter lab (experiment harness, not part of the product
//! pipeline): runs GOF extraction on a fixed splat PLY with overridable
//! `ExtractConfig` knobs and writes the mesh for `brush-mesh-eval` to score.

use std::pin::pin;

use anyhow::Context;
use brush_dataset::config::LoadDataseConfig;
use brush_dataset::load_dataset;
use brush_render::gaussian_splats::SplatRenderMode;
use brush_vfs::DataSource;
use clap::Parser;
use tokio_stream::StreamExt;

#[derive(Parser)]
struct Args {
    #[arg(value_name = "PATH_OR_URL")]
    source: DataSource,
    #[arg(long)]
    ply: std::path::PathBuf,
    #[arg(long)]
    out_mesh: std::path::PathBuf,
    #[arg(long, default_value = "0.5")]
    iso: f32,
    #[arg(long, default_value = "1")]
    smooth_iters: u32,
    /// Multiplier on the edge-scale crossing filter (large = off).
    #[arg(long, default_value = "1.0")]
    edge_filter_scale: f32,
    #[arg(long, default_value = "500")]
    min_component: usize,
    /// Drop components below this fraction of the largest one.
    #[arg(long, default_value = "0.01")]
    min_component_frac: f32,
    /// Mesh region: union of camera frustums truncated at this distance.
    #[arg(long, default_value = "2.5")]
    far: f32,
    /// Seed only gaussians with carve-field center alpha at most this
    /// Simplify to roughly this many faces before texturing (0 = off).
    #[arg(long, default_value = "500000")]
    target_faces: u32,
    /// Atlas side in texels for the baked color texture.
    #[arg(long, default_value = "4096", value_parser = clap::value_parser!(u32).range(64..=16384))]
    texture_size: u32,
    /// Cap on faces entering charting/baking, as a multiple of target
    /// faces (0 = off).
    #[arg(long, default_value = "2")]
    pre_simplify_factor: usize,
    /// Fraction of input views the bake blends.
    #[arg(long, default_value = "0.8")]
    bake_view_frac: f32,
    /// Distance falloff exponent in the blend weight (0 = none).
    #[arg(long, default_value = "2.0")]
    texture_dist_power: f32,
    /// Resolution scale for the carve's cached renders.
    #[arg(long, default_value = "0.5")]
    carve_scale: f32,
    #[arg(long, default_value = "1920")]
    resolution: u32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_target(false)
        .try_init();
    let args = Args::parse();

    brush_process::burn_init_setup().await;
    let device = brush_process::wait_for_device().await.clone();
    let device: burn::tensor::Device = device.into();

    let bytes = std::fs::read(&args.ply)
        .with_context(|| format!("read splat ply {}", args.ply.display()))?;
    let mut stream = pin!(brush_serde::stream_splat_from_ply(
        std::io::Cursor::new(bytes),
        None,
        false
    ));
    let mut splat_msg = None;
    while let Some(msg) = stream.next().await {
        splat_msg = Some(msg?);
    }
    let splat_msg = splat_msg.context("PLY produced no splats")?;
    let mode = splat_msg
        .meta
        .render_mode
        .unwrap_or(SplatRenderMode::Default);
    let splats = splat_msg.data.into_splats(&device, mode);
    log::info!("Loaded {} splats", splats.num_splats());

    let vfs = args.source.into_vfs().await?;
    let load_cfg = LoadDataseConfig {
        max_frames: None,
        max_resolution: args.resolution,
        eval_split_every: None,
        subsample_frames: None,
        subsample_points: None,
        alpha_mode: None,
    };
    let dataset = load_dataset(vfs, &load_cfg).await?;

    let mut views: Vec<(brush_render::camera::Camera, glam::UVec2)> = Vec::new();
    for view in dataset.dataset.train.views.iter() {
        let (w, h) = view.image.dimensions().await.unwrap_or((1, 1));
        views.push((view.camera.with_pinhole(), glam::uvec2(w, h)));
    }

    let tetra_points = brush_mesh::tetra_points::TetraPointsConfig {
        far: args.far,
        ..Default::default()
    };
    let cfg = brush_mesh::ExtractConfig {
        tetra_points,
        iso_value: args.iso,
        smooth_iters: args.smooth_iters,
        edge_filter_scale: args.edge_filter_scale,
        min_component_faces: args.min_component,
        min_component_frac: args.min_component_frac,
        target_faces: args.target_faces,
        texture_size: args.texture_size,
        pre_simplify_factor: args.pre_simplify_factor,
        bake_view_frac: args.bake_view_frac,
        texture_dist_power: args.texture_dist_power,
        carve_scale: args.carve_scale,
        ..Default::default()
    };
    let out = brush_mesh::extract_mesh(splats, &views, &cfg).await;

    if let Some(parent) = args.out_mesh.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let glb_path = args.out_mesh.with_extension("glb");
    let tex = out
        .texture
        .as_ref()
        .context("extraction produced no texture (empty mesh or atlas failure)")?;
    brush_mesh::gltf::write_glb(&out.mesh, tex, &glb_path)
        .with_context(|| format!("writing mesh {}", glb_path.display()))?;
    Ok(())
}
