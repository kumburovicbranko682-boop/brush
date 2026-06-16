use brush_render::gaussian_splats::SplatRenderMode;
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Clone, Parser, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct TrainConfig {
    /// Total number of steps to train for.
    #[arg(long, help_heading = "Training options", default_value = "30000")]
    pub total_train_iters: u32,

    #[arg(long, help_heading = "Training options")]
    pub render_mode: Option<SplatRenderMode>,

    /// Start learning rate for the mean parameters.
    #[arg(long, help_heading = "Training options", default_value = "2e-5")]
    pub lr_mean: f64,

    /// Start learning rate for the mean parameters.
    #[arg(long, help_heading = "Training options", default_value = "2e-7")]
    pub lr_mean_end: f64,

    /// How much noise to add to the mean parameters of low opacity gaussians.
    #[arg(long, help_heading = "Training options", default_value = "50.0")]
    pub mean_noise_weight: f32,

    /// Learning rate for the base SH (RGB) coefficients.
    #[arg(long, help_heading = "Training options", default_value = "2e-3")]
    pub lr_coeffs_dc: f64,

    /// How much to divide the learning rate by for higher SH orders.
    #[arg(long, help_heading = "Training options", default_value = "10.0")]
    pub lr_coeffs_sh_scale: f32,

    /// Learning rate for the opacity parameter.
    #[arg(long, help_heading = "Training options", default_value = "0.012")]
    pub lr_opac: f64,

    /// Learning rate for the scale parameters.
    #[arg(long, help_heading = "Training options", default_value = "5e-3")]
    pub lr_scale: f64,

    /// Learning rate for the rotation parameters.
    #[arg(long, help_heading = "Training options", default_value = "2e-3")]
    pub lr_rotation: f64,

    /// Max nr. of splats. This is only an upper bound, the actual final number of splats is NOT determined by this.
    #[arg(long, help_heading = "Refine options", default_value = "10000000")]
    pub max_splats: u32,

    /// Frequency of 'refinement' where gaussians are replaced and densified. This should
    /// roughly be the number of images it takes to properly "cover" your scene.
    #[arg(
        long,
        help_heading = "Refine options",
        default_value = "200",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub refine_every: u32,

    /// Threshold to control splat growth. Lower means faster growth.
    #[arg(long, help_heading = "Refine options", default_value = "0.0025")]
    pub growth_grad_threshold: f32,

    /// What fraction of splats that are deemed as needing to grow do actually grow.
    /// Increase this to make splats grow more aggressively.
    #[arg(long, help_heading = "Refine options", default_value = "0.25")]
    pub growth_select_fraction: f32,

    /// Period after which splat growth stops.
    #[arg(long, help_heading = "Refine options", default_value = "15000")]
    pub growth_stop_iter: u32,

    /// Split any splat whose max screen-space extent exceeds this fraction of
    /// the image dimension, shrinking the children so they land at (at most)
    /// this size on screen. 0 disables.
    #[arg(long, help_heading = "Refine options", default_value = "0.5")]
    pub split_at_screen_size: f32,

    /// Weight of SSIM loss (compared to l1 loss)
    #[clap(long, help_heading = "Training options", default_value = "0.2")]
    pub ssim_weight: f32,

    /// Factor of the opacity decay.
    #[arg(long, help_heading = "Training options", default_value = "0.004")]
    pub opac_decay: f32,

    /// Weight of l1 loss on alpha if input view has transparency.
    #[arg(long, help_heading = "Refine options", default_value = "0.1")]
    pub match_alpha_weight: f32,

    #[arg(long, help_heading = "Refine options", default_value = "0.0")]
    pub lpips_loss_weight: f32,

    /// Weight of the single-view depth <-> normal consistency loss.
    #[arg(long, help_heading = "Geometry options", default_value = "0.05")]
    pub depth_normal_weight: f32,

    /// Weight of the metric depth supervision loss: L1 between the rendered
    /// depth and the per-view `LiDAR` depth (when the dataset provides it),
    /// confidence-weighted and sparse (at the `LiDAR` grid).
    #[arg(long, help_heading = "Geometry options", default_value = "0.4")]
    pub depth_loss_weight: f32,

    /// Run the alpha-matching loss even when a view has no alpha channel, by
    /// treating the view as fully opaque (alpha == 1). Pulls rendered alpha to 1
    /// over the whole frame so the reconstruction is never see-through (the
    /// transparency trap where a wall stays too transparent to receive depth
    /// gradient). Uses `--match-alpha-weight`. No-op on views with real alpha.
    #[arg(long, help_heading = "Geometry options", default_value = "false")]
    pub force_alpha_loss: bool,

    /// Weight of the depth-distortion loss (GOF `L_d`: squared pairwise error
    /// over NDC-mapped depths, normalized per pixel): pulls each ray's splats
    /// onto a single depth so the surface stops being a fuzzy shell.
    #[arg(long, help_heading = "Geometry options", default_value = "100.0")]
    pub distortion_weight: f32,

    /// Master switch for the self-consistency geometry regularizers: the
    /// iteration to turn them on at (depth-normal + depth-distortion, by their
    /// weights above). Active from the start by default; pass a later iteration
    /// to delay them.
    #[arg(long, help_heading = "Geometry options", default_value = "0")]
    pub geo_from_iter: Option<u32>,

    /// `LiDAR` init cell size in metres: one oriented surfel per occupied cell
    /// per surface-normal direction. Metric, so density is fixed regardless of
    /// object size (a bigger object gets proportionally more seeds).
    #[arg(long, help_heading = "Geometry options", default_value = "0.02")]
    pub lidar_voxel_size: f32,

    /// `ARKit` confidence floor (0/1/2) shared by the init seeds and the depth
    /// loss: a return is used only at confidence >= this. Lower admits noisier
    /// returns for more coverage.
    #[arg(long, help_heading = "Geometry options", default_value = "2")]
    pub lidar_min_conf: u8,

    /// Trust `LiDAR` only out to this distance in metres, for both the init
    /// seeds and the depth loss: returns past it are accurate but add unneeded
    /// far geometry. No-return pixels (+inf) are excluded by the same gate.
    #[arg(long, help_heading = "Geometry options", default_value = "2.0")]
    pub lidar_max_depth: f32,

    /// Base background color (R,G,B) used during training.
    #[arg(
        long,
        help_heading = "Training options",
        default_value = "0,0,0",
        value_delimiter = ',',
        num_args = 3
    )]
    pub background_color: Vec<f32>,

    /// Mip-Splatting 3D-filter strength.
    #[arg(long, help_heading = "Training options", default_value = "0.15")]
    pub min_scale_factor: f32,

    /// Strength of random noise added to the background color each step.
    /// Noise is uniform in [-strength, +strength], clamped to [0, 1].
    #[arg(long, help_heading = "Training options", default_value = "0.1")]
    pub background_noise_strength: f32,

    /// Number of LOD levels to generate after initial training (0 = disabled).
    #[arg(long, help_heading = "LOD options", default_value = "0")]
    pub lod_levels: u32,

    /// Number of refinement training steps per LOD level.
    #[arg(long, help_heading = "LOD options", default_value = "5000")]
    pub lod_refine_steps: u32,

    /// Percentage of gaussians to keep at each LOD level (1-100).
    #[arg(long, help_heading = "LOD options", default_value = "50")]
    pub lod_decimation_keep: u32,

    /// Percentage to scale source images at each LOD level (1-100).
    #[arg(long, help_heading = "LOD options", default_value = "50")]
    pub lod_image_scale: u32,

    /// Scene scale used for random splat initialization.
    /// When no init is provided, splats are randomly placed
    /// inside camera frustums up to this depth. By default this is
    /// estimated from the camera spacing (with a 1m minimum).
    #[arg(long, help_heading = "Training options")]
    pub random_init_scene_scale: Option<f32>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self::parse_from([""])
    }
}

impl TrainConfig {
    pub fn total_iters(&self) -> u32 {
        self.total_train_iters + self.lod_levels * self.lod_refine_steps
    }

    /// Whether the self-consistency geometry regularizers (depth-normal,
    /// distortion) are active at `iter` (gated by `geo_from_iter`).
    pub fn geo_regs_on(&self, iter: u32) -> bool {
        self.geo_from_iter.is_some_and(|from| iter >= from)
    }
}
