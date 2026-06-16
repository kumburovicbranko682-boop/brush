//! Viewer-side geometry visualization: colormap the rendered geometry channels
//! for display (absolute magma depth / hemisphere normals).

use brush_render::{
    burn_glue::{GeoColormapMode, geo_colormap_pack},
    geo::{rendered_depth, rendered_normal},
};
use burn::tensor::{Tensor, s};

use crate::ui::app::ViewChannel;

/// Colormap a geometry render into a packed RGBA8 image for the given viewer
/// channel. `img` is the full `[H, W, 11]` geometry render; this reads the
/// `rgba` then `Nx,Ny,Nz,depth` channels and ignores the distortion lanes.
/// Depth uses an absolute `[0, max_meters]` magma ramp; normals are mapped to
/// the RGB hemisphere.
pub fn visualize_geo(img: Tensor<3>, channel: ViewChannel) -> Tensor<3> {
    let alpha = img.clone().slice(s![.., .., 3..4]);
    let geo = img.slice(s![.., .., 4..8]);
    let normal = rendered_normal(geo.clone());
    let depth = rendered_depth(geo);
    // "No depth" below 0.8 coverage: the expected depth `depth/alpha` is
    // unreliable there, so show those pixels transparent rather than colormapped.
    let covered = alpha.greater_equal_elem(0.8).float();
    let (mode, range) = match channel {
        ViewChannel::Depth { max_meters } => (GeoColormapMode::Depth, max_meters),
        ViewChannel::Normal => (GeoColormapMode::Normal, 1.0),
        _ => unreachable!("visualize_geo handles Depth/Normal only"),
    };
    geo_colormap_pack(depth, normal, covered, 0.0, range.max(1e-6), mode)
}

/// Grayscale the alpha channel of a plain `[H, W, 4]` float render: the
/// coverage view needs no geometry buffers.
pub fn visualize_alpha(img: Tensor<3>) -> Tensor<3> {
    let alpha = img.slice(s![.., .., 3..4]);
    let normal = Tensor::zeros_like(&alpha.clone().repeat_dim(2, 3));
    geo_colormap_pack(
        alpha.clone(),
        normal,
        alpha,
        0.0,
        1.0,
        GeoColormapMode::Alpha,
    )
}
