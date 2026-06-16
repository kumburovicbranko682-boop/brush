//! Mesh-extraction `CubeCL` kernels: opacity-along-ray integration, the
//! bisection helpers, and the texture-bake passes. They build on
//! `brush_render`'s shared kernel types/helpers (brush-mesh depends on
//! brush-render, not the other way around).

#![allow(
    clippy::doc_markdown,
    clippy::manual_div_ceil,
    clippy::manual_range_contains,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::excessive_precision,
    clippy::should_implement_trait,
    clippy::similar_names
)]

pub mod bake;
pub mod bisect;
pub mod integrate;
