#![allow(clippy::match_wildcard_for_single_variants)]

use brush_cube::{MainBackend, MainBackendBase};
use burn::backend::{
    Autodiff, AutodiffBackend, BackendTensor, CheckpointingStrategy, DispatchTensor,
    DispatchTensorKind, TensorMetadata,
    tensor::{FloatTensor, IntTensor},
};
use burn::tensor::{DType, Device, Int, Shape, Tensor};
use burn_cubecl::cubecl::CubeDim;
use burn_cubecl::cubecl::Runtime;
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::tensor::CubeTensor;
use burn_fusion::{
    Fusion, FusionHandle,
    stream::{Operation, StreamId},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use burn_wgpu::{WgpuDevice, WgpuRuntime};

use crate::{RenderAuxInner, SplatOps, camera::Camera, render_aux::RenderOutput, wgpu_kind};

/// Inner Wgpu autodiff backend (same as `Autodiff<burn::backend::Wgpu>`).
/// Used as the primitive backend for autodiff `Tensor<D>` operations.
pub type AutodiffMain = Autodiff<MainBackend>;

// ---------------------------------------------------------------------------
// `Tensor<D>` ↔ backend-level primitive bridges.
//
// `Tensor<D>` is pinned to burn's `Dispatch` backend; brush only ever runs on
// a wgpu device, so every helper here assumes a `DispatchTensorKind::Wgpu`
// (optionally wrapped in `Autodiff`) and panics otherwise. The forward render
// now goes through the `#[backend_extension]`-generated `Dispatch` impl
// instead; these stay for the hand-rolled backward path (brush-render-bwd)
// and the LPIPS custom ops (brush-loss).
// ---------------------------------------------------------------------------

/// Extract the inner fusion-Wgpu float tensor from a non-autodiff
/// `Tensor<D>`.
pub fn unwrap_wgpu_float<const D: usize>(t: Tensor<D>) -> FloatTensor<MainBackend> {
    let dispatch: DispatchTensor = t.into_dispatch();
    match dispatch.kind {
        wgpu_kind!(bt) => bt.float(),
        other => panic!(
            "expected Wgpu tensor, got: {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

/// Extract the inner fusion-Wgpu int tensor from a non-autodiff
/// `Tensor<D, Int>`.
pub fn unwrap_wgpu_int<const D: usize>(t: Tensor<D, Int>) -> IntTensor<MainBackend> {
    let dispatch: DispatchTensor = t.into_dispatch();
    match dispatch.kind {
        wgpu_kind!(bt) => bt.int(),
        other => panic!(
            "expected Wgpu int tensor, got: {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

/// Inverse of [`unwrap_wgpu_float`]: wraps a fusion-Wgpu float tensor as a
/// user-facing `Tensor<D>`.
pub fn wrap_wgpu_float<const D: usize>(t: FloatTensor<MainBackend>) -> Tensor<D> {
    Tensor::from_dispatch(DispatchTensor {
        kind: wgpu_kind!(BackendTensor::Float(t)),
        checkpointing: None,
    })
}

/// Like [`wrap_wgpu_float`] for an int tensor.
pub fn wrap_wgpu_int<const D: usize>(t: IntTensor<MainBackend>) -> Tensor<D, Int> {
    Tensor::from_dispatch(DispatchTensor {
        kind: wgpu_kind!(BackendTensor::Int(t)),
        checkpointing: None,
    })
}

/// Extract the inner `AutodiffTensor<MainBackend>` from a `Tensor<D>` on an
/// autodiff-enabled Wgpu device. Panics on any other shape.
pub fn unwrap_ad_wgpu_float<const D: usize>(t: Tensor<D>) -> FloatTensor<AutodiffMain> {
    let prim: DispatchTensor = t.into_dispatch();
    match prim.kind {
        DispatchTensorKind::Autodiff(inner) => match *inner {
            wgpu_kind!(BackendTensor::Autodiff(t)) => t,
            other => panic!(
                "autodiff inner kind is not Wgpu: {:?}",
                std::mem::discriminant(&other)
            ),
        },
        other => panic!(
            "expected autodiff-enabled tensor; got: {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

/// Extract the inner Wgpu `IntTensor` regardless of whether the tensor is
/// wrapped in an autodiff device — ints are never autodiff-tracked.
pub fn unwrap_ad_wgpu_int<const D: usize>(t: Tensor<D, Int>) -> IntTensor<MainBackend> {
    let dispatch: DispatchTensor = t.into_dispatch();
    let kind = match dispatch.kind {
        DispatchTensorKind::Autodiff(inner) => *inner,
        other => other,
    };
    match kind {
        wgpu_kind!(bt) => bt.int(),
        other => panic!(
            "expected Wgpu int tensor; got: {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

/// Inverse of [`unwrap_ad_wgpu_float`]: wraps an autodiff tensor as a
/// user-facing `Tensor<D>` on the autodiff device.
pub fn wrap_ad_wgpu_float<const D: usize>(t: FloatTensor<AutodiffMain>) -> Tensor<D> {
    Tensor::from_dispatch(DispatchTensor {
        kind: DispatchTensorKind::Autodiff(Box::new(wgpu_kind!(BackendTensor::Autodiff(t)))),
        checkpointing: Some(CheckpointingStrategy::None),
    })
}

/// Strip the autodiff wrapping from a `Tensor<D>` and clear the residual
/// `checkpointing` field.
///
/// Operates directly on the `DispatchTensor` kind so it works both for an
/// autodiff input (unwrap one level) and an already-inner input (passthrough),
/// always landing with `checkpointing: None`. The high-level `.inner()` can't
/// stand in here: it panics on a non-autodiff input, and (via the Bridge path)
/// doesn't reliably normalise `checkpointing`, which downstream ops read as a
/// "came from autodiff" signal and use to re-lift — tripping cross-backend
/// asserts when mixed with a genuinely-inner tensor.
pub fn detach_autodiff<const D: usize>(t: Tensor<D>) -> Tensor<D> {
    let dispatch: DispatchTensor = t.into_dispatch();
    let kind = match dispatch.kind {
        DispatchTensorKind::Autodiff(inner) => *inner,
        other => other,
    };
    Tensor::from_dispatch(DispatchTensor {
        kind,
        checkpointing: None,
    })
}

/// Lift a non-autodiff `Tensor<D>` into the autodiff graph as a constant.
/// A no-op if `t` is already autodiff.
///
/// Lifts at the concrete-Wgpu autodiff level and re-wraps with an explicit
/// `checkpointing`. The high-level `Tensor::from_inner` goes through the
/// Bridge/Dispatch path, which doesn't set `checkpointing` the way the mixed
/// inner/autodiff folds (e.g. `fold_min_scale`) need — a lifted constant then
/// degrades to the inner backend on the next op and trips a cross-backend
/// assert. Keep the hand-rolled lift.
pub fn lift_to_autodiff<const D: usize>(t: Tensor<D>) -> Tensor<D> {
    let dispatch: DispatchTensor = t.into_dispatch();
    match dispatch.kind {
        wgpu_kind!(BackendTensor::Float(inner)) => {
            wrap_ad_wgpu_float(<AutodiffMain as AutodiffBackend>::from_inner(inner))
        }
        DispatchTensorKind::Autodiff(_) => Tensor::from_dispatch(dispatch),
        _ => panic!("expected Wgpu tensor to lift to autodiff"),
    }
}

fn is_autodiff<const D: usize>(t: &Tensor<D>) -> bool {
    matches!(
        t.clone().into_dispatch().kind,
        DispatchTensorKind::Autodiff(_)
    )
}

/// Put `t` on the same autodiff/inner backend variant as `reference`. Brush
/// keeps some frozen tensors (e.g. the 3D-filter floor) on the inner backend
/// but folds them against params that may be lifted to autodiff; this aligns
/// both operands so dispatch ops don't trip a cross-backend assertion.
pub fn match_backend<const D: usize, const DR: usize>(
    t: Tensor<D>,
    reference: &Tensor<DR>,
) -> Tensor<D> {
    if is_autodiff(reference) {
        lift_to_autodiff(t)
    } else {
        detach_autodiff(t)
    }
}

/// Like [`detach_autodiff`] for `Tensor<D, Int>`.
pub fn detach_autodiff_int<const D: usize>(t: Tensor<D, Int>) -> Tensor<D, Int> {
    let dispatch: DispatchTensor = t.into_dispatch();
    let kind = match dispatch.kind {
        DispatchTensorKind::Autodiff(inner) => *inner,
        other => other,
    };
    Tensor::from_dispatch(DispatchTensor {
        kind,
        checkpointing: None,
    })
}

/// Resolve a `Tensor<D>` on a Wgpu device down to the underlying
/// `CubeTensor<WgpuRuntime>`, draining any pending fusion ops. Useful for
/// direct GPU resource access (e.g. binding the buffer into a wgpu pipeline).
pub fn resolve_to_cube_float<const D: usize>(tensor: Tensor<D>) -> CubeTensor<WgpuRuntime> {
    let fusion = unwrap_wgpu_float(tensor);
    let client = fusion.client.clone();
    client.resolve_tensor_float::<MainBackendBase>(fusion)
}

/// Which geometry channel the colormap kernel renders.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Default)]
pub enum GeoColormapMode {
    #[default]
    Depth = 0,
    Normal = 1,
    Alpha = 2,
}

/// Colormap the splat depth / normal maps into a packed RGBA8 `[H, W, 1]`
/// image for the viewer. The output dtype is `u32` (packed) but it is
/// wrapped as a float `Tensor` so the display path can read its buffer
/// directly, matching the packed-rgb render path. `mode` selects depth magma
/// over `[dmin, dmax]`, normal hemisphere, or alpha grayscale.
pub fn geo_colormap_pack(
    depth: Tensor<3>,
    normal: Tensor<3>,
    alpha: Tensor<3>,
    dmin: f32,
    dmax: f32,
    mode: GeoColormapMode,
) -> Tensor<3> {
    let [h, w, _] = depth.dims();
    let dinv_range = if dmax > dmin {
        1.0 / (dmax - dmin)
    } else {
        0.0
    };

    let depth = unwrap_wgpu_float(depth);
    let normal = unwrap_wgpu_float(normal);
    let alpha = unwrap_wgpu_float(alpha);

    #[derive(Debug)]
    struct Op {
        desc: CustomOpIr,
        dmin: f32,
        dinv_range: f32,
        h: usize,
        w: usize,
        mode: GeoColormapMode,
    }

    impl Operation<FusionCubeRuntime<WgpuRuntime>> for Op {
        fn execute(&self, h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>) {
            let (inputs, outputs) = self.desc.as_fixed::<3, 1>();
            let [depth, normal, alpha] = inputs;
            let [out] = outputs;

            let depth = h.get_float_tensor::<MainBackendBase>(depth);
            let normal = h.get_float_tensor::<MainBackendBase>(normal);
            let alpha = h.get_float_tensor::<MainBackendBase>(alpha);

            let device = depth.device.clone();
            let client = depth.client.clone();
            let num_pixels = (self.h * self.w) as u32;
            let out_t = brush_cube::create_tensor([self.h, self.w, 1], &device, DType::U32);

            crate::kernels::geo_visualize::colormap_pack_kernel::launch::<WgpuRuntime>(
                &client,
                brush_cube::calc_cube_count_1d(num_pixels, crate::kernels::geo_visualize::WG_SIZE),
                CubeDim::new_1d(crate::kernels::geo_visualize::WG_SIZE),
                depth.into_tensor_arg(),
                normal.into_tensor_arg(),
                alpha.into_tensor_arg(),
                out_t.clone().into_tensor_arg(),
                self.dmin,
                self.dinv_range,
                num_pixels,
                self.mode,
            );

            h.register_float_tensor::<MainBackendBase>(&out.id, out_t);
        }
    }

    let client = depth.client.clone();
    let inputs = [depth, normal, alpha];
    // Declared F32 so wrap_wgpu_float treats the packed-u32 buffer as a
    // float tensor (same trick as the packed-rgb render output).
    let out_ir = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([h, w, 1]),
        DType::F32,
    );
    let stream = StreamId::current();
    let desc = CustomOpIr::new("geo_colormap_pack", &inputs.map(|t| t.into_ir()), &[out_ir]);
    let op = Op {
        desc: desc.clone(),
        dmin,
        dinv_range,
        h,
        w,
        mode,
    };
    let [out] = client
        .register(stream, OperationIr::Custom(desc), op)
        .outputs();
    wrap_wgpu_float(out)
}

/// Resolve a (non-autodiff) brush `Device` to its `WgpuDevice`. brush only ever
/// runs on a wgpu device; panics on any other backend. `Device` is the
/// `Dispatch` backend's associated device type, which the compiler won't unify
/// with the concrete `DispatchDevice` enum here, so we bounce through a tiny
/// tensor (whose `CubeTensor` carries the resolved `WgpuDevice`) rather than
/// matching the dispatch device enum directly.
pub fn wgpu_device(device: &Device) -> WgpuDevice {
    resolve_to_cube_float(Tensor::<1>::zeros([1], device)).device
}

/// True undistorted z=1 camera-ray grid `[H, W, 3]` for `camera` at `img_size`,
/// one ray per pixel center via the shared in-kernel `unproject_ray`. Returns a
/// non-autodiff constant (splat-independent): the depth-normal consistency loss
/// multiplies it by the GOF median depth to recover camera-space surface points
/// using true rays for any lens, not the pinhole approximation. `device` is the
/// inner (non-autodiff) dispatch device.
pub fn unproject_ray_grid(device: &Device, img_size: glam::UVec2, camera: &Camera) -> Tensor<3> {
    let device = wgpu_device(device);
    let device = &device;
    let h = img_size.y as usize;
    let w = img_size.x as usize;
    let pinhole = camera.build_pinhole_params(img_size);
    let camera_model = camera.camera_model;

    #[derive(Debug)]
    struct Op {
        desc: CustomOpIr,
        device: WgpuDevice,
        h: usize,
        w: usize,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        camera_model: crate::kernels::camera_model::CameraModel,
    }

    impl Operation<FusionCubeRuntime<WgpuRuntime>> for Op {
        fn execute(&self, h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>) {
            let (_inputs, outputs) = self.desc.as_fixed::<0, 1>();
            let [out] = outputs;

            let client = WgpuRuntime::client(&self.device);
            let img_w = self.w as u32;
            let img_h = self.h as u32;
            let num_pixels = img_w * img_h;
            let out_t = brush_cube::create_tensor([self.h, self.w, 3], &self.device, DType::F32);

            crate::kernels::unproject::unproject_ray_grid_kernel::launch::<WgpuRuntime>(
                &client,
                brush_cube::calc_cube_count_1d(num_pixels, crate::kernels::unproject::WG_SIZE),
                CubeDim::new_1d(crate::kernels::unproject::WG_SIZE),
                out_t.clone().into_tensor_arg(),
                img_w,
                img_h,
                self.fx,
                self.fy,
                self.cx,
                self.cy,
                self.camera_model,
            );

            h.register_float_tensor::<MainBackendBase>(&out.id, out_t);
        }
    }

    let fusion_client = burn_fusion::get_client::<MainBackendBase>(device);
    let out_ir = TensorIr::uninit(
        fusion_client.create_empty_handle(),
        Shape::new([h, w, 3]),
        DType::F32,
    );
    let stream = StreamId::current();
    let desc = CustomOpIr::new("unproject_ray_grid", &[], &[out_ir]);
    let op = Op {
        desc: desc.clone(),
        device: device.clone(),
        h,
        w,
        fx: pinhole.fx,
        fy: pinhole.fy,
        cx: pinhole.cx,
        cy: pinhole.cy,
        camera_model,
    };
    let [out] = fusion_client
        .register(stream, OperationIr::Custom(desc), op)
        .outputs();
    wrap_wgpu_float(out)
}

impl SplatOps for Fusion<MainBackendBase> {
    async fn render(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        options: crate::gaussian_splats::RenderOptions,
    ) -> RenderOutput<Self> {
        let client = transforms.client.clone();

        // Resolve fusion inputs to MainBackendBase tensors. This
        // drains any pending fusion operations into a concrete buffer.
        let base_transforms = client
            .clone()
            .resolve_tensor_float::<MainBackendBase>(transforms);
        let base_sh_coeffs = client
            .clone()
            .resolve_tensor_float::<MainBackendBase>(sh_coeffs);
        let base_raw_opac = client
            .clone()
            .resolve_tensor_float::<MainBackendBase>(raw_opacities);

        // Run the full pipeline on MainBackendBase.
        let out = MainBackendBase::render(
            camera,
            img_size,
            base_transforms,
            base_sh_coeffs,
            base_raw_opac,
            options,
        )
        .await;

        // Bind precomputed outputs back into the fusion stream.
        #[derive(Debug)]
        struct BindOp {
            desc: CustomOpIr,
            out_img: FloatTensor<MainBackendBase>,
            visible: FloatTensor<MainBackendBase>,
            max_radius: FloatTensor<MainBackendBase>,
            projected_splats: FloatTensor<MainBackendBase>,
            projected_geo: FloatTensor<MainBackendBase>,
            tile_offsets: IntTensor<MainBackendBase>,
            compact_gid_from_isect: IntTensor<MainBackendBase>,
            global_from_compact_gid: IntTensor<MainBackendBase>,
        }

        impl Operation<FusionCubeRuntime<WgpuRuntime>> for BindOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>,
            ) {
                let (_, outputs) = self.desc.as_fixed::<0, 8>();
                let [
                    out_img,
                    visible,
                    max_radius,
                    projected_splats,
                    projected_geo,
                    tile_offsets,
                    compact_gid_from_isect,
                    global_from_compact_gid,
                ] = outputs;

                h.register_float_tensor::<MainBackendBase>(&out_img.id, self.out_img.clone());
                h.register_float_tensor::<MainBackendBase>(&visible.id, self.visible.clone());
                h.register_float_tensor::<MainBackendBase>(&max_radius.id, self.max_radius.clone());
                h.register_float_tensor::<MainBackendBase>(
                    &projected_splats.id,
                    self.projected_splats.clone(),
                );
                h.register_float_tensor::<MainBackendBase>(
                    &projected_geo.id,
                    self.projected_geo.clone(),
                );
                h.register_int_tensor::<MainBackendBase>(
                    &tile_offsets.id,
                    self.tile_offsets.clone(),
                );
                h.register_int_tensor::<MainBackendBase>(
                    &compact_gid_from_isect.id,
                    self.compact_gid_from_isect.clone(),
                );
                h.register_int_tensor::<MainBackendBase>(
                    &global_from_compact_gid.id,
                    self.global_from_compact_gid.clone(),
                );
            }
        }

        let out_img_ir = TensorIr::uninit(
            client.create_empty_handle(),
            out.out_img.shape(),
            DType::F32,
        );
        let visible_ir = TensorIr::uninit(
            client.create_empty_handle(),
            out.aux.visible.shape(),
            DType::F32,
        );
        let max_radius_ir = TensorIr::uninit(
            client.create_empty_handle(),
            out.aux.max_radius.shape(),
            DType::F32,
        );
        let projected_splats_ir = TensorIr::uninit(
            client.create_empty_handle(),
            out.projected_splats.shape(),
            DType::F32,
        );
        let projected_geo_ir = TensorIr::uninit(
            client.create_empty_handle(),
            out.projected_geo.shape(),
            DType::F32,
        );
        let tile_offsets_ir = TensorIr::uninit(
            client.create_empty_handle(),
            out.aux.tile_offsets.shape(),
            DType::U32,
        );
        let compact_gid_from_isect_ir = TensorIr::uninit(
            client.create_empty_handle(),
            out.compact_gid_from_isect.shape(),
            DType::U32,
        );
        let global_from_compact_gid_ir = TensorIr::uninit(
            client.create_empty_handle(),
            out.global_from_compact_gid.shape(),
            DType::U32,
        );

        let stream = StreamId::current();
        let desc = CustomOpIr::new(
            "render_bind",
            &[],
            &[
                out_img_ir,
                visible_ir,
                max_radius_ir,
                projected_splats_ir,
                projected_geo_ir,
                tile_offsets_ir,
                compact_gid_from_isect_ir,
                global_from_compact_gid_ir,
            ],
        );
        let op = BindOp {
            desc: desc.clone(),
            out_img: out.out_img,
            visible: out.aux.visible,
            max_radius: out.aux.max_radius,
            projected_splats: out.projected_splats,
            projected_geo: out.projected_geo,
            tile_offsets: out.aux.tile_offsets,
            compact_gid_from_isect: out.compact_gid_from_isect,
            global_from_compact_gid: out.global_from_compact_gid,
        };

        let outputs = client
            .register(stream, OperationIr::Custom(desc), op)
            .outputs();

        let [
            out_img,
            visible,
            max_radius,
            projected_splats,
            projected_geo,
            tile_offsets,
            compact_gid_from_isect,
            global_from_compact_gid,
        ] = outputs;

        RenderOutput {
            out_img,
            aux: RenderAuxInner {
                num_visible: out.aux.num_visible,
                num_intersections: out.aux.num_intersections,
                visible,
                max_radius,
                tile_offsets,
                img_size: out.aux.img_size,
            },
            projected_splats,
            projected_geo,
            compact_gid_from_isect,
            project_uniforms: out.project_uniforms,
            global_from_compact_gid,
        }
    }
}
