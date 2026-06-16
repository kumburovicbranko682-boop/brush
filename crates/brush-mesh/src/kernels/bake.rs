//! Depth-buffer based view selection for the texture bake.
//!
//! Instead of integrating opacity along rays (a near-render-cost pass per
//! view), visibility is answered shadow-map style: rasterize the extracted
//! mesh's depth per view at reduced resolution once, then each face tests
//! itself against any view with one buffer fetch. All per-view camera data
//! lives in plain tensors so a single launch can loop over every view.
//!
//! Cameras are treated as pinhole here (the bake pre-renders are pinhole);
//! the final texel sampling kernel stays camera-model aware.

use brush_render::kernels::types::Vec3A;
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

/// Floats per view in the packed camera tensor: 12 world-to-cam rows
/// (3x4, row-major) + fx, fy, cx, cy at the depth-buffer resolution.
pub const VIEW_LANES: u32 = 16;

/// Depth buffers raster at 1/4 of the view resolution.
pub const DEPTH_DOWNSCALE: u32 = 4;

/// Workgroup size for the 1D bake kernels.
pub const BAKE_WG: u32 = 256;

const NEAR_PLANE: f32 = 0.2;
/// Visibility slack: the mesh surface and the rastered depth disagree by
/// up to a few cm (low-res raster + flat-triangle depth interpolation).
const DEPTH_EPS_REL: f32 = 0.03;
const DEPTH_EPS_ABS: f32 = 0.05;
/// Skip degenerate-projection faces larger than this many depth-buffer
/// pixels on a side; they'd serialize the raster and only matter up close.
const MAX_RASTER_SIDE: u32 = 128;

#[cube]
fn cam_from_world(view_data: &Tensor<f32>, base: u32, p: Vec3A) -> Vec3A {
    let b = base as usize;
    let x = view_data[b] * p.x()
        + view_data[b + 1] * p.y()
        + view_data[b + 2] * p.z()
        + view_data[b + 3];
    let y = view_data[b + 4] * p.x()
        + view_data[b + 5] * p.y()
        + view_data[b + 6] * p.z()
        + view_data[b + 7];
    let z = view_data[b + 8] * p.x()
        + view_data[b + 9] * p.y()
        + view_data[b + 10] * p.z()
        + view_data[b + 11];
    Vec3A::new(x, y, z)
}

/// Rasterize one view's mesh depth: one thread per face, scatter
/// `atomicMin` over the covered depth-buffer pixels. Depth is camera-space
/// z, stored as f32 bits (monotonic for positive floats).
#[cube(launch)]
pub fn raster_depth_kernel(
    verts: &Tensor<f32>,
    faces: &Tensor<u32>,
    view_data: &Tensor<f32>,
    depth: &mut Tensor<Atomic<u32>>,
    view_index: u32,
    depth_offset: u32,
    qw: u32,
    qh: u32,
    n_faces: u32,
) {
    let fid = ABSOLUTE_POS as u32;
    if fid >= n_faces {
        terminate!();
    }
    let base = view_index * VIEW_LANES;
    let fx = view_data[(base + 12u32) as usize];
    let fy = view_data[(base + 13u32) as usize];
    let cx = view_data[(base + 14u32) as usize];
    let cy = view_data[(base + 15u32) as usize];

    let mut sx = Array::<f32>::new(3usize);
    let mut sy = Array::<f32>::new(3usize);
    let mut sz = Array::<f32>::new(3usize);
    let mut k = 0u32;
    let mut behind = false;
    while k < 3u32 {
        let vi = faces[(fid * 3u32 + k) as usize];
        let vb = (vi * 3u32) as usize;
        let pc = cam_from_world(
            view_data,
            base,
            Vec3A::new(verts[vb], verts[vb + 1], verts[vb + 2]),
        );
        if pc.z() <= NEAR_PLANE {
            behind = true;
        }
        sz[k as usize] = pc.z();
        let inv_z = 1.0f32 / max(pc.z(), NEAR_PLANE);
        sx[k as usize] = pc.x() * inv_z * fx + cx;
        sy[k as usize] = pc.y() * inv_z * fy + cy;
        k += 1u32;
    }
    // A face straddling the near plane has no stable projection; skip it
    // (visibility tests fall back to "visible", erring permissive).
    if behind {
        terminate!();
    }

    let min_x = min(min(sx[0], sx[1]), sx[2]);
    let max_x = max(max(sx[0], sx[1]), sx[2]);
    let min_y = min(min(sy[0], sy[1]), sy[2]);
    let max_y = max(max(sy[0], sy[1]), sy[2]);
    if max_x < 0.0f32 || max_y < 0.0f32 || min_x >= qw as f32 || min_y >= qh as f32 {
        terminate!();
    }
    let x0 = max(min_x, 0.0f32) as u32;
    let y0 = max(min_y, 0.0f32) as u32;
    let x1 = min(max_x as u32 + 1u32, qw);
    let y1 = min(max_y as u32 + 1u32, qh);
    if x1 - x0 > MAX_RASTER_SIDE || y1 - y0 > MAX_RASTER_SIDE {
        terminate!();
    }

    // Signed-area edge functions; sign-normalized so either winding works.
    let ax = sx[1] - sx[0];
    let ay = sy[1] - sy[0];
    let bx = sx[2] - sx[0];
    let by = sy[2] - sy[0];
    let area = ax * by - ay * bx;
    if area == 0.0f32 {
        terminate!();
    }
    let inv_area = 1.0f32 / area;

    let mut py = y0;
    while py < y1 {
        let mut px = x0;
        while px < x1 {
            let qx = px as f32 + 0.5f32 - sx[0];
            let qy = py as f32 + 0.5f32 - sy[0];
            let w2 = (ax * qy - ay * qx) * inv_area;
            let w1 = (qx * by - qy * bx) * inv_area;
            let w0 = 1.0f32 - w1 - w2;
            if w0 >= 0.0f32 && w1 >= 0.0f32 && w2 >= 0.0f32 {
                let z = w0 * sz[0] + w1 * sz[1] + w2 * sz[2];
                let didx = (depth_offset + py * qw + px) as usize;
                Atomic::fetch_min(&depth[didx], u32::reinterpret(z));
            }
            px += 1u32;
        }
        py += 1u32;
    }
}

/// Pick the best view per face: one thread per face loops every view,
/// depth-tests the face center and scores visible views by
/// `|cos(normal, view dir)| / z^2` (frontal and close wins). `keep_mask`
/// restricts the choice to a subset (1 = allowed); pass all-ones for the
/// unrestricted first round. Writes the view index + 1 as f32, or 0 when
/// no view passes.
#[cube(launch)]
pub fn select_best_view_kernel(
    centers: &Tensor<f32>,
    normals: &Tensor<f32>,
    view_data: &Tensor<f32>,
    depth_offsets: &Tensor<u32>,
    depth: &Tensor<u32>,
    keep_mask: &Tensor<u32>,
    best_view: &mut Tensor<f32>,
    n_faces: u32,
    n_views: u32,
    crop_margin: f32,
) {
    let fid = ABSOLUTE_POS as u32;
    if fid >= n_faces {
        terminate!();
    }
    let cb = (fid * 3u32) as usize;
    let px = centers[cb];
    let py = centers[cb + 1];
    let pz = centers[cb + 2];
    let nx = normals[cb];
    let ny = normals[cb + 1];
    let nz = normals[cb + 2];

    // Plain-literal inits only: expression initializers (like `-1.0f32`)
    // const-fold into immutable vars this cubecl rejects mutating. `best`
    // stores view index + 1, 0.0 = no visible view.
    let mut best_score = 0.0f32;
    let mut best = 0.0f32;
    let mut v = 0u32;
    while v < n_views {
        if keep_mask[v as usize] != 0u32 {
            let base = v * VIEW_LANES;
            let pc = cam_from_world(view_data, base, Vec3A::new(px, py, pz));
            let z = pc.z();
            if z > NEAR_PLANE {
                let fx = view_data[(base + 12u32) as usize];
                let fy = view_data[(base + 13u32) as usize];
                let cx = view_data[(base + 14u32) as usize];
                let cy = view_data[(base + 15u32) as usize];
                let ob = (v * 3u32) as usize;
                let off = depth_offsets[ob];
                let qw = depth_offsets[ob + 1];
                let qh = depth_offsets[ob + 2];
                let inv_z = 1.0f32 / z;
                let sx = pc.x() * inv_z * fx + cx;
                let sy = pc.y() * inv_z * fy + cy;
                if sx >= crop_margin * qw as f32
                    && sx < (1.0f32 - crop_margin) * qw as f32
                    && sy >= crop_margin * qh as f32
                    && sy < (1.0f32 - crop_margin) * qh as f32
                {
                    let sxi = sx as u32;
                    let syi = sy as u32;
                    let d = f32::reinterpret(depth[(off + syi * qw + sxi) as usize]);
                    if z <= d * (1.0f32 + DEPTH_EPS_REL) + DEPTH_EPS_ABS {
                        // View direction at the face = -ray direction; the
                        // mesh winding is arbitrary so take |cos|.
                        // cam-space normal z component vs ray (x,y,z)/|.|:
                        // do it in world space via the w2c rotation rows
                        // applied to the normal (no translation).
                        let b = base as usize;
                        let ncx = view_data[b] * nx + view_data[b + 1] * ny + view_data[b + 2] * nz;
                        let ncy =
                            view_data[b + 4] * nx + view_data[b + 5] * ny + view_data[b + 6] * nz;
                        let ncz =
                            view_data[b + 8] * nx + view_data[b + 9] * ny + view_data[b + 10] * nz;
                        let rlen = f32::sqrt(pc.x() * pc.x() + pc.y() * pc.y() + z * z);
                        let cosv =
                            f32::abs((ncx * pc.x() + ncy * pc.y() + ncz * z) / max(rlen, 1e-6f32));
                        let score = cosv * inv_z * inv_z;
                        // Compound assigns: the cube macro rejects plain
                        // f32 reassignment to outer muts in while loops.
                        if score > best_score {
                            best += f32::cast_from(v) + 1.0f32 - best;
                            best_score += score - best_score;
                        }
                    }
                }
            }
        }
        v += 1u32;
    }
    best_view[fid as usize] = best;
}
/// Per-texel weighted color accumulation for one view: depth-tested
/// visibility, weight = |cos(normal, ray)|^4 / z^dist_power, bilinear
/// color from the packed render. Texel normals arrive snorm8-packed in
/// the w lane.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn blend_texel_colors_kernel(
    texels: &Tensor<f32>,
    view_data: &Tensor<f32>,
    rendered_image: &Tensor<f32>,
    depth: &Tensor<u32>,
    color_sum: &mut Tensor<f32>,
    weight_sum: &mut Tensor<f32>,
    view_index: u32,
    depth_offset: u32,
    qw: u32,
    qh: u32,
    img_w: u32,
    img_h: u32,
    n_texels: u32,
    dist_power: f32,
    crop_margin: f32,
) {
    let pid = ABSOLUTE_POS as u32;
    if pid >= n_texels {
        terminate!();
    }
    let tb = (pid * 4u32) as usize;
    let base = view_index * VIEW_LANES;
    let p = Vec3A::new(texels[tb], texels[tb + 1], texels[tb + 2]);
    let pc = cam_from_world(view_data, base, p);
    let z = pc.z();
    if z <= NEAR_PLANE {
        terminate!();
    }
    let fx = view_data[(base + 12u32) as usize];
    let fy = view_data[(base + 13u32) as usize];
    let cx = view_data[(base + 14u32) as usize];
    let cy = view_data[(base + 15u32) as usize];
    // Intrinsics in `view_data` are at depth-buffer resolution, exactly
    // full-res / DEPTH_DOWNSCALE (deriving the factor from `qw / img_w`
    // would drift for sizes that don't divide evenly).
    let inv_z = 1.0f32 / z;
    let ds = DEPTH_DOWNSCALE as f32;
    let ix = pc.x() * inv_z * fx * ds + cx * ds;
    let iy = pc.y() * inv_z * fy * ds + cy * ds;
    if !(ix >= crop_margin * img_w as f32
        && ix < (1.0f32 - crop_margin) * img_w as f32
        && iy >= crop_margin * img_h as f32
        && iy < (1.0f32 - crop_margin) * img_h as f32)
    {
        terminate!();
    }
    let qx = min((ix / ds) as u32, qw - 1u32);
    let qy = min((iy / ds) as u32, qh - 1u32);
    let d = f32::reinterpret(depth[(depth_offset + qy * qw + qx) as usize]);
    if z > d * (1.0f32 + DEPTH_EPS_REL) + DEPTH_EPS_ABS {
        terminate!();
    }

    // Unpack the snorm8 normal and weight by |cos| against the ray.
    let npack = u32::reinterpret(texels[tb + 3]);
    let nx = f32::cast_from((npack & 0xFFu32) as i32 - 128i32) / 127.0f32;
    let ny = f32::cast_from(((npack >> 8u32) & 0xFFu32) as i32 - 128i32) / 127.0f32;
    let nz = f32::cast_from(((npack >> 16u32) & 0xFFu32) as i32 - 128i32) / 127.0f32;
    let b = base as usize;
    let ncx = view_data[b] * nx + view_data[b + 1] * ny + view_data[b + 2] * nz;
    let ncy = view_data[b + 4] * nx + view_data[b + 5] * ny + view_data[b + 6] * nz;
    let ncz = view_data[b + 8] * nx + view_data[b + 9] * ny + view_data[b + 10] * nz;
    let rlen = f32::sqrt(pc.x() * pc.x() + pc.y() * pc.y() + z * z);
    let cosv = f32::abs((ncx * pc.x() + ncy * pc.y() + ncz * z) / max(rlen, 1e-6f32));
    let cos2 = cosv * cosv;
    // Blend weight: cos^4 angular falloff x (1/z)^dist_power distance. Not
    // squared (was cos^8 x (1/z)^2*power), so texels average more smoothly
    // across nearby views instead of snapping to a single best view.
    let w = cos2 * cos2 * f32::powf(inv_z, dist_power);
    if w <= 0.0f32 {
        terminate!();
    }

    // Bilinear over the packed RGBA8 image, pixel centers at +0.5.
    let xf = max(min(ix - 0.5f32, (img_w - 1u32) as f32), 0.0f32);
    let yf = max(min(iy - 0.5f32, (img_h - 1u32) as f32), 0.0f32);
    let x0 = xf as u32;
    let y0 = yf as u32;
    let x1 = min(x0 + 1u32, img_w - 1u32);
    let y1 = min(y0 + 1u32, img_h - 1u32);
    let fxw = xf - x0 as f32;
    let fyw = yf - y0 as f32;
    let p00 = u32::reinterpret(rendered_image[(y0 * img_w + x0) as usize]);
    let p10 = u32::reinterpret(rendered_image[(y0 * img_w + x1) as usize]);
    let p01 = u32::reinterpret(rendered_image[(y1 * img_w + x0) as usize]);
    let p11 = u32::reinterpret(rendered_image[(y1 * img_w + x1) as usize]);
    let mut sh = 0u32;
    let cb = (pid * 3u32) as usize;
    let mut c = 0usize;
    while sh < 24u32 {
        let v00 = f32::cast_from((p00 >> sh) & 0xFFu32);
        let v10 = f32::cast_from((p10 >> sh) & 0xFFu32);
        let v01 = f32::cast_from((p01 >> sh) & 0xFFu32);
        let v11 = f32::cast_from((p11 >> sh) & 0xFFu32);
        let top = v00 * (1.0f32 - fxw) + v10 * fxw;
        let bot = v01 * (1.0f32 - fxw) + v11 * fxw;
        color_sum[cb + c] += w * (top * (1.0f32 - fyw) + bot * fyw);
        sh += 8u32;
        c += 1usize;
    }
    weight_sum[pid as usize] += w;
}

#[cfg(test)]
mod tests {
    use super::*;
    use brush_cube::{calc_cube_count_1d, create_tensor_from_slice};
    use burn::tensor::DType;
    use burn_cubecl::cubecl::CubeDim;
    use burn_cubecl::cubecl::Runtime;
    use burn_wgpu::WgpuRuntime;

    /// Launch each bake kernel once on tiny data: catches cubecl expansion
    /// panics (which fire lazily at first launch, on the device thread).
    #[tokio::test]
    async fn bake_kernels_launch() {
        let device = brush_cube::test_helpers::test_device().await;
        let client = WgpuRuntime::client(&device);

        // One view, 8x8 depth buffer, identity-ish camera at the origin
        // looking +z; a single triangle 2m ahead.
        let qw = 8u32;
        let qh = 8u32;
        #[rustfmt::skip]
        let view_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            4.0, 4.0, 4.0, 4.0,
        ];
        let verts: Vec<f32> = vec![-0.5, -0.5, 2.0, 0.5, -0.5, 2.0, 0.0, 0.5, 2.0];
        let faces: Vec<u32> = vec![0, 1, 2];
        let centers: Vec<f32> = vec![0.0, -0.16, 2.0];
        let normals: Vec<f32> = vec![0.0, 0.0, 1.0];

        let view_data_t = create_tensor_from_slice(&view_data, &device, DType::F32);
        let verts_t = create_tensor_from_slice(&verts, &device, DType::F32);
        let faces_t = create_tensor_from_slice(&faces, &device, DType::U32);
        let depth_t = create_tensor_from_slice(
            &vec![f32::INFINITY.to_bits(); (qw * qh) as usize],
            &device,
            DType::U32,
        );
        raster_depth_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(1, BAKE_WG),
            CubeDim::new_1d(BAKE_WG),
            verts_t.into_tensor_arg(),
            faces_t.into_tensor_arg(),
            view_data_t.clone().into_tensor_arg(),
            depth_t.clone().into_tensor_arg(),
            0u32,
            0u32,
            qw,
            qh,
            1u32,
        );

        let offsets_t = create_tensor_from_slice(&[0u32, qw, qh], &device, DType::U32);
        let keep_t = create_tensor_from_slice(&[1u32], &device, DType::U32);
        // Sentinel init: device-thread expansion panics leave outputs
        // untouched, so an uninit buffer can fake a pass.
        let best_t = create_tensor_from_slice(&[-7.0f32], &device, DType::F32);
        select_best_view_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(1, BAKE_WG),
            CubeDim::new_1d(BAKE_WG),
            create_tensor_from_slice(&centers, &device, DType::F32).into_tensor_arg(),
            create_tensor_from_slice(&normals, &device, DType::F32).into_tensor_arg(),
            view_data_t.into_tensor_arg(),
            offsets_t.into_tensor_arg(),
            depth_t.into_tensor_arg(),
            keep_t.into_tensor_arg(),
            best_t.clone().into_tensor_arg(),
            1u32,
            1u32,
            0.0f32,
        );

        let best = client.read_one(best_t.handle).expect("read best");
        let best: &[f32] = bytemuck::cast_slice(&best);
        assert_eq!(
            best[0], 1.0,
            "triangle 2m ahead must pick view 0 (+1 encoding)"
        );
    }
}
