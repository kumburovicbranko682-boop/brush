//! Guards that the GOF geometry render undistorts the per-pixel camera ray
//! in-kernel. A tilted plane is rendered with a high-distortion
//! `RadialTangential8` camera and with the same camera swapped to pinhole
//! (`with_pinhole`). The GOF median depth at an EDGE pixel must differ
//! non-trivially between the two (the undistort moves the ray, so it hits the
//! tilted plane at a different depth), while a CENTER pixel must match (no
//! radial distortion on-axis). If the kernel still used the pinhole ray, edge
//! depths would match too.

use brush_render::camera::Camera;
use brush_render::gaussian_splats::{RenderOptions, SplatRenderMode, Splats};
use brush_render::geo;
use brush_render::kernels::camera_model::CameraModel;
use brush_render::kernels::camera_model::radial_tangential_8::RadialTangential8Params;
use brush_render_bwd::render_splats;
use burn::tensor::s;
use glam::{Quat, UVec2, Vec3, vec3};

/// A dense, opaque, flattened plane through the origin with unit normal
/// `n_world`, spanning enough area to fill the camera view at z ~ 3.
fn tilted_plane_splats(n_world: Vec3, device: &burn::tensor::Device) -> Splats {
    let n = n_world.normalize();
    let quat = Quat::from_rotation_arc(Vec3::Z, n);
    let u = if n.x.abs() < 0.9 {
        n.cross(Vec3::X).normalize()
    } else {
        n.cross(Vec3::Y).normalize()
    };
    let v = n.cross(u).normalize();

    let mut means = Vec::new();
    let mut rots = Vec::new();
    let mut log_scales = Vec::new();
    let mut sh_dc = Vec::new();
    let mut raw_opac = Vec::new();

    let half = 46;
    let step = 0.08_f32;
    let big = (step * 1.8).ln();
    let thin = (1.0e-3_f32).ln();
    for iu in -half..=half {
        for iv in -half..=half {
            let p = u * (iu as f32 * step) + v * (iv as f32 * step);
            means.extend_from_slice(&[p.x, p.y, p.z]);
            rots.extend_from_slice(&[quat.w, quat.x, quat.y, quat.z]);
            log_scales.extend_from_slice(&[big, big, thin]);
            sh_dc.extend_from_slice(&[0.5, 0.5, 0.5]);
            raw_opac.push(8.0);
        }
    }

    Splats::from_raw(
        means,
        rots,
        log_scales,
        sh_dc,
        raw_opac,
        SplatRenderMode::Default,
        device,
    )
}

/// Strong barrel distortion (only the numerator coeffs set, so the radial
/// factor moves the ray a lot off-axis). The principal point is placed on the
/// exact center pixel of a `size`x`size` image (`cx = size/2 + 0.5`) so that
/// pixel sits at zero radius (truly on-axis, distortion-free).
fn distorted_cam(size: u32) -> Camera {
    let p = RadialTangential8Params {
        k1: -0.5,
        k2: 0.25,
        k3: 0.0,
        k4: 0.0,
        k5: 0.0,
        k6: 0.0,
        p1: 0.0,
        p2: 0.0,
    };
    let c = (size / 2) as f32 + 0.5;
    Camera::new(
        vec3(0.0, 0.0, -3.0),
        Quat::IDENTITY,
        1.4,
        1.4,
        glam::vec2(c / size as f32, c / size as f32),
        CameraModel::RadialTangential8(p),
    )
}

/// Render geometry with `cam` and return the GOF median depth + coverage as
/// flat `[H*W]` vectors.
async fn render_depth(
    cam: &Camera,
    img_size: UVec2,
    device: &burn::tensor::Device,
) -> (Vec<f32>, Vec<f32>, usize, usize) {
    // Tilted plane so the intersection depth depends on the ray direction.
    let splats = tilted_plane_splats(vec3(0.8, 0.0, -1.0), device);
    let diff = render_splats(splats, cam, img_size, RenderOptions::geometry(), None).await;
    let geo = diff.geo.expect("geo channels");
    let alpha = diff.img.slice(s![.., .., 3..4]);
    let [h, w, _] = geo.dims();
    let depth = geo::rendered_depth(geo);
    let depth_v: Vec<f32> = depth.into_data_async().await.unwrap().into_vec().unwrap();
    let alpha_v: Vec<f32> = alpha.into_data_async().await.unwrap().into_vec().unwrap();
    (depth_v, alpha_v, h, w)
}

#[tokio::test]
async fn gof_depth_uses_undistorted_ray() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let size = 96u32;
    let img_size = UVec2::new(size, size);

    let cam_dist = distorted_cam(size);
    let cam_pin = cam_dist.with_pinhole();

    let (d_dist, a_dist, h, w) = render_depth(&cam_dist, img_size, &device).await;
    let (d_pin, a_pin, _, _) = render_depth(&cam_pin, img_size, &device).await;

    let idx = |px: usize, py: usize| py * w + px;

    // Center pixel: no radial distortion on-axis, both renders must match.
    let cx = w / 2;
    let cy = h / 2;
    let ci = idx(cx, cy);
    assert!(
        a_dist[ci] > 0.8 && a_pin[ci] > 0.8,
        "center pixel not covered (a_dist={}, a_pin={})",
        a_dist[ci],
        a_pin[ci]
    );
    let center_diff = (d_dist[ci] - d_pin[ci]).abs();
    assert!(
        center_diff < 2.0e-3,
        "center depth should match on-axis (distorted={}, pinhole={}, diff={})",
        d_dist[ci],
        d_pin[ci],
        center_diff
    );

    // Edge pixel near the left border on the center row: large radius -> the
    // undistort moves the ray, so the tilted-plane intersection depth shifts.
    // Scan inward for the first column covered in both renders.
    let mut edge_diff = 0.0f32;
    let mut found = false;
    let mut edge_info = (0usize, 0.0f32, 0.0f32);
    for px in 2..w / 3 {
        let i = idx(px, cy);
        if a_dist[i] > 0.8 && a_pin[i] > 0.8 {
            edge_diff = (d_dist[i] - d_pin[i]).abs();
            edge_info = (px, d_dist[i], d_pin[i]);
            found = true;
            break;
        }
    }
    assert!(found, "no covered edge pixel found");
    println!(
        "center: dist={:.5} pin={:.5} diff={:.2e} | edge[x={}]: dist={:.5} pin={:.5} diff={:.2e}",
        d_dist[ci], d_pin[ci], center_diff, edge_info.0, edge_info.1, edge_info.2, edge_diff
    );
    assert!(
        edge_diff > 5.0e-3,
        "edge depth should differ between distorted and pinhole rays \
         (distorted={}, pinhole={}, diff={}); the in-kernel undistort is not exercised",
        edge_info.1,
        edge_info.2,
        edge_diff
    );
}
