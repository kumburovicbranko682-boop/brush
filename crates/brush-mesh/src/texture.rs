//! UV-atlased color texture: decouples color resolution from vertex density.
//!
//! The charter is deliberately simple: faces cluster into charts by BFS over
//! shared edges among faces with similar normals, each chart is planar-
//! projected onto its mean-normal plane, and chart rectangles are skyline-packed
//! at uniform world-space texel density. Real parameterizers (xatlas, uvgen)
//! were tried and ran minutes on extraction-sized meshes; we don't need
//! their distortion guarantees because texels are baked at their true 3D
//! surface position, so chart distortion only modulates sample density.

use glam::Vec3;
use rustc_hash::FxHashMap;

use crate::Mesh;

/// Baked color atlas plus per-vertex UVs (aligned with the atlased mesh's
/// vertices; charting splits vertices along chart seams).
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    pub uvs: Vec<[f32; 2]>,
}

fn face_normal(mesh: &Mesh, f: usize) -> Vec3 {
    let [a, b, c] = mesh.faces[f];
    let (a, b, c) = (
        mesh.vertices[a as usize],
        mesh.vertices[b as usize],
        mesh.vertices[c as usize],
    );
    (b - a).cross(c - a).normalize_or_zero()
}

/// Chart + pack the mesh's UVs into an `atlas_size` square atlas, with
/// per-chart texel density driven by `face_importance` (observed view
/// resolution in pixels per world unit; 0 = unseen). Returns
/// the re-indexed mesh (vertices duplicated per chart, colors carried over)
/// and per-vertex UVs in `[0, 1]`.
pub fn atlas_mesh(
    mesh: &Mesh,
    atlas_size: u32,
    face_importance: &[f32],
) -> Option<(Mesh, Vec<[f32; 2]>, Vec<u32>)> {
    let n_faces = mesh.faces.len();
    if n_faces == 0 {
        return None;
    }

    // Face adjacency over shared edges.
    let adj = face_adjacency(&mesh.faces);

    // BFS face clustering: grow charts while a candidate's normal stays
    // within ~60 degrees of the chart's *running mean* normal (follows
    // gentle curvature) and ~75 degrees of the seed (bounds the projection
    // skew). Capped so packing rectangles stay sane.
    const MEAN_COS: f32 = 0.5;
    const SEED_COS: f32 = 0.25;
    const MAX_CHART_FACES: usize = 16384;
    let normals: Vec<Vec3> = (0..n_faces).map(|f| face_normal(mesh, f)).collect();
    let mut chart_of = vec![u32::MAX; n_faces];
    let mut charts: Vec<Vec<u32>> = Vec::new();
    let mut queue = std::collections::VecDeque::new();
    for seed in 0..n_faces {
        if chart_of[seed] != u32::MAX {
            continue;
        }
        let chart_id = charts.len() as u32;
        let seed_n = normals[seed];
        let mut acc_n = seed_n;
        let mut faces = vec![seed as u32];
        chart_of[seed] = chart_id;
        queue.clear();
        queue.push_back(seed);
        while let Some(f) = queue.pop_front() {
            if faces.len() >= MAX_CHART_FACES {
                break;
            }
            let mean_n = acc_n.normalize_or_zero();
            for &nb in &adj[f] {
                if nb < 0 {
                    continue;
                }
                let nb = nb as usize;
                if chart_of[nb] == u32::MAX
                    && normals[nb].dot(mean_n) > MEAN_COS
                    && normals[nb].dot(seed_n) > SEED_COS
                {
                    chart_of[nb] = chart_id;
                    acc_n += normals[nb];
                    faces.push(nb as u32);
                    queue.push_back(nb);
                }
            }
        }
        charts.push(faces);
    }

    // Merge speck charts into whichever neighbouring chart they share the
    // most edges with; tiny charts are pure packing overhead, and on dense
    // meshes their seam-split vertices lock the later UV-preserving
    // simplification far above its target. Threshold scales with the mesh.
    let speck_faces: usize = (n_faces / 20_000).max(16);
    for _ in 0..2 {
        let sizes: Vec<usize> = charts.iter().map(Vec::len).collect();
        for (ci, faces) in charts.iter().enumerate() {
            if faces.len() >= speck_faces || faces.is_empty() {
                continue;
            }
            let mut votes: FxHashMap<u32, u32> = FxHashMap::default();
            for &f in faces {
                for &nb in &adj[f as usize] {
                    if nb >= 0 {
                        let oc = chart_of[nb as usize];
                        if oc != ci as u32 {
                            *votes.entry(oc).or_default() += 1;
                        }
                    }
                }
            }
            if let Some((&target, _)) = votes
                .iter()
                .max_by_key(|&(&c, &n)| (n, std::cmp::Reverse(sizes[c as usize])))
            {
                for &f in faces {
                    chart_of[f as usize] = target;
                }
            }
        }
        // Rebuild chart face lists from the assignment.
        let mut rebuilt: Vec<Vec<u32>> = vec![Vec::new(); charts.len()];
        for f in 0..n_faces {
            rebuilt[chart_of[f] as usize].push(f as u32);
        }
        charts = rebuilt;
    }
    charts.retain(|c| !c.is_empty());

    // Planar-project each chart onto its mean-normal plane (an arbitrary
    // orthonormal basis in that plane): unlike dominant-axis projection,
    // texel density stays uniform for diagonally oriented surfaces.
    // Charts taller than wide are rotated 90 degrees for the packer.
    struct ChartUv {
        faces: Vec<u32>,
        // Per-face per-corner 2D coords, world units, bbox-relative.
        coords: Vec<[[f32; 2]; 3]>,
        size: [f32; 2],
    }
    let projected: Vec<ChartUv> = charts
        .into_iter()
        .map(|faces| {
            let mut mean_n = Vec3::ZERO;
            for &f in &faces {
                mean_n += normals[f as usize];
            }
            let n = mean_n.normalize_or_zero();
            let n = if n == Vec3::ZERO { Vec3::Z } else { n };
            let (tangent, bitangent) = n.any_orthonormal_pair();
            let mut coords = Vec::with_capacity(faces.len());
            for &f in &faces {
                let mut tri = [[0.0f32; 2]; 3];
                for (k, &vi) in mesh.faces[f as usize].iter().enumerate() {
                    let p = mesh.vertices[vi as usize];
                    tri[k] = [p.dot(tangent), p.dot(bitangent)];
                }
                coords.push(tri);
            }
            // The basis above has arbitrary in-plane rotation; fit the
            // orientation that minimizes the chart's bbox area (sampled
            // angles), so rectangular surfaces pack as tight rectangles.
            let mut best = (f32::INFINITY, 0.0f32, 1.0f32);
            for step in 0..32 {
                let ang = step as f32 * (std::f32::consts::FRAC_PI_2 / 32.0);
                let (sin, cos) = ang.sin_cos();
                let mut lo = [f32::INFINITY; 2];
                let mut hi = [f32::NEG_INFINITY; 2];
                for tri in &coords {
                    for uv in tri {
                        let r = [uv[0] * cos - uv[1] * sin, uv[0] * sin + uv[1] * cos];
                        lo = [lo[0].min(r[0]), lo[1].min(r[1])];
                        hi = [hi[0].max(r[0]), hi[1].max(r[1])];
                    }
                }
                let area = (hi[0] - lo[0]) * (hi[1] - lo[1]);
                if area < best.0 {
                    best = (area, sin, cos);
                }
            }
            let (_, sin, cos) = best;
            let mut lo = [f32::INFINITY; 2];
            let mut hi = [f32::NEG_INFINITY; 2];
            for tri in &mut coords {
                for uv in tri {
                    *uv = [uv[0] * cos - uv[1] * sin, uv[0] * sin + uv[1] * cos];
                    lo = [lo[0].min(uv[0]), lo[1].min(uv[1])];
                    hi = [hi[0].max(uv[0]), hi[1].max(uv[1])];
                }
            }
            let rotate = (hi[1] - lo[1]) > (hi[0] - lo[0]);
            for tri in &mut coords {
                for uv in tri {
                    *uv = if rotate {
                        [uv[1] - lo[1], uv[0] - lo[0]]
                    } else {
                        [uv[0] - lo[0], uv[1] - lo[1]]
                    };
                }
            }
            let size = if rotate {
                [hi[1] - lo[1], hi[0] - lo[0]]
            } else {
                [hi[0] - lo[0], hi[1] - lo[1]]
            };
            ChartUv {
                faces,
                coords,
                size,
            }
        })
        .collect();

    // Per-chart texel density follows observed view resolution: charts
    // photographed up close get proportionally more texels than distant
    // or unseen ones (relative scale clamped to 4x spread), with a global
    // factor fitting everything to ~85% of the atlas.
    let mean_imp = {
        let vals: Vec<f32> = face_importance
            .iter()
            .copied()
            .filter(|&v| v > 0.0)
            .collect();
        if vals.is_empty() {
            1.0
        } else {
            vals.iter().sum::<f32>() / vals.len() as f32
        }
    };
    let chart_rel: Vec<f32> = projected
        .iter()
        .map(|c| {
            let vals: Vec<f32> = c
                .faces
                .iter()
                .map(|&f| face_importance.get(f as usize).copied().unwrap_or(0.0))
                .filter(|&v| v > 0.0)
                .collect();
            if vals.is_empty() {
                0.5
            } else {
                let m = vals.iter().sum::<f32>() / vals.len() as f32;
                (m / mean_imp).clamp(0.5, 2.0)
            }
        })
        .collect();
    let total_area: f64 = projected
        .iter()
        .zip(&chart_rel)
        .map(|(c, &r)| (c.size[0] as f64) * (c.size[1] as f64) * (r as f64) * (r as f64))
        .sum();
    if total_area <= 0.0 {
        return None;
    }
    let px = atlas_size as f64;
    let texels_per_unit = (px * px * 0.85 / total_area).sqrt() as f32;
    const GUTTER: f32 = 2.0;

    // Skyline packing, tallest charts first: each chart lands at the x
    // whose skyline window is lowest (sliding-window max over per-column
    // heights), which fills the gaps shelf packing leaves.
    let mut order: Vec<usize> = (0..projected.len()).collect();
    order.sort_by(|&a, &b| {
        (projected[b].size[1])
            .partial_cmp(&projected[a].size[1])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut origin: Vec<[f32; 2]> = vec![[0.0; 2]; projected.len()];
    let mut scale = texels_per_unit;
    let width = atlas_size as usize;
    'pack: loop {
        let mut heights = vec![GUTTER; width];
        let mut ok = true;
        for &ci in &order {
            let cs = scale * chart_rel[ci];
            let w = (projected[ci].size[0] * cs + GUTTER).ceil() as usize;
            let h = projected[ci].size[1] * cs + GUTTER;
            if w >= width {
                ok = false;
                break;
            }
            // Sliding-window max of `heights` with window `w` via a
            // monotonic deque, tracking the argmin over window positions.
            let mut deque: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
            let mut best_x = usize::MAX;
            let mut best_y = f32::INFINITY;
            for x in 0..width {
                while let Some(&b) = deque.back()
                    && heights[b] <= heights[x]
                {
                    deque.pop_back();
                }
                deque.push_back(x);
                if x + 1 >= w {
                    let start = x + 1 - w;
                    while let Some(&f) = deque.front()
                        && f < start
                    {
                        deque.pop_front();
                    }
                    let y = heights[*deque.front().expect("nonempty window")];
                    if y < best_y {
                        best_y = y;
                        best_x = start;
                    }
                }
            }
            if best_x == usize::MAX || best_y + h > atlas_size as f32 {
                ok = false;
                break;
            }
            origin[ci] = [best_x as f32 + GUTTER, best_y];
            for height in &mut heights[best_x..best_x + w] {
                *height = best_y + h;
            }
        }
        if !ok {
            // Didn't fit: shrink density and repack.
            scale *= 0.97;
            continue 'pack;
        }
        break;
    }
    log::info!(
        "Atlas: {} charts, {:.0} texels/unit (started {:.0})",
        projected.len(),
        scale,
        texels_per_unit
    );

    // Emit the re-indexed mesh: vertices are duplicated per chart (a vertex
    // shared by two charts gets two UVs).
    let inv = 1.0 / atlas_size as f32;
    let mut vertices = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut faces_out = Vec::with_capacity(n_faces);
    let mut face_src = Vec::with_capacity(n_faces);
    let mut vert_map: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    for (ci, chart) in projected.iter().enumerate() {
        let cs = scale * chart_rel[ci];
        for (fi, &f) in chart.faces.iter().enumerate() {
            let mut new_face = [0u32; 3];
            for k in 0..3 {
                let vi = mesh.faces[f as usize][k];
                let new_vi = *vert_map.entry((ci as u32, vi)).or_insert_with(|| {
                    vertices.push(mesh.vertices[vi as usize]);
                    let c = chart.coords[fi][k];
                    uvs.push([
                        (origin[ci][0] + c[0] * cs) * inv,
                        (origin[ci][1] + c[1] * cs) * inv,
                    ]);
                    (vertices.len() - 1) as u32
                });
                new_face[k] = new_vi;
            }
            faces_out.push(new_face);
            face_src.push(f);
        }
    }
    Some((
        Mesh {
            vertices,
            faces: faces_out,
        },
        uvs,
        face_src,
    ))
}

/// Edge-shared face adjacency: `adj[f][k]` is the face across edge `k`
/// of face `f`, or -1 on a boundary.
pub(crate) fn face_adjacency(faces: &[[u32; 3]]) -> Vec<[i64; 3]> {
    let mut edge_owner: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    let mut adj: Vec<[i64; 3]> = vec![[-1; 3]; faces.len()];
    for (fi, f) in faces.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            let key = (a.min(b), a.max(b));
            match edge_owner.get(&key) {
                Some(&other) => {
                    adj[fi][k] = other as i64;
                    let of = &faces[other as usize];
                    for ok in 0..3 {
                        let (oa, ob) = (of[ok], of[(ok + 1) % 3]);
                        if (oa.min(ob), oa.max(ob)) == key {
                            adj[other as usize][ok] = fi as i64;
                        }
                    }
                }
                None => {
                    edge_owner.insert(key, fi as u32);
                }
            }
        }
    }
    adj
}

/// Walk every texel covered by `face` in UV space, invoking `emit` with
/// the texel index and clamped barycentric weights. Texel centers within
/// half a texel of the triangle count as covered so edge texels are owned
/// by someone.
pub fn for_each_face_texel(
    mesh: &Mesh,
    uvs: &[[f32; 2]],
    atlas_px: u32,
    face: usize,
    mut emit: impl FnMut(u32, f32, f32, f32),
) {
    let f = mesh.faces[face];
    let (ia, ib, ic) = (f[0] as usize, f[1] as usize, f[2] as usize);
    let res = atlas_px as f32;
    let (ua, ub, uc) = (uvs[ia], uvs[ib], uvs[ic]);
    let a = [ua[0] * res, ua[1] * res];
    let b = [ub[0] * res, ub[1] * res];
    let c = [uc[0] * res, uc[1] * res];

    let min_x = a[0].min(b[0]).min(c[0]).floor().max(0.0) as u32;
    let max_x = (a[0].max(b[0]).max(c[0]).ceil() as u32).min(atlas_px - 1);
    let min_y = a[1].min(b[1]).min(c[1]).floor().max(0.0) as u32;
    let max_y = (a[1].max(b[1]).max(c[1]).ceil() as u32).min(atlas_px - 1);

    let det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
    if det.abs() < 1e-12 {
        return;
    }
    let inv_det = 1.0 / det;
    // Half-texel tolerance in barycentric units, per edge: 0.5px divided
    // by how many pixels one barycentric unit spans for that weight.
    let len_ca = (c[0] - a[0]).hypot(c[1] - a[1]);
    let len_ba = (b[0] - a[0]).hypot(b[1] - a[1]);
    let tol_b = 0.5 * len_ca / det.abs().max(1e-12);
    let tol_c = 0.5 * len_ba / det.abs().max(1e-12);

    for ty in min_y..=max_y {
        for tx in min_x..=max_x {
            let px = tx as f32 + 0.5;
            let py = ty as f32 + 0.5;
            let mut wb = ((px - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (py - a[1])) * inv_det;
            let mut wc = ((b[0] - a[0]) * (py - a[1]) - (px - a[0]) * (b[1] - a[1])) * inv_det;
            if wb < -tol_b || wc < -tol_c || wb + wc > 1.0 + tol_b.max(tol_c) {
                continue;
            }
            wb = wb.clamp(0.0, 1.0);
            wc = wc.clamp(0.0, 1.0);
            let sum = wb + wc;
            if sum > 1.0 {
                wb /= sum;
                wc /= sum;
            }
            emit(ty * atlas_px + tx, 1.0 - wb - wc, wb, wc);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_and_rasterize_unit_quad() {
        let mesh = Mesh {
            vertices: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
            ],
            faces: vec![[0, 1, 2], [1, 3, 2]],
        };
        let (atlased, uvs, _src) = atlas_mesh(&mesh, 128, &[1.0; 2]).expect("atlas");
        assert_eq!(atlased.vertices.len(), uvs.len(), "uv per vertex");
        let mut pos = Vec::new();
        let mut idx = Vec::new();
        for f in 0..atlased.faces.len() {
            for_each_face_texel(&atlased, &uvs, 64, f, |i, wa, wb, wc| {
                let fc = atlased.faces[f];
                pos.push(
                    atlased.vertices[fc[0] as usize] * wa
                        + atlased.vertices[fc[1] as usize] * wb
                        + atlased.vertices[fc[2] as usize] * wc,
                );
                idx.push(i);
            });
        }
        assert!(pos.len() > 500, "quad covers a decent part of a 64px atlas");
        assert!(idx.iter().all(|&i| i < 64 * 64), "indices in atlas");
        for p in &pos {
            assert!(p.z.abs() < 1e-5, "samples on the quad plane");
        }
    }
}
