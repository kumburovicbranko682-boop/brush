//! GOF's `filter_mesh` step. The decision is per-*crossing* (= per mesh
//! vertex), not per-face: a crossing is kept iff the original Delaunay
//! edge it spans has length ≤ the sum of its two endpoint Gaussian
//! scales. The caller computes that flag from the original Delaunay
//! `(pts, scales)` (see [`crate::extract`]); by the time we get the
//! `Mesh`, vertex positions have been refined by the binary search and
//! the Delaunay edge length is no longer recoverable.
//!
//! Once the per-crossing mask is set, any face touching a dropped
//! crossing is dropped, matching GOF's `face_mask = mask[faces].all(axis=1)`.

use crate::Mesh;

/// Apply a per-vertex keep mask: any face whose any vertex is `false`
/// in `keep` is dropped, and the surviving vertices are compacted with
/// face indices remapped.
pub fn filter_mesh_with_keep(mesh: &Mesh, keep: &[bool]) -> Mesh {
    assert_eq!(keep.len(), mesh.vertices.len(), "keep mask is per-vertex");

    let mut faces = Vec::with_capacity(mesh.faces.len());
    for f in &mesh.faces {
        if keep[f[0] as usize] && keep[f[1] as usize] && keep[f[2] as usize] {
            faces.push(*f);
        }
    }

    let mut remap = vec![u32::MAX; mesh.vertices.len()];
    let mut vertices = Vec::new();
    for (i, &k) in keep.iter().enumerate() {
        if k {
            remap[i] = vertices.len() as u32;
            vertices.push(mesh.vertices[i]);
        }
    }
    for f in &mut faces {
        for v in f {
            *v = remap[*v as usize];
        }
    }
    Mesh { vertices, faces }
}

/// Drop connected components below `max(min_faces, min_frac * largest)`
/// faces. Isolated iso-crossing blobs (the "speckled halo" around real
/// geometry) are tens of faces; the relative term also removes larger
/// disconnected debris that is still dwarfed by the main subject.
pub fn filter_small_components(mesh: &Mesh, min_faces: usize, min_frac: f32) -> Mesh {
    if (min_faces <= 1 && min_frac <= 0.0) || mesh.faces.is_empty() {
        return mesh.clone();
    }

    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }

    let n = mesh.vertices.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    for f in &mesh.faces {
        let a = find(&mut parent, f[0]);
        let b = find(&mut parent, f[1]);
        parent[b as usize] = a;
        let c = find(&mut parent, f[2]);
        parent[c as usize] = a;
    }

    let mut comp_faces = vec![0u32; n];
    for f in &mesh.faces {
        comp_faces[find(&mut parent, f[0]) as usize] += 1;
    }
    let largest = comp_faces.iter().copied().max().unwrap_or(0) as usize;
    let threshold = min_faces.max((largest as f32 * min_frac) as usize);
    let keep: Vec<bool> = (0..n as u32)
        .map(|v| comp_faces[find(&mut parent, v) as usize] as usize >= threshold)
        .collect();

    let dropped = keep.iter().filter(|&&k| !k).count();
    if dropped > 0 {
        log::info!("Component filter: dropping {dropped} verts in components < {threshold} faces");
    }
    filter_mesh_with_keep(mesh, &keep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_components_are_dropped() {
        // A two-face quad plus one isolated triangle far away.
        let mesh = Mesh {
            vertices: vec![
                glam::Vec3::new(0.0, 0.0, 0.0),
                glam::Vec3::new(1.0, 0.0, 0.0),
                glam::Vec3::new(0.0, 1.0, 0.0),
                glam::Vec3::new(1.0, 1.0, 0.0),
                glam::Vec3::new(100.0, 0.0, 0.0),
                glam::Vec3::new(101.0, 0.0, 0.0),
                glam::Vec3::new(100.0, 1.0, 0.0),
            ],
            faces: vec![[0, 1, 2], [1, 3, 2], [4, 5, 6]],
        };
        let out = filter_small_components(&mesh, 2, 0.0);
        assert_eq!(out.faces.len(), 2);
        assert_eq!(out.vertices.len(), 4);
        // min_faces 1 keeps everything.
        let all = filter_small_components(&mesh, 1, 0.0);
        assert_eq!(all.faces.len(), 3);
    }

    #[test]
    fn drops_marked_vertices() {
        let mesh = Mesh {
            vertices: vec![
                glam::Vec3::new(0.0, 0.0, 0.0),
                glam::Vec3::new(0.1, 0.0, 0.0),
                glam::Vec3::new(0.0, 0.1, 0.0),
                glam::Vec3::new(100.0, 0.0, 0.0),
            ],
            faces: vec![[0, 1, 2], [0, 1, 3]],
        };
        let keep = vec![true, true, true, false];
        let out = filter_mesh_with_keep(&mesh, &keep);
        assert_eq!(out.faces.len(), 1);
        assert_eq!(out.vertices.len(), 3);
    }
}
