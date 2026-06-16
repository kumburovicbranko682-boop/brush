//! Self-contained binary glTF (GLB) export: positions + UVs + indices,
//! one embedded PNG atlas, one unlit material.

use std::io::Write;
use std::path::Path;

use crate::Mesh;
use crate::texture::Texture;

fn pad4(v: &mut Vec<u8>, fill: u8) {
    while !v.len().is_multiple_of(4) {
        v.push(fill);
    }
}

/// Write `mesh` with its baked atlas as a single `.glb`. The material is
/// `KHR_materials_unlit` (the bake already contains the scene lighting).
pub fn write_glb(mesh: &Mesh, tex: &Texture, path: &Path) -> std::io::Result<()> {
    let mut png = Vec::new();
    image::RgbaImage::from_raw(tex.width, tex.height, tex.rgba.clone())
        .ok_or_else(|| std::io::Error::other("atlas buffer size"))?
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(std::io::Error::other)?;

    let mut bin: Vec<u8> = Vec::new();
    let idx_off = bin.len();
    for f in &mesh.faces {
        for &i in f {
            bin.extend_from_slice(&i.to_le_bytes());
        }
    }
    let idx_len = bin.len() - idx_off;
    let pos_off = bin.len();
    let mut pmin = [f32::INFINITY; 3];
    let mut pmax = [f32::NEG_INFINITY; 3];
    for v in &mesh.vertices {
        for (c, x) in [v.x, v.y, v.z].into_iter().enumerate() {
            pmin[c] = pmin[c].min(x);
            pmax[c] = pmax[c].max(x);
            bin.extend_from_slice(&x.to_le_bytes());
        }
    }
    let pos_len = bin.len() - pos_off;
    let uv_off = bin.len();
    for uv in &tex.uvs {
        bin.extend_from_slice(&uv[0].to_le_bytes());
        bin.extend_from_slice(&uv[1].to_le_bytes());
    }
    let uv_len = bin.len() - uv_off;
    let png_off = bin.len();
    bin.extend_from_slice(&png);
    let png_len = png.len();
    pad4(&mut bin, 0);

    let json = serde_json::json!({
        "asset": {"version": "2.0", "generator": "brush"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 1, "TEXCOORD_0": 2},
            "indices": 0,
            "material": 0
        }]}],
        "materials": [{
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0
            },
            "extensions": {"KHR_materials_unlit": {}},
            "doubleSided": true
        }],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": 3, "mimeType": "image/png"}],
        "buffers": [{"byteLength": bin.len()}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": idx_off, "byteLength": idx_len, "target": 34963},
            {"buffer": 0, "byteOffset": pos_off, "byteLength": pos_len, "target": 34962},
            {"buffer": 0, "byteOffset": uv_off, "byteLength": uv_len, "target": 34962},
            {"buffer": 0, "byteOffset": png_off, "byteLength": png_len}
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5125, "count": mesh.faces.len() * 3,
             "type": "SCALAR"},
            {"bufferView": 1, "componentType": 5126, "count": mesh.vertices.len(),
             "type": "VEC3", "min": pmin, "max": pmax},
            {"bufferView": 2, "componentType": 5126, "count": tex.uvs.len(),
             "type": "VEC2"}
        ]
    });
    let mut json_bytes = serde_json::to_vec(&json).map_err(std::io::Error::other)?;
    pad4(&mut json_bytes, b' ');

    let total = 12 + 8 + json_bytes.len() + 8 + bin.len();
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    w.write_all(b"glTF")?;
    w.write_all(&2u32.to_le_bytes())?;
    w.write_all(&(total as u32).to_le_bytes())?;
    w.write_all(&(json_bytes.len() as u32).to_le_bytes())?;
    w.write_all(b"JSON")?;
    w.write_all(&json_bytes)?;
    w.write_all(&(bin.len() as u32).to_le_bytes())?;
    w.write_all(b"BIN\0")?;
    w.write_all(&bin)?;
    log::info!(
        "Wrote {} ({} verts, {} faces, {}x{} atlas)",
        path.display(),
        mesh.vertices.len(),
        mesh.faces.len(),
        tex.width,
        tex.height
    );
    Ok(())
}

/// Read a GLB written by [`write_glb`] back into a mesh, UVs and atlas.
pub fn read_glb(path: &Path) -> std::io::Result<(Mesh, Texture)> {
    let data = std::fs::read(path)?;
    let bad = || std::io::Error::other("malformed glb");
    if data.len() < 20 || &data[0..4] != b"glTF" {
        return Err(bad());
    }
    let json_len = u32::from_le_bytes(data[12..16].try_into().expect("len")) as usize;
    let json: serde_json::Value =
        serde_json::from_slice(&data[20..20 + json_len]).map_err(std::io::Error::other)?;
    let bin_start = 20 + json_len + 8;
    let bin = &data[bin_start..];

    let views = json["bufferViews"].as_array().ok_or_else(bad)?;
    let view = |i: usize| -> std::io::Result<&[u8]> {
        let v = views.get(i).ok_or_else(bad)?;
        let off = v["byteOffset"].as_u64().unwrap_or(0) as usize;
        let len = v["byteLength"].as_u64().ok_or_else(bad)? as usize;
        bin.get(off..off + len).ok_or_else(bad)
    };
    let idx: Vec<u32> = view(0)?
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().expect("u32")))
        .collect();
    let pos: Vec<f32> = view(1)?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("f32")))
        .collect();
    let uv: Vec<f32> = view(2)?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("f32")))
        .collect();
    let img = image::load_from_memory(view(3)?)
        .map_err(std::io::Error::other)?
        .into_rgba8();

    Ok((
        Mesh {
            vertices: pos
                .chunks_exact(3)
                .map(|c| Vec3::new(c[0], c[1], c[2]))
                .collect(),
            faces: idx.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
        },
        Texture {
            width: img.width(),
            height: img.height(),
            rgba: img.into_raw(),
            uvs: uv.chunks_exact(2).map(|c| [c[0], c[1]]).collect(),
        },
    ))
}

use glam::Vec3;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glb_roundtrip() {
        let mesh = Mesh {
            vertices: vec![
                Vec3::ZERO,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            faces: vec![[0, 1, 2]],
        };
        let tex = Texture {
            width: 2,
            height: 2,
            rgba: vec![255; 16],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        };
        let dir = std::env::temp_dir().join("brush_glb_test.glb");
        write_glb(&mesh, &tex, &dir).unwrap();
        let (back, btex) = read_glb(&dir).unwrap();
        assert_eq!(back.vertices, mesh.vertices);
        assert_eq!(back.faces, mesh.faces);
        assert_eq!(btex.uvs, tex.uvs);
        assert_eq!((btex.width, btex.height), (2, 2));
    }
}
