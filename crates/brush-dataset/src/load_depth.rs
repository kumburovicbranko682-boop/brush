use brush_vfs::BrushVfs;
use std::{io, path::PathBuf, sync::Arc};
use tokio::io::AsyncReadExt;

/// Loader for a per-view metric `LiDAR` depth + confidence sidecar.
#[derive(Clone, Debug)]
pub struct LoadDepth {
    vfs: Arc<BrushVfs>,
    path: PathBuf,
    /// Image size (w, h): raw depth carries no dimensions, so they are
    /// inferred as the factorization of the sample count closest to the
    /// image's aspect ratio.
    image_size: (u32, u32),
}

/// Decoded depth: row-major `z`-depth in metres + `ARKit` confidence.
#[derive(Clone, Debug)]
pub struct DepthData {
    pub width: usize,
    pub height: usize,
    pub depth: Vec<f32>,
    pub conf: Vec<u8>,
}

impl PartialEq for LoadDepth {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
    }
}

impl LoadDepth {
    pub fn new(vfs: Arc<BrushVfs>, path: PathBuf, image_size: (u32, u32)) -> Self {
        Self {
            vfs,
            path,
            image_size,
        }
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// Headerless raw f32 depth; dimensions inferred from the image aspect
    /// and confidence read from the `{stem}_confidence.bin` sibling (u8 per
    /// sample, or f32 cast down).
    pub async fn load(&self) -> io::Result<DepthData> {
        let mut bytes = vec![];
        self.vfs
            .reader_at_path(&self.path)
            .await?
            .read_to_end(&mut bytes)
            .await?;
        self.load_raw(&bytes).await
    }

    async fn load_raw(&self, bytes: &[u8]) -> io::Result<DepthData> {
        let bad = |m: String| io::Error::new(io::ErrorKind::InvalidData, m);
        if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
            return Err(bad(format!(
                "raw depth {}: size {} is not a multiple of 4",
                self.path.display(),
                bytes.len()
            )));
        }
        let n = bytes.len() / 4;
        let (width, height) = infer_dims(n, self.image_size).ok_or_else(|| {
            bad(format!(
                "raw depth {}: no {}x-aspect factorization of {n} samples",
                self.path.display(),
                self.image_size.0 as f32 / self.image_size.1 as f32
            ))
        })?;
        let depth = bytemuck::pod_collect_to_vec::<u8, f32>(bytes);
        // Invalid / no-return samples become +inf, the "no depth here" sentinel.
        let depth: Vec<f32> = depth
            .iter()
            .map(|&d| {
                if d.is_finite() && d > 0.0 {
                    d
                } else {
                    f32::INFINITY
                }
            })
            .collect();

        let stem = self
            .path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let conf_path = self.path.with_file_name(format!("{stem}_confidence.bin"));
        let conf = if let Ok(mut reader) = self.vfs.reader_at_path(&conf_path).await {
            let mut cbytes = vec![];
            reader.read_to_end(&mut cbytes).await?;
            if cbytes.len() == n {
                cbytes
            } else if cbytes.len() == n * 4 {
                bytemuck::pod_collect_to_vec::<u8, f32>(&cbytes)
                    .iter()
                    .map(|&c| c.clamp(0.0, 2.0) as u8)
                    .collect()
            } else {
                return Err(bad(format!(
                    "confidence {}: {} bytes for {n} depth samples",
                    conf_path.display(),
                    cbytes.len()
                )));
            }
        } else {
            log::warn!(
                "no confidence sidecar for {}; assuming full confidence",
                self.path.display()
            );
            vec![2u8; n]
        };
        Ok(DepthData {
            width,
            height,
            depth,
            conf,
        })
    }
}

/// Factor `n` into `(w, h)` with `w / h` nearest the image aspect: walks
/// divisors of `n` near the ideal height. `ARKit` depth (256x192 for 4:3
/// captures) resolves exactly.
fn infer_dims(n: usize, image_size: (u32, u32)) -> Option<(usize, usize)> {
    let aspect = image_size.0 as f64 / image_size.1 as f64;
    let ideal_h = ((n as f64) / aspect).sqrt();
    let mut best: Option<(usize, usize, f64)> = None;
    let lo = ((ideal_h * 0.5) as usize).max(1);
    let hi = (ideal_h * 2.0) as usize + 1;
    for h in lo..=hi {
        if n.is_multiple_of(h) {
            let w = n / h;
            let err = ((w as f64 / h as f64) / aspect).ln().abs();
            if best.is_none_or(|(_, _, e)| err < e) {
                best = Some((w, h, err));
            }
        }
    }
    // Reject factorizations that are nowhere near the image aspect (a
    // prime-sized file would otherwise come out as a 1-pixel strip).
    best.filter(|&(_, _, e)| e < 0.2).map(|(w, h, _)| (w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infers_arkit_dims() {
        // ARKit depth on a 4:3 capture.
        assert_eq!(infer_dims(256 * 192, (4032, 3024)), Some((256, 192)));
        // 16:9 capture.
        assert_eq!(infer_dims(256 * 144, (1920, 1080)), Some((256, 144)));
        // Prime sample count has no sane factorization.
        assert_eq!(infer_dims(49157, (4032, 3024)), None);
    }
}
