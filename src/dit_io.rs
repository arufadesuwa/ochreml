// OchreML Binary Serialization Helpers for .dit Persistence (*≧ω≦*)
// Shared cursor reader/writer for DIT\x01 model and preprocessor formats.

use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;

pub const DIT_MAGIC: &[u8; 4] = b"DIT\x01";

pub fn validate_dit_path(path: &str) -> OchreResult<()> {
    if !path.trim().to_lowercase().ends_with(".dit") {
        return Err(OchreError::InvalidModelExtension(path.to_string()));
    }
    Ok(())
}

pub fn write_u8(buf: &mut Vec<u8>, v: u8) {
    buf.push(v);
}

pub fn write_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn write_bool(buf: &mut Vec<u8>, v: bool) {
    buf.push(u8::from(v));
}

pub fn write_usize(buf: &mut Vec<u8>, v: usize) {
    buf.extend_from_slice(&(v as u64).to_le_bytes());
}

pub fn write_i64(buf: &mut Vec<u8>, v: i64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn write_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn write_f64(buf: &mut Vec<u8>, v: f64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn write_bytes(buf: &mut Vec<u8>, bytes: &[u8]) {
    write_usize(buf, bytes.len());
    buf.extend_from_slice(bytes);
}

pub fn write_str(buf: &mut Vec<u8>, s: &str) {
    write_usize(buf, s.len());
    buf.extend_from_slice(s.as_bytes());
}

pub fn write_opt_str(buf: &mut Vec<u8>, s: &Option<String>) {
    match s {
        Some(val) => {
            write_u8(buf, 1);
            write_str(buf, val);
        }
        None => write_u8(buf, 0),
    }
}

pub fn write_opt_usize(buf: &mut Vec<u8>, v: Option<usize>) {
    match v {
        Some(val) => {
            write_u8(buf, 1);
            write_usize(buf, val);
        }
        None => write_u8(buf, 0),
    }
}

pub fn write_opt_u64(buf: &mut Vec<u8>, v: Option<u64>) {
    match v {
        Some(val) => {
            write_u8(buf, 1);
            buf.extend_from_slice(&val.to_le_bytes());
        }
        None => write_u8(buf, 0),
    }
}

pub fn write_opt_f64(buf: &mut Vec<u8>, v: Option<f64>) {
    match v {
        Some(val) => {
            write_u8(buf, 1);
            write_f64(buf, val);
        }
        None => write_u8(buf, 0),
    }
}

pub fn write_f32_vec(buf: &mut Vec<u8>, v: &[f32]) {
    write_usize(buf, v.len());
    for &x in v {
        write_f32(buf, x);
    }
}

pub fn write_u16_vec(buf: &mut Vec<u8>, v: &[u16]) {
    write_usize(buf, v.len());
    for &x in v {
        write_u16(buf, x);
    }
}

pub fn write_f64_vec(buf: &mut Vec<u8>, v: &[f64]) {
    write_usize(buf, v.len());
    for &x in v {
        write_f64(buf, x);
    }
}

pub fn write_usize_vec(buf: &mut Vec<u8>, v: &[usize]) {
    write_usize(buf, v.len());
    for &x in v {
        write_usize(buf, x);
    }
}

pub fn write_opt_f64_vec(buf: &mut Vec<u8>, v: &Option<Vec<f64>>) {
    match v {
        Some(vec) => {
            write_u8(buf, 1);
            write_f64_vec(buf, vec);
        }
        None => write_u8(buf, 0),
    }
}

pub fn write_opt_usize_vec(buf: &mut Vec<u8>, v: &Option<Vec<usize>>) {
    match v {
        Some(vec) => {
            write_u8(buf, 1);
            write_usize_vec(buf, vec);
        }
        None => write_u8(buf, 0),
    }
}

pub fn write_str_vec(buf: &mut Vec<u8>, v: &[String]) {
    write_usize(buf, v.len());
    for s in v {
        write_str(buf, s);
    }
}

pub fn write_matrix(buf: &mut Vec<u8>, mat: &Matrix) {
    write_usize(buf, mat.rows);
    write_usize(buf, mat.cols);
    write_f64_vec(buf, &mat.data);
}

pub fn write_opt_matrix(buf: &mut Vec<u8>, m: &Option<Matrix>) {
    match m {
        Some(mat) => {
            write_u8(buf, 1);
            write_matrix(buf, mat);
        }
        None => write_u8(buf, 0),
    }
}

pub fn write_header(tag: u8) -> Vec<u8> {
    let mut buf = Vec::with_capacity(64);
    buf.extend_from_slice(DIT_MAGIC);
    write_u8(&mut buf, tag);
    buf
}

pub struct ByteReader<'a> {
    pub data: &'a [u8],
    pub pos: usize,
}

impl<'a> ByteReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn read_exact(&mut self, len: usize) -> OchreResult<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or(OchreError::CorruptedDitFile)?;
        if end > self.data.len() {
            return Err(OchreError::CorruptedDitFile);
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    pub fn read_u8(&mut self) -> OchreResult<u8> {
        Ok(self.read_exact(1)?[0])
    }

    pub fn read_u16(&mut self) -> OchreResult<u16> {
        let b = self.read_exact(2)?;
        let mut arr = [0u8; 2];
        arr.copy_from_slice(b);
        Ok(u16::from_le_bytes(arr))
    }

    pub fn read_bool(&mut self) -> OchreResult<bool> {
        Ok(self.read_u8()? != 0)
    }

    pub fn read_usize(&mut self) -> OchreResult<usize> {
        let b = self.read_exact(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(b);
        Ok(u64::from_le_bytes(arr) as usize)
    }

    pub fn read_i64(&mut self) -> OchreResult<i64> {
        let b = self.read_exact(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(b);
        Ok(i64::from_le_bytes(arr))
    }

    pub fn read_f32(&mut self) -> OchreResult<f32> {
        let b = self.read_exact(4)?;
        let mut arr = [0u8; 4];
        arr.copy_from_slice(b);
        Ok(f32::from_le_bytes(arr))
    }

    pub fn read_f64(&mut self) -> OchreResult<f64> {
        let b = self.read_exact(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(b);
        Ok(f64::from_le_bytes(arr))
    }

    pub fn read_bytes(&mut self) -> OchreResult<Vec<u8>> {
        let len = self.read_usize()?;
        Ok(self.read_exact(len)?.to_vec())
    }

    pub fn read_str(&mut self) -> OchreResult<String> {
        let len = self.read_usize()?;
        let b = self.read_exact(len)?;
        String::from_utf8(b.to_vec()).map_err(|_| OchreError::InvalidUtf8InDit)
    }

    pub fn read_opt_str(&mut self) -> OchreResult<Option<String>> {
        if self.read_bool()? {
            Ok(Some(self.read_str()?))
        } else {
            Ok(None)
        }
    }

    pub fn read_opt_usize(&mut self) -> OchreResult<Option<usize>> {
        if self.read_bool()? {
            Ok(Some(self.read_usize()?))
        } else {
            Ok(None)
        }
    }

    pub fn read_opt_u64(&mut self) -> OchreResult<Option<u64>> {
        if self.read_bool()? {
            let b = self.read_exact(8)?;
            let mut arr = [0u8; 8];
            arr.copy_from_slice(b);
            Ok(Some(u64::from_le_bytes(arr)))
        } else {
            Ok(None)
        }
    }

    pub fn read_opt_f64(&mut self) -> OchreResult<Option<f64>> {
        if self.read_bool()? {
            Ok(Some(self.read_f64()?))
        } else {
            Ok(None)
        }
    }

    pub fn read_f32_vec(&mut self) -> OchreResult<Vec<f32>> {
        let len = self.read_usize()?;
        let byte_len = len.checked_mul(4).ok_or(OchreError::CorruptedDitFile)?;
        let bytes = self.read_exact(byte_len)?;
        let (chunks, _) = bytes.as_chunks::<4>();
        let mut vec = Vec::with_capacity(len);
        for &chunk in chunks {
            vec.push(f32::from_le_bytes(chunk));
        }
        Ok(vec)
    }

    pub fn read_u16_vec(&mut self) -> OchreResult<Vec<u16>> {
        let len = self.read_usize()?;
        let byte_len = len.checked_mul(2).ok_or(OchreError::CorruptedDitFile)?;
        let bytes = self.read_exact(byte_len)?;
        let (chunks, _) = bytes.as_chunks::<2>();
        let mut vec = Vec::with_capacity(len);
        for &chunk in chunks {
            vec.push(u16::from_le_bytes(chunk));
        }
        Ok(vec)
    }

    pub fn read_f64_vec(&mut self) -> OchreResult<Vec<f64>> {
        let len = self.read_usize()?;
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            vec.push(self.read_f64()?);
        }
        Ok(vec)
    }

    pub fn read_usize_vec(&mut self) -> OchreResult<Vec<usize>> {
        let len = self.read_usize()?;
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            vec.push(self.read_usize()?);
        }
        Ok(vec)
    }

    pub fn read_opt_f64_vec(&mut self) -> OchreResult<Option<Vec<f64>>> {
        if self.read_bool()? {
            Ok(Some(self.read_f64_vec()?))
        } else {
            Ok(None)
        }
    }

    pub fn read_opt_usize_vec(&mut self) -> OchreResult<Option<Vec<usize>>> {
        if self.read_bool()? {
            Ok(Some(self.read_usize_vec()?))
        } else {
            Ok(None)
        }
    }

    pub fn read_str_vec(&mut self) -> OchreResult<Vec<String>> {
        let len = self.read_usize()?;
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            vec.push(self.read_str()?);
        }
        Ok(vec)
    }

    pub fn read_matrix(&mut self) -> OchreResult<Matrix> {
        let rows = self.read_usize()?;
        let cols = self.read_usize()?;
        let data = self.read_f64_vec()?;
        Matrix::from_vec(rows, cols, data)
    }

    pub fn read_opt_matrix(&mut self) -> OchreResult<Option<Matrix>> {
        if self.read_bool()? {
            Ok(Some(self.read_matrix()?))
        } else {
            Ok(None)
        }
    }
}

pub fn check_magic_and_tag(reader: &mut ByteReader<'_>, expected_tag: u8) -> OchreResult<()> {
    let magic = reader.read_exact(4)?;
    if magic != DIT_MAGIC {
        return Err(OchreError::InvalidDitHeader);
    }
    let tag = reader.read_u8()?;
    if tag != expected_tag {
        return Err(OchreError::DitTagMismatch {
            expected: expected_tag,
            actual: tag,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dit_io_f32_and_u16_roundtrip() {
        let mut buf = write_header(100);
        write_f32(&mut buf, -3.25);
        write_f32_vec(&mut buf, &[0.0, 1.5, -2.75, 42.125]);
        write_u16_vec(&mut buf, &[0x3c00, 0x3f80, 0xffff]);

        let mut r = ByteReader::new(&buf);
        check_magic_and_tag(&mut r, 100).unwrap();
        assert!((r.read_f32().unwrap() + 3.25).abs() < 1e-6);
        assert_eq!(r.read_f32_vec().unwrap(), vec![0.0, 1.5, -2.75, 42.125]);
        assert_eq!(r.read_u16_vec().unwrap(), vec![0x3c00, 0x3f80, 0xffff]);
    }
}
