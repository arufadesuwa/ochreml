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

pub fn write_bool(buf: &mut Vec<u8>, v: bool) {
    buf.push(u8::from(v));
}

pub fn write_usize(buf: &mut Vec<u8>, v: usize) {
    buf.extend_from_slice(&(v as u64).to_le_bytes());
}

pub fn write_i64(buf: &mut Vec<u8>, v: i64) {
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
