// OchreML Feature Extraction & Text Vectorization Engine (*≧ω≦*)
// Implements exact MurmurHash3_32, FeatureHasher, DictVectorizer,
// CountVectorizer, TfidfTransformer, and HashingVectorizer with Rayon + CSR output.

use crate::device::{DeviceType, parse_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f64, write_header, write_opt_f64_vec,
    write_opt_str, write_opt_usize, write_str, write_str_vec, write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};

pub const TAG_DICT_VECTORIZER: u8 = 28;
pub const TAG_FEATURE_HASHER: u8 = 29;
pub const TAG_COUNT_VECTORIZER: u8 = 30;
pub const TAG_TFIDF_TRANSFORMER: u8 = 31;
pub const TAG_TFIDF_VECTORIZER: u8 = 32;
pub const TAG_HASHING_VECTORIZER: u8 = 33;

fn resolve_loaded_device(
    saved_device: Option<String>,
    is_fitted: bool,
    device: Option<&str>,
) -> OchreResult<(DeviceType, Option<String>)> {
    let dev = parse_device(device.or(Some("auto")))?;
    let device_ = if saved_device.is_some() || is_fitted {
        Some(dev.to_string())
    } else {
        None
    };
    Ok((dev, device_))
}

/// Exact MurmurHash3_x86_32 matching `sklearn.utils.murmurhash.murmurhash3_bytes_s32(key, seed=0)`.
#[inline]
pub fn murmurhash3_bytes_s32(data: &[u8], seed: u32) -> i32 {
    const C1: u32 = 0xcc9e_2d51;
    const C2: u32 = 0x1b87_3593;
    let len = data.len();
    let nblocks = len / 4;
    let mut h1: u32 = seed;

    for i in 0..nblocks {
        let off = i * 4;
        let mut k1 = u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        k1 = k1.wrapping_mul(C1);
        k1 = k1.rotate_left(15);
        k1 = k1.wrapping_mul(C2);
        h1 ^= k1;
        h1 = h1.rotate_left(13);
        h1 = h1.wrapping_mul(5).wrapping_add(0xe654_6b64);
    }

    let tail = &data[nblocks * 4..];
    let mut k1: u32 = 0;
    if tail.len() >= 3 {
        k1 ^= (tail[2] as u32) << 16;
    }
    if tail.len() >= 2 {
        k1 ^= (tail[1] as u32) << 8;
    }
    if !tail.is_empty() {
        k1 ^= tail[0] as u32;
        k1 = k1.wrapping_mul(C1);
        k1 = k1.rotate_left(15);
        k1 = k1.wrapping_mul(C2);
        h1 ^= k1;
    }

    h1 ^= len as u32;
    h1 ^= h1 >> 16;
    h1 = h1.wrapping_mul(0x85eb_ca6b);
    h1 ^= h1 >> 13;
    h1 = h1.wrapping_mul(0xc2b2_ae35);
    h1 ^= h1 >> 16;
    h1 as i32
}

/// CSR matrix raw parts `(data, indices, indptr, n_rows, n_cols)`.
pub type CsrParts = (Vec<f64>, Vec<usize>, Vec<usize>, usize, usize);

fn hash_row_pairs(
    items: &[(String, f64)],
    n_features: usize,
    alternate_sign: bool,
) -> Vec<(usize, f64)> {
    let mut acc: HashMap<usize, f64> = HashMap::with_capacity(items.len());
    for (key, val) in items {
        let h = murmurhash3_bytes_s32(key.as_bytes(), 0);
        let col = if h == i32::MIN {
            2_147_483_648usize % n_features
        } else {
            (h.unsigned_abs() as usize) % n_features
        };
        let signed_val = if alternate_sign && h < 0 { -*val } else { *val };
        *acc.entry(col).or_insert(0.0) += signed_val;
    }
    let mut cols: Vec<(usize, f64)> = acc.into_iter().filter(|&(_, v)| v != 0.0).collect();
    cols.sort_by_key(|&(c, _)| c);
    cols
}

/// Implements feature hashing ("the hashing trick") via MurmurHash3 (*≧ω≦*)
#[derive(Debug, Clone)]
pub struct FeatureHasher {
    pub n_features: usize,
    pub input_type: String,
    pub alternate_sign: bool,
    pub device: DeviceType,
    pub device_: Option<String>,
}

impl FeatureHasher {
    pub fn new(
        n_features: usize,
        input_type: String,
        alternate_sign: bool,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if n_features == 0 {
            return Err(OchreError::InvalidParameter(
                "n_features must be > 0 ( >_< )".to_string(),
            ));
        }
        if !matches!(input_type.as_str(), "dict" | "pair" | "string") {
            return Err(OchreError::InvalidParameter(format!(
                "input_type must be 'dict', 'pair', or 'string', got '{}' ( >_< )",
                input_type
            )));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            n_features,
            input_type,
            alternate_sign,
            device: dev,
            device_: None,
        })
    }

    pub fn transform_pairs(&self, rows: &[Vec<(String, f64)>]) -> OchreResult<CsrParts> {
        let n_rows = rows.len();
        let n_feat = self.n_features;
        let alt = self.alternate_sign;

        let per_row: Vec<Vec<(usize, f64)>> = rows
            .par_iter()
            .map(|items| hash_row_pairs(items, n_feat, alt))
            .collect();

        let total_nnz: usize = per_row.iter().map(|r| r.len()).sum();
        let mut data = Vec::with_capacity(total_nnz);
        let mut indices = Vec::with_capacity(total_nnz);
        let mut indptr = Vec::with_capacity(n_rows + 1);
        indptr.push(0);
        for r in per_row {
            for (c, v) in r {
                indices.push(c);
                data.push(v);
            }
            indptr.push(data.len());
        }
        Ok((data, indices, indptr, n_rows, n_feat))
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_FEATURE_HASHER);
        write_usize(&mut buf, self.n_features);
        write_str(&mut buf, &self.input_type);
        write_bool(&mut buf, self.alternate_sign);
        write_opt_str(&mut buf, &self.device_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_FEATURE_HASHER)?;
        let n_features = r.read_usize()?;
        let input_type = r.read_str()?;
        let alternate_sign = r.read_bool()?;
        let saved_device = r.read_opt_str()?;
        let (dev, device_) = resolve_loaded_device(saved_device, false, device)?;
        Ok(Self {
            n_features,
            input_type,
            alternate_sign,
            device: dev,
            device_,
        })
    }
}

/// Transforms lists of feature-value mappings to vectors (＾▽＾)
#[derive(Debug, Clone)]
pub struct DictVectorizer {
    pub separator: String,
    pub sort: bool,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub feature_names_: Option<Vec<String>>,
    pub vocabulary_: Option<HashMap<String, usize>>,
}

impl DictVectorizer {
    pub fn new(separator: String, sort: bool, device: Option<String>) -> OchreResult<Self> {
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            separator,
            sort,
            device: dev,
            device_: None,
            feature_names_: None,
            vocabulary_: None,
        })
    }

    pub fn fit(&mut self, rows: &[Vec<(String, f64)>]) -> OchreResult<()> {
        let mut seen: HashSet<String> = HashSet::new();
        let mut names: Vec<String> = Vec::new();
        for row in rows {
            for (k, _) in row {
                if seen.insert(k.clone()) {
                    names.push(k.clone());
                }
            }
        }
        if self.sort {
            names.sort();
        }
        let mut vocab = HashMap::with_capacity(names.len());
        for (i, name) in names.iter().enumerate() {
            vocab.insert(name.clone(), i);
        }
        self.feature_names_ = Some(names);
        self.vocabulary_ = Some(vocab);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn transform(&self, rows: &[Vec<(String, f64)>]) -> OchreResult<CsrParts> {
        let vocab = self.vocabulary_.as_ref().ok_or(OchreError::NotFitted)?;
        let n_cols = self.feature_names_.as_ref().map_or(0, |f| f.len());
        let n_rows = rows.len();

        let per_row: Vec<Vec<(usize, f64)>> = rows
            .par_iter()
            .map(|row| {
                let mut acc: BTreeMap<usize, f64> = BTreeMap::new();
                for (k, v) in row {
                    if let Some(&idx) = vocab.get(k) {
                        *acc.entry(idx).or_insert(0.0) += *v;
                    }
                }
                acc.into_iter().filter(|&(_, v)| v != 0.0).collect()
            })
            .collect();

        let total_nnz: usize = per_row.iter().map(|r| r.len()).sum();
        let mut data = Vec::with_capacity(total_nnz);
        let mut indices = Vec::with_capacity(total_nnz);
        let mut indptr = Vec::with_capacity(n_rows + 1);
        indptr.push(0);
        for r in per_row {
            for (c, v) in r {
                indices.push(c);
                data.push(v);
            }
            indptr.push(data.len());
        }
        Ok((data, indices, indptr, n_rows, n_cols))
    }

    pub fn restrict(&mut self, support: &[bool]) -> OchreResult<()> {
        let names = self.feature_names_.as_ref().ok_or(OchreError::NotFitted)?;
        if support.len() != names.len() {
            return Err(OchreError::FeatureCountMismatch {
                features: support.len(),
                expected: names.len(),
            });
        }
        let new_names: Vec<String> = names
            .iter()
            .zip(support.iter())
            .filter(|&(_, &keep)| keep)
            .map(|(n, _)| n.clone())
            .collect();
        let mut new_vocab = HashMap::with_capacity(new_names.len());
        for (i, n) in new_names.iter().enumerate() {
            new_vocab.insert(n.clone(), i);
        }
        self.feature_names_ = Some(new_names);
        self.vocabulary_ = Some(new_vocab);
        Ok(())
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_DICT_VECTORIZER);
        write_str(&mut buf, &self.separator);
        write_bool(&mut buf, self.sort);
        write_opt_str(&mut buf, &self.device_);
        match &self.feature_names_ {
            Some(names) => {
                write_bool(&mut buf, true);
                write_str_vec(&mut buf, names);
            }
            None => write_bool(&mut buf, false),
        }
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_DICT_VECTORIZER)?;
        let separator = r.read_str()?;
        let sort = r.read_bool()?;
        let saved_device = r.read_opt_str()?;
        let feature_names_ = if r.read_bool()? {
            Some(r.read_str_vec()?)
        } else {
            None
        };
        let vocabulary_ = feature_names_.as_ref().map(|names| {
            let mut m = HashMap::with_capacity(names.len());
            for (i, n) in names.iter().enumerate() {
                m.insert(n.clone(), i);
            }
            m
        });
        let (dev, device_) = resolve_loaded_device(saved_device, feature_names_.is_some(), device)?;
        Ok(Self {
            separator,
            sort,
            device: dev,
            device_,
            feature_names_,
            vocabulary_,
        })
    }
}

/// Tokenize text using sklearn's default word pattern `(?u)\b\w\w+\b`.
pub fn tokenize_default_words(text: &str, lowercase: bool) -> Vec<String> {
    let lower_owned;
    let s: &str = if lowercase {
        lower_owned = text.to_lowercase();
        &lower_owned
    } else {
        text
    };
    let mut tokens = Vec::new();
    let mut start: Option<usize> = None;
    let mut char_count = 0usize;
    for (idx, ch) in s.char_indices() {
        if ch.is_alphanumeric() || ch == '_' {
            if start.is_none() {
                start = Some(idx);
                char_count = 0;
            }
            char_count += 1;
        } else if let Some(st) = start {
            if char_count >= 2 {
                tokens.push(s[st..idx].to_string());
            }
            start = None;
        }
    }
    if let Some(st) = start
        && char_count >= 2
    {
        tokens.push(s[st..].to_string());
    }
    tokens
}

/// Extract word n-grams from an already-tokenized sequence, skipping stop words.
pub fn extract_word_ngrams(
    tokens: &[String],
    min_n: usize,
    max_n: usize,
    stop_words: &HashSet<String>,
) -> Vec<String> {
    let filtered: Vec<&str> = tokens
        .iter()
        .filter(|t| !stop_words.contains(t.as_str()))
        .map(|t| t.as_str())
        .collect();
    let mut ngrams = Vec::new();
    let len = filtered.len();
    for n in min_n..=max_n {
        if n == 0 || n > len {
            continue;
        }
        if n == 1 {
            for &t in &filtered {
                ngrams.push(t.to_string());
            }
        } else {
            for i in 0..=len - n {
                ngrams.push(filtered[i..i + n].join(" "));
            }
        }
    }
    ngrams
}

/// Extract character n-grams (`analyzer='char'` or `'char_wb'`).
pub fn extract_char_ngrams(
    text: &str,
    lowercase: bool,
    min_n: usize,
    max_n: usize,
    word_boundaries: bool,
) -> Vec<String> {
    let norm = text.split_whitespace().collect::<Vec<&str>>().join(" ");
    let s = if lowercase { norm.to_lowercase() } else { norm };
    let mut out = Vec::new();
    if !word_boundaries {
        let chars: Vec<char> = s.chars().collect();
        for n in min_n..=max_n {
            if n == 0 || n > chars.len() {
                continue;
            }
            for i in 0..=chars.len() - n {
                out.push(chars[i..i + n].iter().collect());
            }
        }
    } else {
        for word in s.split_whitespace() {
            let mut chars = Vec::with_capacity(word.len() + 2);
            chars.push(' ');
            chars.extend(word.chars());
            chars.push(' ');
            for n in min_n..=max_n {
                if n == 0 || n > chars.len() {
                    continue;
                }
                for i in 0..=chars.len() - n {
                    out.push(chars[i..i + n].iter().collect());
                }
            }
        }
    }
    out
}

/// Convert a collection of text documents to a matrix of token counts (o´∀｀o)
#[derive(Debug, Clone)]
pub struct CountVectorizer {
    pub lowercase: bool,
    pub analyzer: String,
    pub ngram_range: (usize, usize),
    pub max_df: f64,
    pub min_df: f64,
    pub max_features: Option<usize>,
    pub binary: bool,
    pub stop_words: HashSet<String>,
    pub fixed_vocabulary_: bool,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub vocabulary_: Option<HashMap<String, usize>>,
    pub stop_words_: HashSet<String>,
}

impl CountVectorizer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lowercase: bool,
        analyzer: String,
        ngram_range: (usize, usize),
        max_df: f64,
        min_df: f64,
        max_features: Option<usize>,
        binary: bool,
        stop_words: HashSet<String>,
        fixed_vocab: Option<HashMap<String, usize>>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if ngram_range.0 == 0 || ngram_range.0 > ngram_range.1 {
            return Err(OchreError::InvalidParameter(format!(
                "Invalid ngram_range=({}, {}) ( >_< )",
                ngram_range.0, ngram_range.1
            )));
        }
        let dev = parse_device(device.as_deref())?;
        let fixed_vocabulary_ = fixed_vocab.is_some();
        Ok(Self {
            lowercase,
            analyzer,
            ngram_range,
            max_df,
            min_df,
            max_features,
            binary,
            stop_words,
            fixed_vocabulary_,
            device: dev,
            device_: None,
            vocabulary_: fixed_vocab,
            stop_words_: HashSet::new(),
        })
    }

    pub fn analyze_docs(&self, docs: &[String]) -> Vec<Vec<String>> {
        let (min_n, max_n) = self.ngram_range;
        let lowercase = self.lowercase;
        let stop = &self.stop_words;
        let analyzer = self.analyzer.as_str();

        docs.par_iter()
            .map(|doc| match analyzer {
                "char" => extract_char_ngrams(doc, lowercase, min_n, max_n, false),
                "char_wb" => extract_char_ngrams(doc, lowercase, min_n, max_n, true),
                _ => {
                    let toks = tokenize_default_words(doc, lowercase);
                    extract_word_ngrams(&toks, min_n, max_n, stop)
                }
            })
            .collect()
    }

    pub fn fit_from_tokenized(&mut self, docs_tokens: &[Vec<String>]) -> OchreResult<()> {
        if docs_tokens.is_empty() {
            return Err(OchreError::EmptyData);
        }
        self.device_ = Some(self.device.to_string());
        if self.fixed_vocabulary_ {
            return Ok(());
        }

        let n_docs = docs_tokens.len();
        let mut df: HashMap<String, usize> = HashMap::new();
        let mut tf_total: HashMap<String, usize> = HashMap::new();
        for toks in docs_tokens {
            let mut doc_seen: HashSet<&str> = HashSet::with_capacity(toks.len());
            for t in toks {
                *tf_total.entry(t.clone()).or_insert(0) += 1;
                if doc_seen.insert(t.as_str()) {
                    *df.entry(t.clone()).or_insert(0) += 1;
                }
            }
        }

        let min_doc_cnt = if self.min_df < 1.0 && self.min_df >= 0.0 {
            (self.min_df * n_docs as f64).ceil() as usize
        } else {
            self.min_df as usize
        };
        let max_doc_cnt = if self.max_df <= 1.0 {
            (self.max_df * n_docs as f64).floor() as usize
        } else {
            self.max_df as usize
        };

        if max_doc_cnt < min_doc_cnt {
            return Err(OchreError::InvalidParameter(
                "max_df corresponds to fewer documents than min_df ( >_< )".to_string(),
            ));
        }

        let mut kept: Vec<(String, usize)> = Vec::new();
        let mut removed = HashSet::new();
        for (term, count) in df {
            if count < min_doc_cnt || count > max_doc_cnt {
                removed.insert(term);
            } else {
                let total = tf_total.get(&term).copied().unwrap_or(0);
                kept.push((term, total));
            }
        }

        if let Some(max_f) = self.max_features
            && kept.len() > max_f
        {
            kept.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            for (t, _) in kept.drain(max_f..) {
                removed.insert(t);
            }
        }

        kept.sort_by(|a, b| a.0.cmp(&b.0));
        let mut vocab = HashMap::with_capacity(kept.len());
        for (idx, (term, _)) in kept.into_iter().enumerate() {
            vocab.insert(term, idx);
        }

        self.vocabulary_ = Some(vocab);
        self.stop_words_ = removed;
        Ok(())
    }

    pub fn transform_from_tokenized(&self, docs_tokens: &[Vec<String>]) -> OchreResult<CsrParts> {
        let vocab = self.vocabulary_.as_ref().ok_or(OchreError::NotFitted)?;
        let n_cols = vocab.len();
        let n_rows = docs_tokens.len();
        let binary = self.binary;

        let per_row: Vec<Vec<(usize, f64)>> = docs_tokens
            .par_iter()
            .map(|toks| {
                let mut counts: BTreeMap<usize, f64> = BTreeMap::new();
                for t in toks {
                    if let Some(&idx) = vocab.get(t) {
                        if binary {
                            counts.insert(idx, 1.0);
                        } else {
                            *counts.entry(idx).or_insert(0.0) += 1.0;
                        }
                    }
                }
                counts.into_iter().collect()
            })
            .collect();

        let total_nnz: usize = per_row.iter().map(|r| r.len()).sum();
        let mut data = Vec::with_capacity(total_nnz);
        let mut indices = Vec::with_capacity(total_nnz);
        let mut indptr = Vec::with_capacity(n_rows + 1);
        indptr.push(0);
        for r in per_row {
            for (c, v) in r {
                indices.push(c);
                data.push(v);
            }
            indptr.push(data.len());
        }
        Ok((data, indices, indptr, n_rows, n_cols))
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_COUNT_VECTORIZER);
        write_bool(&mut buf, self.lowercase);
        write_str(&mut buf, &self.analyzer);
        write_usize(&mut buf, self.ngram_range.0);
        write_usize(&mut buf, self.ngram_range.1);
        write_f64(&mut buf, self.max_df);
        write_f64(&mut buf, self.min_df);
        write_opt_usize(&mut buf, self.max_features);
        write_bool(&mut buf, self.binary);
        let sw: Vec<String> = self.stop_words.iter().cloned().collect();
        write_str_vec(&mut buf, &sw);
        write_bool(&mut buf, self.fixed_vocabulary_);
        write_opt_str(&mut buf, &self.device_);
        match &self.vocabulary_ {
            Some(v) => {
                write_bool(&mut buf, true);
                write_usize(&mut buf, v.len());
                for (k, &idx) in v {
                    write_str(&mut buf, k);
                    write_usize(&mut buf, idx);
                }
            }
            None => write_bool(&mut buf, false),
        }
        let sw_learned: Vec<String> = self.stop_words_.iter().cloned().collect();
        write_str_vec(&mut buf, &sw_learned);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_COUNT_VECTORIZER)?;
        let lowercase = r.read_bool()?;
        let analyzer = r.read_str()?;
        let n0 = r.read_usize()?;
        let n1 = r.read_usize()?;
        let max_df = r.read_f64()?;
        let min_df = r.read_f64()?;
        let max_features = r.read_opt_usize()?;
        let binary = r.read_bool()?;
        let stop_words: HashSet<String> = r.read_str_vec()?.into_iter().collect();
        let fixed_vocabulary_ = r.read_bool()?;
        let saved_device = r.read_opt_str()?;
        let vocabulary_ = if r.read_bool()? {
            let len = r.read_usize()?;
            let mut m = HashMap::with_capacity(len);
            for _ in 0..len {
                let k = r.read_str()?;
                let idx = r.read_usize()?;
                m.insert(k, idx);
            }
            Some(m)
        } else {
            None
        };
        let stop_words_: HashSet<String> = r.read_str_vec()?.into_iter().collect();
        let (dev, device_) = resolve_loaded_device(saved_device, vocabulary_.is_some(), device)?;
        Ok(Self {
            lowercase,
            analyzer,
            ngram_range: (n0, n1),
            max_df,
            min_df,
            max_features,
            binary,
            stop_words,
            fixed_vocabulary_,
            device: dev,
            device_,
            vocabulary_,
            stop_words_,
        })
    }
}

/// Normalize CSR rows in place (`l1` or `l2`).
fn normalize_csr_rows(data: &mut [f64], indptr: &[usize], norm: &str) {
    let n_rows = indptr.len().saturating_sub(1);
    for r in 0..n_rows {
        let start = indptr[r];
        let end = indptr[r + 1];
        let slice = &mut data[start..end];
        let n = match norm {
            "l1" => slice.iter().map(|v| v.abs()).sum::<f64>(),
            "l2" => slice.iter().map(|v| v * v).sum::<f64>().sqrt(),
            _ => 1.0,
        };
        if n > 0.0 {
            let inv = 1.0 / n;
            for v in slice.iter_mut() {
                *v *= inv;
            }
        }
    }
}

/// Transform a count matrix to a normalized TF or TF-IDF representation (＾▽＾)
#[derive(Debug, Clone)]
pub struct TfidfTransformer {
    pub norm: Option<String>,
    pub use_idf: bool,
    pub smooth_idf: bool,
    pub sublinear_tf: bool,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub idf_: Option<Vec<f64>>,
    pub n_features_in_: Option<usize>,
}

impl TfidfTransformer {
    pub fn new(
        norm: Option<String>,
        use_idf: bool,
        smooth_idf: bool,
        sublinear_tf: bool,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if let Some(ref n) = norm
            && n != "l1"
            && n != "l2"
        {
            return Err(OchreError::InvalidParameter(format!(
                "norm must be 'l1', 'l2', or None, got '{}' ( >_< )",
                n
            )));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            norm,
            use_idf,
            smooth_idf,
            sublinear_tf,
            device: dev,
            device_: None,
            idf_: None,
            n_features_in_: None,
        })
    }

    pub fn fit_csr(
        &mut self,
        data: &[f64],
        indices: &[usize],
        indptr: &[usize],
        n_rows: usize,
        n_cols: usize,
    ) -> OchreResult<()> {
        if n_rows == 0 || n_cols == 0 {
            return Err(OchreError::EmptyData);
        }
        self.n_features_in_ = Some(n_cols);
        self.device_ = Some(self.device.to_string());
        if !self.use_idf {
            self.idf_ = None;
            return Ok(());
        }

        let mut df = vec![0usize; n_cols];
        for r in 0..n_rows {
            for idx in indptr[r]..indptr[r + 1] {
                if data[idx] != 0.0 {
                    df[indices[idx]] += 1;
                }
            }
        }
        let add = if self.smooth_idf { 1.0 } else { 0.0 };
        let n_s = (n_rows as f64) + add;
        let idf: Vec<f64> = df
            .into_iter()
            .map(|d| (n_s / ((d as f64) + add)).ln() + 1.0)
            .collect();
        self.idf_ = Some(idf);
        Ok(())
    }

    pub fn fit_dense(&mut self, x: &Matrix) -> OchreResult<()> {
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        self.n_features_in_ = Some(x.cols);
        self.device_ = Some(self.device.to_string());
        if !self.use_idf {
            self.idf_ = None;
            return Ok(());
        }

        let cols = x.cols;
        let rows = x.rows;
        let df: Vec<usize> = (0..cols)
            .into_par_iter()
            .map(|j| (0..rows).filter(|&r| x.data[r * cols + j] != 0.0).count())
            .collect();
        let add = if self.smooth_idf { 1.0 } else { 0.0 };
        let n_s = (rows as f64) + add;
        let idf: Vec<f64> = df
            .into_iter()
            .map(|d| (n_s / ((d as f64) + add)).ln() + 1.0)
            .collect();
        self.idf_ = Some(idf);
        Ok(())
    }

    pub fn transform_csr(
        &self,
        data: &[f64],
        indices: &[usize],
        indptr: &[usize],
        n_rows: usize,
        n_cols: usize,
    ) -> OchreResult<CsrParts> {
        let expected = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        if n_cols != expected {
            return Err(OchreError::FeatureCountMismatch {
                features: n_cols,
                expected,
            });
        }
        let mut out_data = data.to_vec();
        if self.sublinear_tf {
            for v in &mut out_data {
                if *v > 0.0 {
                    *v = 1.0 + v.ln();
                }
            }
        }
        if self.use_idf {
            let idf = self.idf_.as_ref().ok_or(OchreError::NotFitted)?;
            for (v, &col_i) in out_data.iter_mut().zip(indices.iter()) {
                *v *= idf[col_i];
            }
        }
        if let Some(ref norm) = self.norm {
            normalize_csr_rows(&mut out_data, indptr, norm);
        }
        Ok((out_data, indices.to_vec(), indptr.to_vec(), n_rows, n_cols))
    }

    pub fn transform_dense(&self, x: &Matrix) -> OchreResult<Matrix> {
        let expected = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        if x.cols != expected {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected,
            });
        }
        let mut out = x.data.clone();
        if self.sublinear_tf {
            for v in &mut out {
                if *v > 0.0 {
                    *v = 1.0 + v.ln();
                }
            }
        }
        if self.use_idf {
            let idf = self.idf_.as_ref().ok_or(OchreError::NotFitted)?;
            for r in 0..x.rows {
                for j in 0..x.cols {
                    out[r * x.cols + j] *= idf[j];
                }
            }
        }
        let mat = Matrix::from_vec(x.rows, x.cols, out)?;
        if let Some(ref norm) = self.norm {
            let (normed, _) = crate::preprocessing::scalers::normalize_matrix(&mat, norm, 1)?;
            return Ok(normed);
        }
        Ok(mat)
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_TFIDF_TRANSFORMER);
        write_opt_str(&mut buf, &self.norm);
        write_bool(&mut buf, self.use_idf);
        write_bool(&mut buf, self.smooth_idf);
        write_bool(&mut buf, self.sublinear_tf);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.idf_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_TFIDF_TRANSFORMER)?;
        let norm = r.read_opt_str()?;
        let use_idf = r.read_bool()?;
        let smooth_idf = r.read_bool()?;
        let sublinear_tf = r.read_bool()?;
        let saved_device = r.read_opt_str()?;
        let idf_ = r.read_opt_f64_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            norm,
            use_idf,
            smooth_idf,
            sublinear_tf,
            device: dev,
            device_,
            idf_,
            n_features_in_,
        })
    }
}

/// Convert a collection of text documents to a matrix of token occurrences via MurmurHash3 (*≧ω≦*)
#[derive(Debug, Clone)]
pub struct HashingVectorizer {
    pub lowercase: bool,
    pub analyzer: String,
    pub ngram_range: (usize, usize),
    pub n_features: usize,
    pub binary: bool,
    pub norm: Option<String>,
    pub alternate_sign: bool,
    pub stop_words: HashSet<String>,
    pub device: DeviceType,
    pub device_: Option<String>,
}

impl HashingVectorizer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lowercase: bool,
        analyzer: String,
        ngram_range: (usize, usize),
        n_features: usize,
        binary: bool,
        norm: Option<String>,
        alternate_sign: bool,
        stop_words: HashSet<String>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if n_features == 0 {
            return Err(OchreError::InvalidParameter(
                "n_features must be > 0 ( >_< )".to_string(),
            ));
        }
        if let Some(ref n) = norm
            && n != "l1"
            && n != "l2"
        {
            return Err(OchreError::InvalidParameter(format!(
                "norm must be 'l1', 'l2', or None, got '{}' ( >_< )",
                n
            )));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            lowercase,
            analyzer,
            ngram_range,
            n_features,
            binary,
            norm,
            alternate_sign,
            stop_words,
            device: dev,
            device_: None,
        })
    }

    pub fn analyze_docs(&self, docs: &[String]) -> Vec<Vec<String>> {
        let (min_n, max_n) = self.ngram_range;
        let lowercase = self.lowercase;
        let stop = &self.stop_words;
        let analyzer = self.analyzer.as_str();
        docs.par_iter()
            .map(|doc| match analyzer {
                "char" => extract_char_ngrams(doc, lowercase, min_n, max_n, false),
                "char_wb" => extract_char_ngrams(doc, lowercase, min_n, max_n, true),
                _ => {
                    let toks = tokenize_default_words(doc, lowercase);
                    extract_word_ngrams(&toks, min_n, max_n, stop)
                }
            })
            .collect()
    }

    pub fn transform_from_tokenized(&self, docs_tokens: &[Vec<String>]) -> OchreResult<CsrParts> {
        let n_rows = docs_tokens.len();
        let n_feat = self.n_features;
        let alt = self.alternate_sign;
        let binary = self.binary;

        let per_row: Vec<Vec<(usize, f64)>> = docs_tokens
            .par_iter()
            .map(|toks| {
                let pairs: Vec<(String, f64)> = toks.iter().map(|t| (t.clone(), 1.0)).collect();
                let mut cols = hash_row_pairs(&pairs, n_feat, alt);
                if binary {
                    for &mut (_, ref mut v) in &mut cols {
                        *v = 1.0;
                    }
                }
                cols
            })
            .collect();

        let total_nnz: usize = per_row.iter().map(|r| r.len()).sum();
        let mut data = Vec::with_capacity(total_nnz);
        let mut indices = Vec::with_capacity(total_nnz);
        let mut indptr = Vec::with_capacity(n_rows + 1);
        indptr.push(0);
        for r in per_row {
            for (c, v) in r {
                indices.push(c);
                data.push(v);
            }
            indptr.push(data.len());
        }
        if let Some(ref norm) = self.norm {
            normalize_csr_rows(&mut data, &indptr, norm);
        }
        Ok((data, indices, indptr, n_rows, n_feat))
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_HASHING_VECTORIZER);
        write_bool(&mut buf, self.lowercase);
        write_str(&mut buf, &self.analyzer);
        write_usize(&mut buf, self.ngram_range.0);
        write_usize(&mut buf, self.ngram_range.1);
        write_usize(&mut buf, self.n_features);
        write_bool(&mut buf, self.binary);
        write_opt_str(&mut buf, &self.norm);
        write_bool(&mut buf, self.alternate_sign);
        let sw: Vec<String> = self.stop_words.iter().cloned().collect();
        write_str_vec(&mut buf, &sw);
        write_opt_str(&mut buf, &self.device_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_HASHING_VECTORIZER)?;
        let lowercase = r.read_bool()?;
        let analyzer = r.read_str()?;
        let n0 = r.read_usize()?;
        let n1 = r.read_usize()?;
        let n_features = r.read_usize()?;
        let binary = r.read_bool()?;
        let norm = r.read_opt_str()?;
        let alternate_sign = r.read_bool()?;
        let stop_words: HashSet<String> = r.read_str_vec()?.into_iter().collect();
        let saved_device = r.read_opt_str()?;
        let (dev, device_) = resolve_loaded_device(saved_device, false, device)?;
        Ok(Self {
            lowercase,
            analyzer,
            ngram_range: (n0, n1),
            n_features,
            binary,
            norm,
            alternate_sign,
            stop_words,
            device: dev,
            device_,
        })
    }
}
