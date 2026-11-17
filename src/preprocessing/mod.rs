// OchreML Preprocessing, Imputation & Feature Extraction Engine (*≧ω≦*)
// Modular Rust + Rayon + Backend-accelerated implementations for scalers, encoders,
// distribution transforms, discretizers, feature engineering, imputers, and text vectorizers.

pub mod encoders;
pub mod features;
pub mod impute;
pub mod math_utils;
pub mod py_encoders;
pub mod py_helpers;
pub mod py_impute_text;
pub mod py_scalers;
pub mod scalers;
pub mod text_fe;
pub mod transforms;

use pyo3::prelude::*;

/// Register all preprocessing, imputation, and feature extraction classes/functions on `_ochreml`.
pub fn register_py_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_scalers::_rust_normalize_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(py_scalers::_rust_normalize_into, m)?)?;
    m.add_function(wrap_pyfunction!(py_scalers::_rust_binarize_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(py_scalers::_rust_binarize_into, m)?)?;

    m.add_class::<py_scalers::PyStandardScaler>()?;
    m.add_class::<py_scalers::PyMinMaxScaler>()?;
    m.add_class::<py_scalers::PyMaxAbsScaler>()?;
    m.add_class::<py_scalers::PyRobustScaler>()?;
    m.add_class::<py_scalers::PyNormalizer>()?;
    m.add_class::<py_scalers::PyBinarizer>()?;
    m.add_class::<py_scalers::PyKernelCenterer>()?;
    m.add_class::<py_scalers::PyPolynomialFeatures>()?;
    m.add_class::<py_scalers::PySplineTransformer>()?;
    m.add_class::<py_scalers::PyPowerTransformer>()?;
    m.add_class::<py_scalers::PyQuantileTransformer>()?;
    m.add_class::<py_scalers::PyKBinsDiscretizer>()?;

    m.add_class::<py_encoders::PyOneHotEncoder>()?;
    m.add_class::<py_encoders::PyOrdinalEncoder>()?;
    m.add_class::<py_encoders::PyLabelEncoder>()?;
    m.add_class::<py_encoders::PyLabelBinarizer>()?;
    m.add_class::<py_encoders::PyMultiLabelBinarizer>()?;
    m.add_class::<py_encoders::PyTargetEncoder>()?;

    m.add_class::<py_impute_text::PyMissingIndicator>()?;
    m.add_class::<py_impute_text::PySimpleImputer>()?;
    m.add_class::<py_impute_text::PyKNNImputer>()?;
    m.add_class::<py_impute_text::PyFeatureHasher>()?;
    m.add_class::<py_impute_text::PyDictVectorizer>()?;
    m.add_class::<py_impute_text::PyCountVectorizer>()?;
    m.add_class::<py_impute_text::PyTfidfTransformer>()?;
    m.add_class::<py_impute_text::PyHashingVectorizer>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::encoders::{CatValue, LabelEncoder, OneHotEncoder, OrdinalEncoder};
    use super::features::{KernelCenterer, PolynomialFeatures, SplineTransformer};
    use super::impute::{KNNImputer, SimpleImputer};
    use super::math_utils::{ndtr, ndtri};
    use super::scalers::{MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler, StandardScaler};
    use super::text_fe::{CountVectorizer, FeatureHasher, TfidfTransformer, murmurhash3_bytes_s32};
    use super::transforms::{KBinsDiscretizer, PowerTransformer, QuantileTransformer};
    use crate::matrix::Matrix;
    use std::collections::HashSet;

    #[test]
    fn test_scalers_roundtrip_and_dit() {
        let x = Matrix::from_2d(&[vec![1.0, 10.0], vec![3.0, 20.0], vec![5.0, 30.0]]).unwrap();

        let mut ss = StandardScaler::new(true, true, Some("cpu".to_string())).unwrap();
        ss.fit(&x, None).unwrap();
        let z = ss.transform(&x).unwrap();
        let inv = ss.inverse_transform(&z).unwrap();
        for (&a, &b) in x.data.iter().zip(inv.data.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
        let ss2 = StandardScaler::from_bytes(&ss.to_bytes(), Some("cpu")).unwrap();
        assert_eq!(ss.mean_, ss2.mean_);

        let mut mm = MinMaxScaler::new((0.0, 1.0), false, Some("cpu".to_string())).unwrap();
        mm.fit(&x).unwrap();
        let m_out = mm.transform(&x).unwrap();
        assert!((m_out.data[0] - 0.0).abs() < 1e-12);
        assert!((m_out.data[4] - 1.0).abs() < 1e-12);
        let m_inv = mm.inverse_transform(&m_out).unwrap();
        for (&a, &b) in x.data.iter().zip(m_inv.data.iter()) {
            assert!((a - b).abs() < 1e-12);
        }

        let mut ma = MaxAbsScaler::new(Some("cpu".to_string())).unwrap();
        ma.fit(&x).unwrap();
        let ma_inv = ma.inverse_transform(&ma.transform(&x).unwrap()).unwrap();
        for (&a, &b) in x.data.iter().zip(ma_inv.data.iter()) {
            assert!((a - b).abs() < 1e-12);
        }

        let mut rs = RobustScaler::new(true, true, (25.0, 75.0), false, None).unwrap();
        rs.fit(&x).unwrap();
        let rs_inv = rs.inverse_transform(&rs.transform(&x).unwrap()).unwrap();
        for (&a, &b) in x.data.iter().zip(rs_inv.data.iter()) {
            assert!((a - b).abs() < 1e-12);
        }

        let mut norm = Normalizer::new("l2".to_string(), None).unwrap();
        norm.fit(&x).unwrap();
        let n_out = norm.transform(&x).unwrap();
        let row0_norm = (n_out.data[0].powi(2) + n_out.data[1].powi(2)).sqrt();
        assert!((row0_norm - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_ndtri_ndtr_and_power_quantile() {
        for &p in &[1e-7, 0.001, 0.25, 0.5, 0.75, 0.999, 1.0 - 1e-7] {
            let z = ndtri(p);
            let p_back = ndtr(z);
            assert!((p - p_back).abs() < 1e-13, "p={p} p_back={p_back}");
        }

        let x = Matrix::from_2d(&[
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![4.0, 8.0],
            vec![8.0, 16.0],
            vec![16.0, 32.0],
        ])
        .unwrap();

        let mut pt = PowerTransformer::new("yeo-johnson".to_string(), true, None).unwrap();
        let pt_out = pt.fit_transform(&x).unwrap();
        let pt_inv = pt.inverse_transform(&pt_out).unwrap();
        for (&a, &b) in x.data.iter().zip(pt_inv.data.iter()) {
            assert!((a - b).abs() < 1e-10);
        }

        let mut qt =
            QuantileTransformer::new(5, "normal".to_string(), false, 100, Some(0), None).unwrap();
        qt.fit(&x).unwrap();
        let q_out = qt.transform(&x).unwrap();
        let q_inv = qt.inverse_transform(&q_out).unwrap();
        for (&a, &b) in x.data.iter().zip(q_inv.data.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn test_kbins_poly_spline_kernel() {
        let x = Matrix::from_2d(&[
            vec![0.0, 1.0],
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![4.0, 5.0],
        ])
        .unwrap();

        let mut kb = KBinsDiscretizer::new(
            vec![3],
            "onehot-dense".to_string(),
            "kmeans".to_string(),
            None,
            None,
        )
        .unwrap();
        kb.fit(&x).unwrap();
        let kb_out = kb.transform_dense(&x).unwrap();
        assert_eq!(kb_out.rows, 5);

        let mut pf = PolynomialFeatures::new(0, 2, false, true, "C".to_string(), None).unwrap();
        pf.fit(&x).unwrap();
        let pf_out = pf.transform(&x).unwrap();
        assert_eq!(pf_out.cols, 6);
        assert!((pf_out.data[0] - 1.0).abs() < 1e-12);

        let mut sp = SplineTransformer::new(
            4,
            3,
            "uniform".to_string(),
            "constant".to_string(),
            true,
            "C".to_string(),
            None,
        )
        .unwrap();
        sp.fit(&x, None).unwrap();
        let sp_out = sp.transform(&x).unwrap();
        assert_eq!(sp_out.rows, 5);
        assert_eq!(sp_out.cols, 12);

        let k = Matrix::from_2d(&[vec![2.0, 1.0], vec![1.0, 2.0]]).unwrap();
        let mut kc = KernelCenterer::new(None).unwrap();
        kc.fit(&k).unwrap();
        let kc_out = kc.transform(&k).unwrap();
        assert!((kc_out.data[0] - 0.5).abs() < 1e-12);
        assert!((kc_out.data[1] + 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_encoders_and_imputers() {
        let cols = vec![
            vec![
                CatValue::Str("b".into()),
                CatValue::Str("a".into()),
                CatValue::Str("b".into()),
            ],
            vec![CatValue::Int(2), CatValue::Int(1), CatValue::Int(2)],
        ];
        let mut ohe = OneHotEncoder::new("error".into(), None, None, None, None).unwrap();
        ohe.fit(&cols, None, None).unwrap();
        let dense = ohe.transform_dense(&cols).unwrap();
        assert_eq!(dense.rows, 3);
        assert_eq!(dense.cols, 4);
        let inv = ohe.inverse_transform(&dense).unwrap();
        assert_eq!(inv[0][0], Some(CatValue::Str("b".into())));

        let mut ord =
            OrdinalEncoder::new("error".into(), None, f64::NAN, None, None, None).unwrap();
        ord.fit(&cols, None).unwrap();
        let ord_out = ord.transform(&cols).unwrap();
        assert_eq!(ord_out.data, vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);

        let mut le = LabelEncoder::new(None).unwrap();
        le.fit(&cols[0]).unwrap();
        assert_eq!(le.transform(&cols[0]).unwrap(), vec![1, 0, 1]);

        let x_nan = Matrix::from_2d(&[
            vec![1.0, 2.0, f64::NAN],
            vec![3.0, 4.0, 3.0],
            vec![f64::NAN, 6.0, 5.0],
            vec![8.0, 8.0, 7.0],
        ])
        .unwrap();
        let mut si =
            SimpleImputer::new(CatValue::Missing, "mean".into(), None, false, false, None).unwrap();
        si.fit_numeric(&x_nan).unwrap();
        let si_out = si.transform_numeric(&x_nan).unwrap();
        assert!((si_out.data[2] - 5.0).abs() < 1e-12);

        let mut knn = KNNImputer::new(f64::NAN, 2, "uniform".into(), false, false, None).unwrap();
        knn.fit(&x_nan).unwrap();
        let knn_out = knn.transform(&x_nan).unwrap();
        assert!((knn_out.data[2] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_murmurhash3_and_text_fe() {
        assert_eq!(murmurhash3_bytes_s32(b"foo", 0), -156_908_512);
        assert_eq!(murmurhash3_bytes_s32(b"bar", 0), 1_158_584_717);
        assert_eq!(murmurhash3_bytes_s32(b"hello", 0), 613_153_351);

        let fh = FeatureHasher::new(8, "dict".into(), true, None).unwrap();
        let (data, indices, indptr, nr, nc) = fh
            .transform_pairs(&[vec![("foo".into(), 2.0), ("bar".into(), -1.0)]])
            .unwrap();
        assert_eq!((nr, nc), (1, 8));
        assert_eq!(indptr, vec![0, 2]);
        assert_eq!(indices, vec![0, 5]);
        assert_eq!(data, vec![-2.0, -1.0]);

        let mut cv = CountVectorizer::new(
            true,
            "word".into(),
            (1, 1),
            1.0,
            1.0,
            None,
            false,
            HashSet::new(),
            None,
            None,
        )
        .unwrap();
        let docs = vec!["hello world".into(), "hello rust".into()];
        let toks = cv.analyze_docs(&docs);
        cv.fit_from_tokenized(&toks).unwrap();
        let (cd, ci, cp, cnr, cnc) = cv.transform_from_tokenized(&toks).unwrap();
        let mut tfidf = TfidfTransformer::new(Some("l2".into()), true, true, false, None).unwrap();
        tfidf.fit_csr(&cd, &ci, &cp, cnr, cnc).unwrap();
        let (td, _, _, _, _) = tfidf.transform_csr(&cd, &ci, &cp, cnr, cnc).unwrap();
        assert!((td[0].powi(2) + td[1].powi(2) - 1.0).abs() < 1e-12);
    }
}
