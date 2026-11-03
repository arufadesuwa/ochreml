// OchreML Error Module (*≧ω≦*)
// Strongly-typed error enum replacing Result<T, String> across the entire engine.

use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum OchreError {
    #[error("Incompatible matrix dimensions for multiplication: ({rows_a}x{cols_a}) and ({rows_b}x{cols_b}) (´-ω-`)")]
    MatmulDimensionMismatch {
        rows_a: usize,
        cols_a: usize,
        rows_b: usize,
        cols_b: usize,
    },

    #[error("Matrix column count ({cols}) must match vector length ({len}) (・`ω´・)")]
    MatvecDimensionMismatch { cols: usize, len: usize },

    #[error("Shape mismatch: expected {expected_rows}x{expected_cols}, got {actual_rows}x{actual_cols} ( >_< )")]
    DimensionMismatch {
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },

    #[error("Number of feature rows ({features}) does not match target length ({targets}) ( >_< )")]
    LengthMismatch { features: usize, targets: usize },

    #[error("Sample weight length ({weights}) must match sample count ({samples}) ( >_< )")]
    WeightLengthMismatch { weights: usize, samples: usize },

    #[error("Training data cannot be empty (´；ω；`)")]
    EmptyData,

    #[error("Coefficient matrix A must be square (n x n) (´；ω；`)")]
    NonSquareMatrix,

    #[error("Constant vector b length must match row count of matrix A ( >_< )")]
    VectorLengthMismatch,

    #[error("Matrix B row count must match row count of matrix A ( >_< )")]
    MatrixRowMismatch,

    #[error("Matrix is singular or near-singular: {0} (´-ω-`)")]
    SingularMatrix(String),

    #[error("Failed to solve linear system after numerical regularization: {0} (T_T)")]
    SolverRegularizationFailed(String),

    #[error("Sample weights must be non-negative valid numbers (・`ω´・)")]
    InvalidWeights,

    #[error("Sum of sample weights must be strictly positive (´；ω；`)")]
    NonPositiveWeightSum,

    #[error("Model is not fitted yet. Call fit() first (´-ω-`)")]
    NotFitted,

    #[error("Number of features in test data ({features}) does not match model ({expected}) (・`ω´・)")]
    FeatureCountMismatch { features: usize, expected: usize },

    #[error("Unrecognized device string '{0}'. Available devices: 'auto', 'cpu', 'cuda:0', 'cuda:all', 'tpu' ( >_< )")]
    InvalidDevice(String),

    #[error("{0}")]
    DeviceError(String),

    #[error("Invalid file extension for '{0}'. Model files must use the '.dit' extension ( >_< )")]
    InvalidModelExtension(String),

    #[error("Invalid .dit file format (missing DIT header) (´-ω-`)")]
    InvalidDitHeader,

    #[error("Model type mismatch in .dit file (expected tag {expected}, got {actual}) (・`ω´・)")]
    DitTagMismatch { expected: u8, actual: u8 },

    #[error("Corrupted or truncated .dit model file (´；ω；`)")]
    CorruptedDitFile,

    #[error("Invalid tree node tag in .dit file ( >_< )")]
    InvalidTreeNodeTag,

    #[error("Invalid UTF-8 in .dit model file ( >_< )")]
    InvalidUtf8InDit,

    #[error("Invalid criterion '{0}'. Supported criteria are 'gini' and 'entropy' (・`ω´・)")]
    InvalidCriterion(String),

    #[error("Invalid regressor criterion '{0}'. Supported criterion is 'squared_error' (・`ω´・)")]
    InvalidRegressorCriterion(String),

    #[error("{0}")]
    General(String),
}

pub type OchreResult<T> = Result<T, OchreError>;
