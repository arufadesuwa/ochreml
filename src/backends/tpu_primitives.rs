// OchreML Google TPU StableHLO Primitives (N1, N2) (*≧ω≦*)
// Emits power-of-two padded StableHLO MLIR programs executed via PJRT.
// N6 (batched_histogram_split) intentionally routes to CPU Rayon per MODEL_ROADMAP.md.

use crate::backends::tpu::{execute_tpu_unary_program, next_pow2_dim};
use crate::error::OchreResult;
use crate::matrix::Matrix;
use crate::primitives::stats_ops::ElementwiseOp;

const MIN_TPU_ELEMENTS: usize = 65_536;

fn emit_elementwise_mlir(rows: usize, cols: usize, op: ElementwiseOp) -> String {
    let ty = format!("tensor<{}x{}xf32>", rows, cols);
    let body = match op {
        ElementwiseOp::Sigmoid => format!("    %0 = stablehlo.logistic %arg0 : {ty}"),
        ElementwiseOp::Tanh => format!("    %0 = stablehlo.tanh %arg0 : {ty}"),
        ElementwiseOp::Relu => format!(
            "    %zero = stablehlo.constant dense<0.0> : {ty}\n    %0 = stablehlo.maximum %arg0, %zero : {ty}"
        ),
        ElementwiseOp::Exp => format!("    %0 = stablehlo.exponential %arg0 : {ty}"),
        ElementwiseOp::Log { eps } => format!(
            "    %eps = stablehlo.constant dense<{:.8e}> : {ty}\n    %clamped = stablehlo.maximum %arg0, %eps : {ty}\n    %0 = stablehlo.log %clamped : {ty}",
            eps as f32
        ),
        ElementwiseOp::Clip { min, max } => format!(
            "    %minv = stablehlo.constant dense<{:.8e}> : {ty}\n    %maxv = stablehlo.constant dense<{:.8e}> : {ty}\n    %0 = stablehlo.clamp %minv, %arg0, %maxv : {ty}",
            min as f32, max as f32
        ),
        ElementwiseOp::SoftThreshold { lambda } => format!(
            "    %lam = stablehlo.constant dense<{:.8e}> : {ty}\n    %nlam = stablehlo.constant dense<{:.8e}> : {ty}\n    %zero = stablehlo.constant dense<0.0> : {ty}\n    %gt = stablehlo.compare GT, %arg0, %lam, FLOAT : ({ty}, {ty}) -> tensor<{rows}x{cols}xi1>\n    %lt = stablehlo.compare LT, %arg0, %nlam, FLOAT : ({ty}, {ty}) -> tensor<{rows}x{cols}xi1>\n    %pos = stablehlo.subtract %arg0, %lam : {ty}\n    %neg = stablehlo.add %arg0, %lam : {ty}\n    %r1 = stablehlo.select %lt, %neg, %zero : tensor<{rows}x{cols}xi1>, {ty}\n    %0 = stablehlo.select %gt, %pos, %r1 : tensor<{rows}x{cols}xi1>, {ty}",
            lambda as f32,
            -(lambda as f32)
        ),
        ElementwiseOp::Affine { scale, shift } => format!(
            "    %sc = stablehlo.constant dense<{:.8e}> : {ty}\n    %sh = stablehlo.constant dense<{:.8e}> : {ty}\n    %mul = stablehlo.multiply %arg0, %sc : {ty}\n    %0 = stablehlo.add %mul, %sh : {ty}",
            scale as f32, shift as f32
        ),
    };

    format!(
        "module @ochreml_elem {{\n  func.func @main(%arg0: {ty}) -> {ty} {{\n{body}\n    return %0 : {ty}\n  }}\n}}\n"
    )
}

/// TPU implementation of N2 (`elementwise_op`) via cached StableHLO programs.
pub fn tpu_elementwise_op(
    device_id: usize,
    a: &Matrix,
    op: ElementwiseOp,
) -> Option<OchreResult<Matrix>> {
    if a.data.len() < MIN_TPU_ELEMENTS {
        return None;
    }
    let pad_rows = next_pow2_dim(a.rows, 16);
    let pad_cols = next_pow2_dim(a.cols, 16);
    let key = format!("elem:{:?}:{}x{}", op, pad_rows, pad_cols);
    let mlir = emit_elementwise_mlir(pad_rows, pad_cols, op);
    let out =
        execute_tpu_unary_program(device_id, key, &mlir, a, a.rows, a.cols, pad_rows, pad_cols)?;
    Some(Ok(out))
}
