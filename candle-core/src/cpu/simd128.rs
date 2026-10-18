use super::{Cpu, CpuBF16, CpuF16};
use core::arch::wasm32::*;
use half::{bf16, f16};

pub struct CurrentCpu {}

const STEP: usize = 16;
const EPR: usize = 4;
const ARR: usize = STEP / EPR;

impl Cpu for CurrentCpu {
    type Unit = v128;
    type Array = [v128; ARR];

    const STEP: usize = STEP;
    const EPR: usize = EPR;
    const ARR: usize = ARR;

    unsafe fn zero() -> Self::Unit {
        f32x4_splat(0.0)
    }

    unsafe fn zero_array() -> Self::Array {
        [Self::zero(); ARR]
    }

    unsafe fn from_f32(v: f32) -> Self::Unit {
        f32x4_splat(v)
    }

    unsafe fn load(mem_addr: *const f32) -> Self::Unit {
        v128_load(mem_addr as *mut v128)
    }

    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit {
        f32x4_add(a, b)
    }

    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit {
        f32x4_add(f32x4_mul(b, c), a)
    }

    unsafe fn vec_store(mem_addr: *mut f32, a: Self::Unit) {
        v128_store(mem_addr as *mut v128, a);
    }

    unsafe fn vec_reduce(mut x: Self::Array, y: *mut f32) {
        for i in 0..ARR / 2 {
            x[2 * i] = f32x4_add(x[2 * i], x[2 * i + 1]);
        }
        for i in 0..ARR / 4 {
            x[4 * i] = f32x4_add(x[4 * i], x[4 * i + 2]);
        }
        for i in 0..ARR / 8 {
            x[8 * i] = f32x4_add(x[8 * i], x[8 * i + 4]);
        }
        *y = f32x4_extract_lane::<0>(x[0])
            + f32x4_extract_lane::<1>(x[0])
            + f32x4_extract_lane::<2>(x[0])
            + f32x4_extract_lane::<3>(x[0]);
    }
}

// WebAssembly SIMD has no 16-bit float arithmetic. `f16` and `bf16` data is widened to `f32` lanes when it is loaded and
// rounded back when it is stored, which is what the scalar fallback does one element at a time (`f16 + f16` is an `f32`
// add rounded to `f16`), so the results are identical. These types exist because `vec_add_f16`, `vec_add_bf16`,
// `vec_scalar_add_f16` and `vec_scalar_add_bf16` in `cpu/mod.rs` are compiled whenever `simd128` is on and name
// `CurrentCpuF16` / `CurrentCpuBF16`; without them `candle-core` does not build with `-C target-feature=+simd128`.
macro_rules! half_cpu {
    ($name:ident, $trait:ident, $half:ty) => {
        pub struct $name {}

        impl $trait for $name {
            type Unit = v128;
            type Array = [v128; ARR];

            const STEP: usize = STEP;
            const EPR: usize = EPR;
            const ARR: usize = ARR;

            unsafe fn zero() -> Self::Unit {
                f32x4_splat(0.0)
            }

            unsafe fn zero_array() -> Self::Array {
                [f32x4_splat(0.0); ARR]
            }

            unsafe fn from_f32(v: f32) -> Self::Unit {
                f32x4_splat(v)
            }

            unsafe fn load(mem_addr: *const $half) -> Self::Unit {
                f32x4(
                    (*mem_addr).to_f32(),
                    (*mem_addr.add(1)).to_f32(),
                    (*mem_addr.add(2)).to_f32(),
                    (*mem_addr.add(3)).to_f32(),
                )
            }

            unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit {
                f32x4_add(a, b)
            }

            unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit {
                f32x4_add(f32x4_mul(b, c), a)
            }

            unsafe fn vec_store(mem_addr: *mut $half, a: Self::Unit) {
                *mem_addr = <$half>::from_f32(f32x4_extract_lane::<0>(a));
                *mem_addr.add(1) = <$half>::from_f32(f32x4_extract_lane::<1>(a));
                *mem_addr.add(2) = <$half>::from_f32(f32x4_extract_lane::<2>(a));
                *mem_addr.add(3) = <$half>::from_f32(f32x4_extract_lane::<3>(a));
            }

            unsafe fn vec_reduce(x: Self::Array, y: *mut f32) {
                <CurrentCpu as Cpu>::vec_reduce(x, y)
            }
        }
    };
}

half_cpu!(CurrentCpuF16, CpuF16, f16);
half_cpu!(CurrentCpuBF16, CpuBF16, bf16);

#[cfg(test)]
mod tests {
    use crate::cpu::{vec_add_bf16, vec_add_f16, vec_scalar_add_bf16, vec_scalar_add_f16};
    use half::{bf16, f16};

    /// A fixed pseudo-random sequence in [-4, 4) (xorshift), so the test needs no random-number crate.
    fn values(n: usize, mut state: u64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 24) as f32 * 8.0 - 4.0
            })
            .collect()
    }

    #[test]
    fn half_precision_adds_match_the_scalar_result_at_every_length() {
        for k in 0..70 {
            let (a, b) = (values(k, 11), values(k, 29));
            let (ah, bh): (Vec<f16>, Vec<f16>) = (
                a.iter().map(|v| f16::from_f32(*v)).collect(),
                b.iter().map(|v| f16::from_f32(*v)).collect(),
            );
            let mut out = vec![f16::ZERO; k];
            unsafe { vec_add_f16(ah.as_ptr(), bh.as_ptr(), out.as_mut_ptr(), k) };
            let want: Vec<f16> = ah.iter().zip(&bh).map(|(x, y)| *x + *y).collect();
            assert_eq!(out, want, "f16 add, k = {k}");
            unsafe { vec_scalar_add_f16(f16::from_f32(0.75), ah.as_ptr(), out.as_mut_ptr(), k) };
            let want: Vec<f16> = ah.iter().map(|x| *x + f16::from_f32(0.75)).collect();
            assert_eq!(out, want, "f16 scalar add, k = {k}");

            let (ab, bb): (Vec<bf16>, Vec<bf16>) = (
                a.iter().map(|v| bf16::from_f32(*v)).collect(),
                b.iter().map(|v| bf16::from_f32(*v)).collect(),
            );
            let mut out = vec![bf16::ZERO; k];
            unsafe { vec_add_bf16(ab.as_ptr(), bb.as_ptr(), out.as_mut_ptr(), k) };
            let want: Vec<bf16> = ab.iter().zip(&bb).map(|(x, y)| *x + *y).collect();
            assert_eq!(out, want, "bf16 add, k = {k}");
            unsafe { vec_scalar_add_bf16(bf16::from_f32(0.75), ab.as_ptr(), out.as_mut_ptr(), k) };
            let want: Vec<bf16> = ab.iter().map(|x| *x + bf16::from_f32(0.75)).collect();
            assert_eq!(out, want, "bf16 scalar add, k = {k}");
        }
    }
}
