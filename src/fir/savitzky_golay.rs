// SPDX-License-Identifier: MIT

//! Streaming Savitzky-Golay filter using Gram polynomial weight.
//!
//! The Savitzky-Golay filter fits a polynomial of degree `n` to a sliding
//! window of `2m + 1` samples in a least-squares sense and evaluates that
//! polynomial (or one of its derivatives) at the newest sample.  Compared to a
//! simple moving average it preserves peaks and edges better, and compared to a
//! biquad + finite-difference chain it computes a smoothed derivative in a
//! single pass with lower noise amplification.
//!
//! # weight
//!
//! Coefficients are computed once at construction via the Gram polynomial
//! recurrence of Gorry (1990) — no lookup tables, no matrix inversion.  The
//! recurrence is numerically stable for polynomial orders up to 3 (the builder
//! enforces `n ≤ 3`).
//!
//! # Causal operation
//!
//! The filter always evaluates at the most recent sample (`data_point = m`), so
//! it introduces no future-sample look-ahead and is suitable for real-time use.
//!
//! # Cold start
//!
//! On the first call to [`apply`](crate::Filter::apply) the entire window is
//! edge-padded with that sample.  For smoothing (`deriv_order = 0`) this
//! returns the input unchanged; for derivatives it returns zero (constant
//! signal → zero gradient). No silent output-before-ready period.
//!
//! # `no_std` window limit
//!
//! Without the `std` feature the weight and sample buffers are fixed-size stack
//! arrays.  The maximum window size is 19 samples (`m ≤ 9`).  Larger windows
//! require the `std` feature.
//!
//! # Example — smoothing
//!
//! ```
//! use air_filters::fir::savitzky_golay::{SavitzkyGolayFilter, SgConfigBuilder};
//! use air_filters::Filter;
//!
//! let cfg = SgConfigBuilder::new()
//!     .window_size(7)
//!     .order(2)
//!     .deriv_order(0)
//!     .sample_frequency_hz(1_000.0_f64)
//!     .build()
//!     .unwrap();
//! let mut f = SavitzkyGolayFilter::new(cfg);
//!
//! // Constant signal → output equals input.
//! for _ in 0..20 {
//!     assert!((f.apply(3.0) - 3.0).abs() < 1e-10);
//! }
//! ```
//!
//! # Example — first derivative
//!
//! ```
//! use air_filters::fir::savitzky_golay::{SavitzkyGolayFilter, SgConfigBuilder};
//! use air_filters::Filter;
//!
//! // 1 kHz sample rate, ramp of 1 unit/sample → derivative ≈ 1000 units/s.
//! let cfg = SgConfigBuilder::new()
//!     .window_size(7)
//!     .order(2)
//!     .deriv_order(1)
//!     .sample_frequency_hz(1_000.0_f64)
//!     .build()
//!     .unwrap();
//! let mut f = SavitzkyGolayFilter::new(cfg);
//!
//! // Prime with 20 samples of a unit ramp; derivative should converge to 1000 rad/s².
//! for i in 0..20 {
//!     let _ = f.apply(i as f64);
//! }
//! let out = f.apply(20.0_f64);
//! assert!((out - 1_000.0).abs() < 1.0, "got {out}");
//! ```

use num_traits::float::FloatCore;

use crate::{Error, Filter};

// ── Gram polynomial math
// ──────────────────────────────────────────────────────
//
// Direct translation of Gorry (1990), "General Least-Squares Smoothing and
// Differentiation by the Convolution (Savitzky-Golay) Method".

/// Evaluates the Gram polynomial (s=0) or its s-th derivative at point i,
/// within a window of 2m+1 points, for polynomial order k.
fn gram_poly<T: FloatCore>(i: T, m: i32, k: i32, s: i32) -> T {
    if k > 0 {
        let ms = t!(m);
        let ks = t!(k);
        let ss = t!(s);
        ((t!(4)) * ks - t!(2)) / (ks * (t!(2) * ms - ks + T::one()))
            * (i * gram_poly(i, m, k - 1, s) + ss * gram_poly(i, m, k - 1, s - 1))
            - (ks - T::one()) * (t!(2) * ms + ks) / (ks * (t!(2) * ms - ks + T::one()))
                * gram_poly(i, m, k - 2, s)
    } else if k == 0 && s == 0 {
        T::one()
    } else {
        T::zero()
    }
}

/// Generalised factorial: a(a-1)(a-2)...(a-b+1).
fn gen_fact(a: i32, b: i32) -> i32 {
    let mut gf = 1;
    for j in (a - b + 1)..=a {
        gf *= j;
    }
    gf
}

/// Weight of the i-th data point for the t-th evaluation point of the s-th
/// derivative, over a window of 2m+1 points, polynomial order n.
fn weight<T: FloatCore>(i: T, t: i32, m: i32, n: i32, s: i32) -> T {
    let ts = T::from(t).unwrap();
    (0..=n).fold(T::zero(), |sum, k| {
        let ks = T::from(k).unwrap();
        sum + (T::from(2).unwrap() * ks + T::one()) * T::from(gen_fact(2 * m, k)).unwrap()
            / T::from(gen_fact(2 * m + k + 1, k + 1)).unwrap()
            * gram_poly(i, m, k, 0)
            * gram_poly(ts, m, k, s)
    })
}

#[cfg(not(feature = "std"))]
mod weight_array {
    use core::ops::{Index, IndexMut};
    use num_traits::float::FloatCore;

    pub(super) const MAX_NUM_POINTS: usize = 20;

    /// Size constrained array to hold weights, size is determined at initialization time, but cannot
    /// exceed MAX_NUM_POINTS; API
    #[derive(Debug, Clone)]
    pub struct WeightArray<T: FloatCore> {
        weights: [T; MAX_NUM_POINTS],
        size: usize,
    }

    impl<T: FloatCore> WeightArray<T> {
        pub fn new(size: usize) -> Self {
            assert!(
                size <= MAX_NUM_POINTS,
                "Size ({size}) must be less than or equal to MAX_NUM_POINTS ({MAX_NUM_POINTS})"
            );
            Self {
                weights: [T::zero(); MAX_NUM_POINTS],
                size,
            }
        }

        pub fn iter(&self) -> impl Iterator<Item = &T> {
            self.weights.iter().take(self.size)
        }
    }

    // Implement iteration and indexing, but not push or pop
    impl<T: FloatCore> Index<usize> for WeightArray<T> {
        type Output = T;

        fn index(&self, index: usize) -> &Self::Output {
            &self.weights[index]
        }
    }

    impl<T: FloatCore> IndexMut<usize> for WeightArray<T> {
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            &mut self.weights[index]
        }
    }

    pub fn make_weight_array<T: FloatCore>(size: usize) -> WeightArray<T> {
        WeightArray::new(size)
    }
}

#[cfg(feature = "std")]
mod weight_array {
    extern crate std;
    use std::vec::Vec;

    use num_traits::Zero;

    pub type WeightArray<T> = Vec<T>;
    pub fn make_weight_array<T: Zero + Copy>(size: usize) -> WeightArray<T> {
        std::vec![T::zero(); size]
    }
}
use weight_array::{make_weight_array, WeightArray};
/// Maximum ring-buffer capacity under `no_std` — same as the weight array limit.
#[cfg(not(feature = "std"))]
const SG_BUF_CAP: usize = weight_array::MAX_NUM_POINTS;

/// Sample window buffer type.  Under `no_std` this is a const-generic stack
/// array; under `std` it is a [`VecDeque`](std::collections::VecDeque).
#[cfg(not(feature = "std"))]
type SgBuf<T> = crate::util::ring_buf::RingBuf<T, SG_BUF_CAP>;
#[cfg(feature = "std")]
type SgBuf<T> = crate::util::ring_buf::GrowableRingBuf<T>;

fn generate_weights<T: FloatCore>(m: i32, t: i32, n: i32, s: i32) -> WeightArray<T> {
    let mut weights = make_weight_array(2 * m as usize + 1);
    for i in 0..=(2 * m) {
        // Cast i and m individually to avoid underflow when i < m
        weights[i as usize] = weight(T::from(i).unwrap() - T::from(m).unwrap(), t, m, n, s);
    }
    weights
}

// ── Configuration
// ─────────────────────────────────────────────────────────────

/// Configuration for a [`SavitzkyGolayFilter`].
///
/// Construct via [`SgConfigBuilder`].
#[derive(Debug, Clone, Copy)]
pub struct SgConfig<T: FloatCore> {
    m: i32,      // half-window; full window = 2m+1
    n: i32,      // polynomial order
    s: i32,      // derivative order
    dt_pow_s: T, // (1 / sample_frequency_hz)^s, pre-computed
}

/// Builder for [`SgConfig`].
///
/// All fields are optional; defaults match a 7-point quadratic smoother at 1
/// kHz.
///
/// | Field                 | Default |
/// |-----------------------|---------|
/// | `window_size`         | 7       |
/// | `order`               | 2       |
/// | `deriv_order`         | 0       |
/// | `sample_frequency_hz` | 1000 |
#[derive(Debug, Clone, Copy)]
pub struct SgConfigBuilder<T: FloatCore> {
    window_size: Option<i32>,
    order: Option<i32>,
    deriv_order: Option<i32>,
    sample_frequency_hz: Option<T>,
}

impl<T: FloatCore> Default for SgConfigBuilder<T> {
    fn default() -> Self {
        Self {
            window_size: None,
            order: None,
            deriv_order: None,
            sample_frequency_hz: None,
        }
    }
}

impl<T: FloatCore> SgConfigBuilder<T> {
    /// Creates a new builder with all fields unset (defaults will apply at
    /// [`build`](Self::build)).
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the window size `2m + 1`.  Must be a positive odd integer ≥ 3.
    pub fn window_size(mut self, window_size: i32) -> Self {
        self.window_size = Some(window_size);
        self
    }

    /// Sets the polynomial order `n`.  Must satisfy `0 ≤ n < window_size` and
    /// `n ≤ 3`.
    pub fn order(mut self, order: i32) -> Self {
        self.order = Some(order);
        self
    }

    /// Sets the derivative order `s`.  `0` = smoothing, `1` = first derivative,
    /// etc. Must satisfy `0 ≤ s ≤ n`.
    pub fn deriv_order(mut self, deriv_order: i32) -> Self {
        self.deriv_order = Some(deriv_order);
        self
    }

    /// Sets the sample frequency in Hz.  Used to scale derivative outputs:
    /// the filter divides by `(1 / sample_frequency_hz)^s` so that a
    /// first-derivative output is in signal-units/second rather than
    /// signal-units/sample.
    pub fn sample_frequency_hz(mut self, hz: T) -> Self {
        self.sample_frequency_hz = Some(hz);
        self
    }

    /// Validates all parameters and returns a [`SgConfig`], or an [`Error`] if
    /// any parameter is invalid.
    ///
    /// # Errors
    /// - [`Error::SgNonPositiveWindowSize`] — window size ≤ 0
    /// - [`Error::SgEvenWindowSize`] — window size is even
    /// - [`Error::SgWindowTooLargeForNoStd`] — window size > 19 without `std`
    ///   feature
    /// - [`Error::SgOrderTooHigh`] — `n ≥ window_size` or `n > 3` or `n < 0`
    /// - [`Error::SgDerivationOrderTooHigh`] — `s > n` or `s < 0`
    /// - [`Error::NonPositiveSampleFrequency`] — `sample_frequency_hz ≤ 0`
    /// - [`Error::NonFiniteSampleFrequency`] — `sample_frequency_hz` is NaN or
    ///   infinite
    pub fn build(self) -> Result<SgConfig<T>, Error> {
        let window_size = self.window_size.unwrap_or(7);
        let n = self.order.unwrap_or(2);
        let s = self.deriv_order.unwrap_or(0);
        let fs = self
            .sample_frequency_hz
            .unwrap_or_else(|| T::from(1000).unwrap());

        if window_size <= 0 {
            return Err(Error::SgNonPositiveWindowSize);
        }
        if window_size % 2 == 0 {
            return Err(Error::SgEvenWindowSize);
        }

        #[cfg(not(feature = "std"))]
        if window_size as usize > SG_BUF_CAP {
            return Err(Error::SgWindowTooLargeForNoStd);
        }

        if n < 0 || n >= window_size || n > 3 {
            return Err(Error::SgOrderTooHigh);
        }
        if s < 0 || s > n {
            return Err(Error::SgDerivationOrderTooHigh);
        }
        if !fs.is_finite() {
            return Err(Error::NonFiniteSampleFrequency);
        }
        if fs <= T::zero() {
            return Err(Error::NonPositiveSampleFrequency);
        }

        let dt = T::one() / fs;
        let dt_pow_s = dt.powi(s);
        let m = window_size / 2;

        Ok(SgConfig { m, n, s, dt_pow_s })
    }
}

// ── Streaming filter
// ──────────────────────────────────────────────────────────

/// Streaming Savitzky-Golay filter that accepts one sample at a time.
///
/// See the [module documentation](self) for usage guidance and examples.
#[derive(Clone)]
pub struct SavitzkyGolayFilter<T: FloatCore> {
    weights: WeightArray<T>,
    buf: SgBuf<T>,
    config: SgConfig<T>,
}

impl<T: FloatCore> SavitzkyGolayFilter<T> {
    /// Creates a new filter from the given configuration.
    pub fn new(config: SgConfig<T>) -> Self {
        let SgConfig { m, n, s, .. } = config;
        let window = 2 * m as usize + 1;
        Self {
            weights: generate_weights(m, m, n, s),
            buf: SgBuf::new_empty(window, T::zero()),
            config,
        }
    }

    /// Returns the window size `2m + 1`.
    pub fn window_size(&self) -> usize {
        2 * self.config.m as usize + 1
    }

    /// Returns the polynomial order `n`.
    pub fn order(&self) -> i32 {
        self.config.n
    }

    /// Returns the derivative order `s` encoded in the weights.
    ///
    /// `0` = smoothing, `1` = first derivative, etc.
    pub fn deriv_order(&self) -> i32 {
        self.config.s
    }

    /// Returns the filter configuration.
    pub fn config(&self) -> &SgConfig<T> {
        &self.config
    }

    fn dot(&self) -> T {
        self.weights
            .iter()
            .enumerate()
            .fold(T::zero(), |acc, (i, w)| acc + *w * self.buf.get(i))
    }
}

impl<T: FloatCore> Filter<T> for SavitzkyGolayFilter<T> {
    /// Pushes one sample and returns the filtered value (or derivative
    /// estimate).
    ///
    /// On the **first call** the entire window is edge-padded with `input`, so
    /// the output is always valid.  For smoothing (`deriv_order = 0`) the first
    /// output equals `input`; for derivatives it equals zero.
    fn apply(&mut self, input: T) -> T {
        if !self.buf.is_primed() {
            self.buf.fill(input);
        } else {
            self.buf.push_back(input);
        }
        self.dot() / self.config.dt_pow_s
    }

    /// Fills the sample window with `steady_output`, equivalent to the filter
    /// having seen that value at every past sample.
    ///
    /// For smoothing filters this sets the output to `steady_output`.  For
    /// derivative filters this sets the output to zero (constant signal).
    fn reset(&mut self, steady_output: T) -> Result<(), Error> {
        if !steady_output.is_finite() {
            return Err(Error::NonFiniteState);
        }
        self.buf.fill(steady_output);
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "alloc")]
    use alloc::vec::Vec;
    use approx::assert_relative_eq;

    fn cfg(window: i32, order: i32, deriv: i32, fs: f64) -> SgConfig<f64> {
        SgConfigBuilder::new()
            .window_size(window)
            .order(order)
            .deriv_order(deriv)
            .sample_frequency_hz(fs)
            .build()
            .unwrap()
    }

    // ── Config validation ─────────────────────────────────────────────────────

    #[test]
    fn rejects_non_positive_window() {
        let e = SgConfigBuilder::<f64>::new()
            .window_size(0)
            .build()
            .unwrap_err();
        assert_eq!(e, Error::SgNonPositiveWindowSize);
        let e = SgConfigBuilder::<f64>::new()
            .window_size(-3)
            .build()
            .unwrap_err();
        assert_eq!(e, Error::SgNonPositiveWindowSize);
    }

    #[test]
    fn rejects_even_window() {
        let e = SgConfigBuilder::<f64>::new()
            .window_size(6)
            .build()
            .unwrap_err();
        assert_eq!(e, Error::SgEvenWindowSize);
    }

    #[test]
    fn rejects_order_too_high() {
        // n ≥ window_size
        let e = SgConfigBuilder::<f64>::new()
            .window_size(7)
            .order(7)
            .build()
            .unwrap_err();
        assert_eq!(e, Error::SgOrderTooHigh);
        // n > 3
        let e = SgConfigBuilder::<f64>::new()
            .window_size(11)
            .order(4)
            .build()
            .unwrap_err();
        assert_eq!(e, Error::SgOrderTooHigh);
    }

    #[test]
    fn rejects_deriv_order_too_high() {
        let e = SgConfigBuilder::<f64>::new()
            .window_size(7)
            .order(2)
            .deriv_order(3)
            .build()
            .unwrap_err();
        assert_eq!(e, Error::SgDerivationOrderTooHigh);
    }

    #[test]
    fn rejects_bad_sample_frequency() {
        let e = SgConfigBuilder::<f64>::new()
            .sample_frequency_hz(0.0)
            .build()
            .unwrap_err();
        assert_eq!(e, Error::NonPositiveSampleFrequency);
        let e = SgConfigBuilder::<f64>::new()
            .sample_frequency_hz(f64::NAN)
            .build()
            .unwrap_err();
        assert_eq!(e, Error::NonFiniteSampleFrequency);
    }

    // ── weight — Gorry (1990) reference values ───────────────────────────────

    #[test]
    fn test_gorry_tables() {
        // Convolution weights for quadratic initial-point smoothing:
        // m=3 (window 7), t=-3 (oldest point), n=2, s=0
        let sg7_gram = [32.0, 15.0, 3.0, -4.0, -6.0, -3.0, 5.0];

        let weights = generate_weights::<f64>(3, -3, 2, 0);

        // The Gorry paper weights are often scaled for integer representation.
        // For m=3, n=2, s=0, the common denominator is 42.
        for (i, &expected) in sg7_gram.iter().enumerate() {
            let computed = weights[i] * 42.0;
            assert_relative_eq!(computed, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn gorry_derivative_weight_sum_zero() {
        // First-derivative weight must sum to zero (derivative of a constant = 0).
        let sg = SavitzkyGolayFilter::new(cfg(7, 2, 1, 1000.0));
        let sum: f64 = (0..7).map(|i| sg.weights[i]).sum();
        assert_relative_eq!(sum, 0.0, epsilon = 1e-10);
    }

    // ── Edge-padding cold start ───────────────────────────────────────────────

    #[test]
    fn cold_start_smoothing_returns_input() {
        let mut f = SavitzkyGolayFilter::new(cfg(7, 2, 0, 1000.0));
        let out = f.apply(42.0);
        assert_relative_eq!(out, 42.0, epsilon = 1e-10);
    }

    #[test]
    fn cold_start_derivative_returns_zero() {
        let mut f = SavitzkyGolayFilter::new(cfg(7, 2, 1, 1000.0));
        let out = f.apply(42.0);
        assert_relative_eq!(out, 0.0, epsilon = 1e-10);
    }

    // ── Steady-state behaviour ────────────────────────────────────────────────

    #[test]
    fn constant_signal_smoothing_unity_gain() {
        let mut f = SavitzkyGolayFilter::new(cfg(7, 2, 0, 1000.0));
        for _ in 0..50 {
            assert_relative_eq!(f.apply(5.0), 5.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn constant_signal_derivative_is_zero() {
        let mut f = SavitzkyGolayFilter::new(cfg(7, 2, 1, 1000.0));
        // After the window fills with the constant, derivative should be ~0.
        for _ in 0..50 {
            let out = f.apply(5.0);
            assert!(out.abs() < 1e-9, "expected ≈0, got {out}");
        }
    }

    #[test]
    fn linear_ramp_derivative_matches_slope() {
        // Ramp of 1 unit/sample at 1 kHz → derivative = 1000 units/s.
        let fs = 1000.0;
        let mut f = SavitzkyGolayFilter::new(cfg(7, 2, 1, fs));
        let mut last = 0.0f64;
        for i in 0..50i64 {
            last = f.apply(i as f64);
        }
        // After the window is full of a clean ramp, the derivative should be exact.
        assert_relative_eq!(last, fs, epsilon = 1.0);
    }

    // ── Reset ─────────────────────────────────────────────────────────────────

    #[test]
    fn reset_seeds_state_for_smoothing() {
        let mut f = SavitzkyGolayFilter::new(cfg(7, 2, 0, 1000.0));
        f.reset(99.0).unwrap();
        assert_relative_eq!(f.apply(99.0), 99.0, epsilon = 1e-10);
    }

    #[test]
    fn reset_seeds_state_for_derivative() {
        let mut f = SavitzkyGolayFilter::new(cfg(7, 2, 1, 1000.0));
        f.reset(50.0).unwrap();
        // Window full of 50.0; derivative should be 0.
        assert_relative_eq!(f.apply(50.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn reset_rejects_non_finite() {
        let mut f = SavitzkyGolayFilter::new(cfg(7, 2, 0, 1000.0));
        assert_eq!(f.reset(f64::NAN).unwrap_err(), Error::NonFiniteState);
        assert_eq!(f.reset(f64::INFINITY).unwrap_err(), Error::NonFiniteState);
    }

    // ── Comparison with batch filter ─────────────────────────────────────────

    #[cfg(feature = "alloc")]
    #[test]
    fn streaming_matches_batch_after_warmup() {
        // After the window is full of "real" data, the streaming output must match
        // the batch filter called with the same window slice.
        use super::*;

        let m = 3;
        let n = 2;
        let s = 1;
        let fs = 1000.0f64;
        let dt = 1.0 / fs;

        let batch_weight = generate_weights(m, 2 * m + 1, n, s);

        let mut f = SavitzkyGolayFilter::new(
            SgConfigBuilder::new()
                .window_size(2 * m + 1)
                .order(n)
                .deriv_order(s)
                .sample_frequency_hz(fs)
                .build()
                .unwrap(),
        );

        // Feed 20 samples of a ramp; from sample 7 onward the window is "real" data.
        let data: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let mut stream_out = 0.0f64;
        for &x in &data {
            stream_out = f.apply(x);
        }

        // Batch over the last 7 samples.
        let window: Vec<f64> = data[data.len() - 7..].to_vec();
        let batch_out: f64 = batch_weight
            .iter()
            .zip(&window)
            .map(|(w, x)| w * x)
            .sum::<f64>()
            / dt.powi(s);

        assert_relative_eq!(stream_out, batch_out, epsilon = 1e-10);
    }
}
