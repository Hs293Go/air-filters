// SPDX-License-Identifier: MIT

//! Streaming median filter for despiking signals with isolated outliers.
//!
//! Betaflight uses a short (typically length-5) median on decoded ESC
//! telemetry so a single bad DShot frame cannot propagate into downstream
//! RPM filtering.  This module provides the same shape of filter as a
//! reusable building block: a fixed, const-generic, odd window `W` whose
//! middle element (after per-sample sort) is the filter's output.
//!
//! The median is complementary to [`SlewFilter`](super::slew::SlewFilter).
//! `SlewFilter` rejects a single sample that is too far from the last
//! *output*; the median filter rejects a minority of samples that disagree
//! with the majority of the *window*.  A median survives a streak of
//! corrupted samples up to `(W − 1) / 2` long, at the cost of `(W − 1) / 2`
//! samples of delay on legitimate step transitions.

use num_traits::float::FloatCore;

use crate::util::ring_buf::RingBuf;
use crate::{Error, Filter};

/// Streaming median filter with a const-generic odd window size `W`.
///
/// On each [`apply`](Filter::apply), the new sample is pushed onto a
/// length-`W` ring buffer and the middle element of the sorted window is
/// returned.  An isolated spike (up to `(W − 1) / 2` consecutive corrupted
/// samples) is rejected without the blurring that a mean filter produces
/// on clean step transitions.
///
/// `W` must be positive and odd; both are enforced at compile time.  An
/// even or zero `W` fails to compile the first time `new()` is instantiated
/// for that `W`.
///
/// # Priming
///
/// The first sample fills every slot with that value so the filter starts
/// in steady state with no transient from zero.  Call
/// [`reset`](Filter::reset) at arm-time to seed with a known-good reading
/// rather than relying on cold-start fill.
///
/// # Complexity
///
/// `O(W²)` per sample — a fresh insertion sort of a scratch copy.
/// Intended for small `W` (typically 3, 5, 7, or 9 for ESC telemetry
/// despiking).  NaN inputs are not ordered and will corrupt the median
/// when fed through [`apply`](Filter::apply); callers should strip them
/// upstream or use [`apply_checked`](MedianFilter::apply_checked), which
/// drops non-finite samples and leaves the window untouched.
///
/// # Example
///
/// ```
/// use air_filters::nonlinear::median::MedianFilter;
/// use air_filters::Filter;
///
/// let mut f: MedianFilter<f64, 5> = MedianFilter::new();
/// assert_eq!(f.apply(100.0), 100.0); // primes: window = [100; 5]
/// assert_eq!(f.apply(101.0), 100.0); // [100,100,100,100,101] → 100
/// assert_eq!(f.apply(102.0), 100.0); // [100,100,100,101,102] → 100
/// assert_eq!(f.apply(103.0), 101.0); // [100,100,101,102,103] → 101
/// assert_eq!(f.apply(1e6),   102.0); // spike: [100,101,102,103,1e6] → 102
/// ```
#[derive(Debug, Clone)]
pub struct MedianFilter<T: FloatCore, const W: usize> {
    buf: RingBuf<T, W>,
}

impl<T: FloatCore, const W: usize> MedianFilter<T, W> {
    /// Creates a new median filter with an unprimed window.
    ///
    /// The first call to [`apply`](Filter::apply) fills every slot with
    /// that input so the filter starts in steady state.
    pub fn new() -> Self {
        const { assert!(W > 0, "MedianFilter window size must be positive") };
        const { assert!(W % 2 == 1, "MedianFilter window size must be odd") };
        Self {
            buf: RingBuf::<T, W>::new_empty(W, t!(0)),
        }
    }

    /// Returns the window size `W`.
    pub const fn window_size(&self) -> usize {
        W
    }

    /// Like [`apply`](Filter::apply) but drops non-finite inputs.
    ///
    /// Returns `None` if `input` is NaN or infinite; the window is left
    /// untouched, so the next finite sample sees the same context as if
    /// the bad sample had never arrived.  Returns `Some(median)` for
    /// finite inputs, priming the window on the first accepted sample
    /// exactly as [`apply`](Filter::apply) does.
    pub fn apply_checked(&mut self, input: T) -> Option<T> {
        if !input.is_finite() {
            return None;
        }
        Some(self.apply(input))
    }
}

impl<T: FloatCore, const W: usize> Default for MedianFilter<T, W> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatCore, const W: usize> Filter<T> for MedianFilter<T, W> {
    /// Pushes one sample and returns the median of the current window.
    ///
    /// On the first call the window is edge-filled with `input`, so the
    /// returned value equals `input` itself.
    fn apply(&mut self, input: T) -> T {
        if !self.buf.is_primed() {
            self.buf.fill(input);
        } else {
            self.buf.push_back(input);
        }

        let mut scratch: [T; W] = [t!(0); W];
        for (i, slot) in scratch.iter_mut().enumerate() {
            *slot = self.buf.get(i);
        }

        for i in 1..W {
            let mut j = i;
            while j > 0 && scratch[j - 1] > scratch[j] {
                scratch.swap(j - 1, j);
                j -= 1;
            }
        }

        scratch[W / 2]
    }

    /// Seeds every slot with `steady_output` so the next `apply` sees a
    /// pre-filled window.
    fn reset(&mut self, steady_output: T) -> Result<(), Error> {
        if !steady_output.is_finite() {
            return Err(Error::NonFiniteState);
        }
        self.buf.fill(steady_output);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_start_primes_with_input() {
        let mut f: MedianFilter<f64, 5> = MedianFilter::new();
        assert_eq!(f.apply(42.0), 42.0);
    }

    #[test]
    fn step_passthrough_after_fill() {
        let mut f: MedianFilter<f64, 3> = MedianFilter::new();
        assert_eq!(f.apply(5.0), 5.0); // fill: [5,5,5]
        assert_eq!(f.apply(7.0), 5.0); // [5,5,7] → 5
        assert_eq!(f.apply(7.0), 7.0); // [5,7,7] → 7
        assert_eq!(f.apply(7.0), 7.0); // [7,7,7] → 7
    }

    #[test]
    fn single_spike_rejected_w3() {
        let mut f: MedianFilter<f64, 3> = MedianFilter::new();
        f.apply(10.0); // fill
        f.apply(10.0);
        assert_eq!(f.apply(1e6), 10.0, "single spike outvoted");
        assert_eq!(f.apply(10.0), 10.0, "spike not remembered");
    }

    #[test]
    fn consecutive_spikes_at_window_half_rejected_w5() {
        // W=5 tolerates up to 2 consecutive corrupted samples.
        let mut f: MedianFilter<f64, 5> = MedianFilter::new();
        f.apply(10.0); // fill: [10;5]
        assert_eq!(f.apply(1e6), 10.0, "1st spike outvoted 4-1");
        assert_eq!(f.apply(1e6), 10.0, "2nd spike: [10,10,10,1e6,1e6] → 10");
    }

    #[test]
    fn three_consecutive_spikes_break_through_w5() {
        // Past (W-1)/2 = 2, spikes become the majority and dominate.
        let mut f: MedianFilter<f64, 5> = MedianFilter::new();
        f.apply(10.0); // fill: [10;5]
        f.apply(1e6);
        f.apply(1e6);
        assert_eq!(f.apply(1e6), 1e6, "majority switched to spikes");
    }

    #[test]
    fn two_of_three_window_majority_rules() {
        let mut f: MedianFilter<f64, 3> = MedianFilter::new();
        f.apply(0.0); // fill: [0,0,0]
        assert_eq!(f.apply(100.0), 0.0, "[0,0,100] → 0");
        assert_eq!(f.apply(100.0), 100.0, "[0,100,100] → 100");
    }

    #[test]
    fn monotonic_input_median_tracks_centre() {
        let mut f: MedianFilter<f64, 5> = MedianFilter::new();
        f.apply(1.0); // fill: [1;5]
        f.apply(2.0); // [1,1,1,1,2] → 1
        f.apply(3.0); // [1,1,1,2,3] → 1
        f.apply(4.0); // [1,1,2,3,4] → 2
        assert_eq!(f.apply(5.0), 3.0); // [1,2,3,4,5] → 3
        assert_eq!(f.apply(6.0), 4.0); // [2,3,4,5,6] → 4
    }

    #[test]
    fn reset_seeds_window() {
        let mut f: MedianFilter<f64, 5> = MedianFilter::new();
        f.apply(0.0);
        assert!(f.reset(50.0).is_ok());
        assert_eq!(f.apply(50.0), 50.0);
        assert_eq!(f.apply(1e6), 50.0, "spike against reset-filled window");
    }

    #[test]
    fn reset_rejects_non_finite() {
        let mut f: MedianFilter<f64, 3> = MedianFilter::new();
        assert_eq!(f.reset(f64::NAN).unwrap_err(), Error::NonFiniteState);
        assert_eq!(f.reset(f64::INFINITY).unwrap_err(), Error::NonFiniteState);
        assert_eq!(
            f.reset(f64::NEG_INFINITY).unwrap_err(),
            Error::NonFiniteState
        );
    }

    #[test]
    fn window_size_accessor() {
        let f: MedianFilter<f64, 7> = MedianFilter::new();
        assert_eq!(f.window_size(), 7);
    }

    #[test]
    fn apply_checked_drops_non_finite_and_preserves_window() {
        let mut f: MedianFilter<f64, 3> = MedianFilter::new();
        f.apply(10.0); // fill: [10,10,10]
        assert_eq!(f.apply_checked(f64::NAN), None);
        assert_eq!(f.apply_checked(f64::INFINITY), None);
        assert_eq!(f.apply_checked(f64::NEG_INFINITY), None);
        // Window still [10,10,10] — drops did not push.
        assert_eq!(f.apply_checked(11.0), Some(10.0)); // [10,10,11] → 10
        assert_eq!(f.apply_checked(11.0), Some(11.0)); // [10,11,11] → 11
    }

    #[test]
    fn apply_checked_leaves_unprimed_when_first_sample_non_finite() {
        let mut f: MedianFilter<f64, 3> = MedianFilter::new();
        assert_eq!(f.apply_checked(f64::NAN), None);
        // Next finite call should prime, not average with a poisoned slot.
        assert_eq!(f.apply_checked(5.0), Some(5.0));
        assert_eq!(f.apply_checked(6.0), Some(5.0)); // [5,5,6] → 5
    }

    #[test]
    fn w1_is_identity() {
        let mut f: MedianFilter<f64, 1> = MedianFilter::new();
        assert_eq!(f.apply(3.0), 3.0);
        assert_eq!(f.apply(7.0), 7.0);
        assert_eq!(f.apply(-2.0), -2.0);
    }
}
