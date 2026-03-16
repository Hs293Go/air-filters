// SPDX-License-Identifier: MIT

//! Slew rate limiter and slew filter.
//!
//! Both types act on signals that change too rapidly, but with different output
//! behaviour:
//!
//! - [`SlewRateLimiter`]: the output always tracks the input, clamped to at
//!   most `max_rate` units per second.  Use when the signal is physically
//!   rate-limited (actuator commands, reference ramps).
//!
//! - [`SlewFilter`]: the output holds its previous value when the input changes
//!   too rapidly, treating the sample as corrupt.  Use when large jumps
//!   indicate bad data rather than a legitimate fast change (e.g. DShot RPM
//!   decode errors).
//!
//! The distinction maps to the hold behaviour on a rejected sample:
//! `SlewRateLimiter` advances the output to the permitted boundary
//! (first-order-hold ramp), while `SlewFilter` freezes it (zero-order hold).

use num_traits::float::FloatCore;

use crate::{Error, Filter};

// # SlewRateLimiter

/// Limits the rate of change of a signal by clamping the per-sample output
/// delta.
///
/// The output always moves toward the input but cannot advance by more than
/// `max_rate / sample_frequency_hz` per call to [`apply`](Filter::apply).  On a
/// large step the output ramps toward the target at the maximum rate and
/// catches up over subsequent samples.
///
/// > **Warning**: The maximum slew is calculated relative to the previous
/// > *admitted* input subject to slew rate limiting, not the previous *raw*
/// > input.
///
/// # Construction
///
/// ```
/// use air_filters::nonlinear::slew::SlewRateLimiter;
///
/// // Motor RPM: permit up to 200 000 erpm/s change at a 1 kHz loop
/// let limiter = SlewRateLimiter::<f32>::from_rate_hz(200_000.0, 1_000.0).unwrap();
/// assert_eq!(limiter.max_delta(), 200.0); // 200 erpm per sample
/// ```
///
/// # Behaviour
///
/// ```
/// use air_filters::nonlinear::slew::SlewRateLimiter;
/// use air_filters::Filter;
///
/// let mut lim = SlewRateLimiter::<f64>::new(1.0).unwrap(); // 1 unit/sample
/// assert_eq!(lim.apply(5.0), 1.0); // large step → clamped to max_delta
/// assert_eq!(lim.apply(5.0), 2.0); // still ramping toward 5.0
/// assert_eq!(lim.apply(2.5), 2.5); // small step → passes through
/// ```
#[derive(Debug, Copy, Clone)]
pub struct SlewRateLimiter<T: FloatCore> {
    max_delta: T,
    state: T,
}

impl<T: FloatCore> SlewRateLimiter<T> {
    /// Creates a new `SlewRateLimiter` to limit changes an input signal to
    /// `max_delta` units per sample.
    ///
    /// # Parameters
    /// - `max_delta`: maximum permitted change per sample in signal units. Must
    ///   be finite and positive.
    ///
    /// Use [`from_rate_hz`](Self::from_rate_hz) to specify the limit in terms
    /// of a per-second rate and sample frequency.
    pub fn new(max_delta: T) -> Result<Self, Error> {
        if !max_delta.is_finite() {
            return Err(Error::NonFiniteSlewLimit);
        }
        if max_delta <= t!(0) {
            return Err(Error::NonPositiveSlewLimit);
        }
        Ok(Self {
            max_delta,
            state: t!(0),
        })
    }

    /// Creates a new `SlewRateLimiter` to limit changes an input signal to
    /// `max_rate` units per second at a sample frequency of
    /// `sample_frequency_hz`.
    ///
    /// # Parameters
    /// - `max_rate`: maximum permitted change per second in signal units.  Must
    ///   be finite and positive.
    /// - `sample_frequency_hz`: the rate at which [`apply`](Filter::apply) is
    ///   called. Must be finite and positive.
    ///
    /// The per-sample delta limit is computed as `max_rate /
    /// sample_frequency_hz`.
    pub fn from_rate_hz(max_rate: T, sample_frequency_hz: T) -> Result<Self, Error> {
        if !max_rate.is_finite() {
            return Err(Error::NonFiniteSlewLimit);
        }
        if max_rate <= t!(0) {
            return Err(Error::NonPositiveSlewLimit);
        }
        if !sample_frequency_hz.is_finite() {
            return Err(Error::NonFiniteSampleFrequency);
        }
        if sample_frequency_hz <= t!(0) {
            return Err(Error::NonPositiveSampleFrequency);
        }
        Ok(Self {
            max_delta: max_rate / sample_frequency_hz,
            state: t!(0),
        })
    }

    /// Returns the value returned by the most recent call to
    /// [`apply`](Filter::apply).
    ///
    /// The initial state before the first call is zero.
    pub fn last_output(&self) -> T {
        self.state
    }

    /// Returns the per-sample delta limit (`max_rate / sample_frequency_hz`).
    pub fn max_delta(&self) -> T {
        self.max_delta
    }
}

impl<T: FloatCore> Filter<T> for SlewRateLimiter<T> {
    /// Advances the output toward `input` by at most `max_delta`, returning the
    /// new output.
    fn apply(&mut self, input: T) -> T {
        let delta = input - self.state;
        self.state = self.state + delta.clamp(-self.max_delta, self.max_delta);
        self.state
    }

    /// Seeds the output state.  Call before the first sample to avoid a
    /// cold-start ramp from zero.
    fn reset(&mut self, state: T) -> Result<(), Error> {
        if !state.is_finite() {
            return Err(Error::NonFiniteState);
        }
        self.state = state;
        Ok(())
    }
}

// # SlewFilter

/// Rejects samples whose change from the previous output exceeds a rate limit,
/// holding the last accepted value (zero-order hold on rejection).
///
/// When `|input − last_output| > max_rate / sample_frequency_hz`, the sample is
/// discarded and the output is unchanged.  Inputs within the limit pass through
/// unmodified so the filter introduces no lag on clean data.
///
/// The very first sample is **always accepted** regardless of magnitude,
/// avoiding the cold-start problem that arises when the initial state is zero
/// and the first valid reading is large.
///
/// # Construction
///
/// ```
/// use air_filters::nonlinear::slew::SlewFilter;
///
/// // DShot RPM: reject changes larger than 100 000 erpm/s at a 1 kHz loop
/// let filter = SlewFilter::<f32>::from_rate_hz(100_000.0, 1_000.0).unwrap();
/// assert_eq!(filter.max_delta(), 100.0); // 100 erpm per sample
/// ```
///
/// # Behaviour
///
/// ```
/// use air_filters::nonlinear::slew::SlewFilter;
/// use air_filters::Filter;
///
/// let mut f = SlewFilter::<f64>::new(1.0).unwrap();
/// assert_eq!(f.apply(500.0), 500.0); // cold start → accepted
/// assert_eq!(f.apply(600.0), 500.0); // |600-500| = 100 > 1 → held
/// assert_eq!(f.apply(501.0), 501.0); // |501-500| = 1 ≤ 1 → accepted
/// ```
#[derive(Debug, Copy, Clone)]
pub struct SlewFilter<T: FloatCore> {
    max_delta: T,
    /// `None` until the first sample is accepted; avoids rejecting the first
    /// reading when the motor is already spinning at arm time.
    state: Option<T>,
}

impl<T: FloatCore> SlewFilter<T> {
    /// Creates a new `SlewFilter` to reject changes in an input signal larger
    /// than `max_delta` units per sample.
    pub fn new(max_delta: T) -> Result<Self, Error> {
        if !max_delta.is_finite() {
            return Err(Error::NonFiniteSlewLimit);
        }
        if max_delta <= t!(0) {
            return Err(Error::NonPositiveSlewLimit);
        }
        Ok(Self {
            max_delta,
            state: None,
        })
    }

    /// Creates a new `SlewFilter` to reject changes in an input signal larger
    /// than `max_rate` units per second at a sample frequency of
    /// `sample_frequency_hz`.
    ///
    /// # Parameters
    /// - `max_rate`: maximum per-second change that is considered a legitimate
    ///   sample. Changes larger than `max_rate / sample_frequency_hz` per step
    ///   are rejected. Must be finite and positive.
    /// - `sample_frequency_hz`: the rate at which [`apply`](Filter::apply) is
    ///   called. Must be finite and positive.
    pub fn from_rate_hz(max_rate: T, sample_frequency_hz: T) -> Result<Self, Error> {
        if !max_rate.is_finite() {
            return Err(Error::NonFiniteSlewLimit);
        }
        if max_rate <= t!(0) {
            return Err(Error::NonPositiveSlewLimit);
        }
        if !sample_frequency_hz.is_finite() {
            return Err(Error::NonFiniteSampleFrequency);
        }
        if sample_frequency_hz <= t!(0) {
            return Err(Error::NonPositiveSampleFrequency);
        }
        Ok(Self {
            max_delta: max_rate / sample_frequency_hz,
            state: None,
        })
    }

    /// Returns the value returned by the most recent call to
    /// [`apply`](Filter::apply), or `None` if no sample has been processed
    /// yet.
    pub fn last_output(&self) -> Option<T> {
        self.state
    }

    /// Returns the per-sample rejection threshold (`max_rate /
    /// sample_frequency_hz`).
    ///
    /// Inputs that differ from the current state by more than this value are
    /// discarded.
    pub fn max_delta(&self) -> T {
        self.max_delta
    }
}

impl<T: FloatCore> Filter<T> for SlewFilter<T> {
    /// Returns `input` if it is within `max_delta` of the current state,
    /// otherwise returns the current state unchanged (zero-order hold).
    ///
    /// The first call always returns `input` and seeds the state, regardless of
    /// magnitude.
    fn apply(&mut self, input: T) -> T {
        match self.state {
            None => {
                self.state = Some(input);
                input
            }
            Some(state) => {
                if (input - state).abs() > self.max_delta {
                    state // outlier: hold previous output
                } else {
                    self.state = Some(input);
                    input
                }
            }
        }
    }

    /// Seeds the filter state, after which the next [`apply`](Filter::apply)
    /// call compares the input against `state`.  Use at arm time to prime
    /// the filter with a known-good initial reading rather than relying on
    /// cold-start acceptance.
    fn reset(&mut self, state: T) -> Result<(), Error> {
        if !state.is_finite() {
            return Err(Error::NonFiniteState);
        }
        self.state = Some(state);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ── SlewRateLimiter ───────────────────────────────────────────────────────

    #[test]
    fn rate_limiter_construction() {
        assert!(SlewRateLimiter::<f64>::new(1.0).is_ok());
        assert_eq!(SlewRateLimiter::<f64>::new(1.0).unwrap().max_delta(), 1.0);

        assert!(SlewRateLimiter::<f64>::from_rate_hz(1000.0, 1000.0).is_ok());
        assert_eq!(
            SlewRateLimiter::<f64>::from_rate_hz(1000.0, 1000.0)
                .unwrap()
                .max_delta(),
            1.0
        );
    }

    #[test]
    fn rate_limiter_rejects_invalid_construction() {
        assert_eq!(
            SlewRateLimiter::<f64>::new(0.0).unwrap_err(),
            Error::NonPositiveSlewLimit
        );
        assert_eq!(
            SlewRateLimiter::<f64>::new(-1.0).unwrap_err(),
            Error::NonPositiveSlewLimit
        );
        for invalid in [f64::NAN, f64::INFINITY] {
            assert_eq!(
                SlewRateLimiter::<f64>::new(invalid).unwrap_err(),
                Error::NonFiniteSlewLimit
            );
        }

        assert_eq!(
            SlewRateLimiter::<f64>::from_rate_hz(0.0, 1000.0).unwrap_err(),
            Error::NonPositiveSlewLimit
        );
        assert_eq!(
            SlewRateLimiter::<f64>::from_rate_hz(-1.0, 1000.0).unwrap_err(),
            Error::NonPositiveSlewLimit
        );
        for invalid in [f64::NAN, f64::INFINITY] {
            assert_eq!(
                SlewRateLimiter::<f64>::from_rate_hz(invalid, 1000.0).unwrap_err(),
                Error::NonFiniteSlewLimit
            );
        }
    }

    #[test]
    fn rate_limiter_rejects_invalid_sample_frequency() {
        assert_eq!(
            SlewRateLimiter::<f64>::from_rate_hz(1000.0, 0.0).unwrap_err(),
            Error::NonPositiveSampleFrequency
        );
        assert_eq!(
            SlewRateLimiter::<f64>::from_rate_hz(1000.0, -500.0).unwrap_err(),
            Error::NonPositiveSampleFrequency
        );
        assert_eq!(
            SlewRateLimiter::<f64>::from_rate_hz(1000.0, f64::NAN).unwrap_err(),
            Error::NonFiniteSampleFrequency
        );
    }

    #[test]
    fn rate_limiter_small_step_passes_through() {
        // max_delta = 1000 / 1000 = 1.0
        let mut lim = SlewRateLimiter::<f64>::from_rate_hz(1000.0, 1000.0).unwrap();
        lim.reset(10.0).unwrap();

        // delta = 0.5 ≤ 1.0 → passes through exactly
        assert_relative_eq!(lim.apply(10.5), 10.5);
    }

    #[test]
    fn rate_limiter_large_step_clamped() {
        // max_delta = 1.0
        let mut lim = SlewRateLimiter::<f64>::from_rate_hz(1000.0, 1000.0).unwrap();

        // From 0 → 5: clamped to max_delta each step
        assert_relative_eq!(lim.apply(5.0), 1.0);
        assert_relative_eq!(lim.apply(5.0), 2.0);
        assert_relative_eq!(lim.apply(5.0), 3.0);
    }

    #[test]
    fn rate_limiter_ramps_in_both_directions() {
        let mut lim = SlewRateLimiter::<f64>::from_rate_hz(1000.0, 1000.0).unwrap();
        lim.reset(0.0).unwrap();

        assert_relative_eq!(lim.apply(-5.0), -1.0);
        assert_relative_eq!(lim.apply(-5.0), -2.0);
    }

    #[test]
    fn rate_limiter_at_boundary_passes_through() {
        // Input exactly at max_delta from state: should not be clamped
        let mut lim = SlewRateLimiter::<f64>::from_rate_hz(1000.0, 1000.0).unwrap();
        // max_delta = 1.0; step from 0 to exactly 1.0
        assert_relative_eq!(lim.apply(1.0), 1.0);
    }

    #[test]
    fn rate_limiter_reset_seeds_state() {
        let mut lim = SlewRateLimiter::<f64>::from_rate_hz(1000.0, 1000.0).unwrap();
        lim.reset(500.0).unwrap();
        assert_relative_eq!(lim.last_output(), 500.0);

        // Small step from 500 → 500.5 passes through
        assert_relative_eq!(lim.apply(500.5), 500.5);
    }

    #[test]
    fn rate_limiter_reset_rejects_non_finite() {
        let mut lim = SlewRateLimiter::<f64>::from_rate_hz(1000.0, 1000.0).unwrap();
        assert_eq!(lim.reset(f64::NAN).unwrap_err(), Error::NonFiniteState);
        assert_eq!(lim.reset(f64::INFINITY).unwrap_err(), Error::NonFiniteState);
    }

    // ── SlewFilter ────────────────────────────────────────────────────────────

    #[test]
    fn slew_filter_construction() {
        assert!(SlewFilter::<f64>::from_rate_hz(1000.0, 1000.0).is_ok());
        assert_eq!(
            SlewFilter::<f64>::from_rate_hz(1000.0, 1000.0)
                .unwrap()
                .max_delta(),
            1.0
        );
    }

    #[test]
    fn slew_filter_rejects_invalid_construction() {
        assert_eq!(
            SlewFilter::<f64>::new(0.0).unwrap_err(),
            Error::NonPositiveSlewLimit
        );

        assert_eq!(
            SlewFilter::<f64>::new(-1.0).unwrap_err(),
            Error::NonPositiveSlewLimit
        );

        for invalid in [f64::NAN, f64::INFINITY] {
            assert_eq!(
                SlewFilter::<f64>::new(invalid).unwrap_err(),
                Error::NonFiniteSlewLimit
            );
        }

        assert_eq!(
            SlewFilter::<f64>::from_rate_hz(0.0, 1000.0).unwrap_err(),
            Error::NonPositiveSlewLimit
        );
        assert_eq!(
            SlewFilter::<f64>::from_rate_hz(-1.0, 1000.0).unwrap_err(),
            Error::NonPositiveSlewLimit
        );
        for invalid in [f64::NAN, f64::INFINITY] {
            assert_eq!(
                SlewFilter::<f64>::from_rate_hz(invalid, 1000.0).unwrap_err(),
                Error::NonFiniteSlewLimit
            );
        }
    }

    #[test]
    fn slew_filter_rejects_invalid_sample_frequency() {
        assert_eq!(
            SlewFilter::<f64>::from_rate_hz(1000.0, 0.0).unwrap_err(),
            Error::NonPositiveSampleFrequency
        );
        assert_eq!(
            SlewFilter::<f64>::from_rate_hz(1000.0, f64::NAN).unwrap_err(),
            Error::NonFiniteSampleFrequency
        );
    }

    #[test]
    fn slew_filter_cold_start_always_accepts() {
        let mut f = SlewFilter::<f64>::from_rate_hz(1.0, 1000.0).unwrap();
        // max_delta = 0.001; first reading of 15 000 should still be accepted
        assert_eq!(f.last_output(), None);
        assert_relative_eq!(f.apply(15_000.0), 15_000.0);
        assert_eq!(f.last_output(), Some(15_000.0));
    }

    #[test]
    fn slew_filter_within_limit_passes_through() {
        let mut f = SlewFilter::<f64>::from_rate_hz(1000.0, 1000.0).unwrap(); // max_delta = 1.0
        f.reset(100.0).unwrap();

        assert_relative_eq!(f.apply(100.5), 100.5); // delta = 0.5 ≤ 1.0
        assert_relative_eq!(f.apply(101.5), 101.5); // delta = 1.0 ≤ 1.0
                                                    // (boundary: accepted)
    }

    #[test]
    fn slew_filter_beyond_limit_holds() {
        let mut f = SlewFilter::<f64>::from_rate_hz(1000.0, 1000.0).unwrap(); // max_delta = 1.0
        f.reset(100.0).unwrap();

        // Outlier: delta = 50 > 1.0 → held at 100.0
        assert_relative_eq!(f.apply(150.0), 100.0);

        // State is still 100.0; next within-limit sample is accepted
        assert_relative_eq!(f.apply(100.8), 100.8);
    }

    #[test]
    fn slew_filter_hold_does_not_ramp() {
        // Verify no creep toward the rejected value over multiple outlier samples
        let mut f = SlewFilter::<f64>::from_rate_hz(1000.0, 1000.0).unwrap();
        f.reset(0.0).unwrap();

        for _ in 0..10 {
            assert_relative_eq!(f.apply(1000.0), 0.0); // held every time
        }
    }

    #[test]
    fn slew_filter_reset_primes_state() {
        let mut f = SlewFilter::<f64>::from_rate_hz(1000.0, 1000.0).unwrap();
        f.reset(500.0).unwrap();
        assert_eq!(f.last_output(), Some(500.0));

        // delta = 1000 > 1.0 → held at 500
        assert_relative_eq!(f.apply(1500.0), 500.0);
    }

    #[test]
    fn slew_filter_reset_rejects_non_finite() {
        let mut f = SlewFilter::<f64>::from_rate_hz(1000.0, 1000.0).unwrap();
        assert_eq!(f.reset(f64::NAN).unwrap_err(), Error::NonFiniteState);
        assert_eq!(f.reset(f64::INFINITY).unwrap_err(), Error::NonFiniteState);
    }
}
