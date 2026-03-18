// SPDX-License-Identifier: MIT

//! A 1st-order low-pass filter (PT1/Proportional Time-Lag) implemented using the forward Euler
//! method for discretization.
//!
//! This filter is usable in highly constrained environments where not even libm is available, and
//! is the only filter available if this crate is built with `--no-default-features`, but not
//! `--features libm`.`
//!
//! This filter is mathematically equivalent to an exponential moving average (EMA) filter.

use num_traits::{Float, FloatConst};

use crate::{
    internal::ConfigurableFilter, CommonConfigurableFilter, CommonFilterConfig, Error, Filter,
};

/// A 1st-order low-pass filter implemented using the forward Euler method for discretization.
pub struct Pt1Filter<T: Float> {
    k: T,
    state: T,
    config: CommonFilterConfig<T>,
}

impl<T: Float + FloatConst> ConfigurableFilter<T> for Pt1Filter<T> {
    fn update_configuration(&mut self) -> Result<(), Error> {
        self.k = Self::compute_gain(&self.config);
        Ok(())
    }

    fn config_mut(&mut self) -> &mut CommonFilterConfig<T> {
        &mut self.config
    }
}

impl<T: Float + FloatConst> CommonConfigurableFilter<T> for Pt1Filter<T> {
    fn config(&self) -> &CommonFilterConfig<T> {
        &self.config
    }
}

impl<T: Float + FloatConst> Pt1Filter<T> {
    fn compute_gain(config: &CommonFilterConfig<T>) -> T {
        let rc = t!(1) / (T::TAU() * config.cutoff_frequency_hz);
        let sample_time = t!(1) / config.sample_frequency_hz;
        sample_time / (rc + sample_time)
    }

    /// Creates a new Pt1Filter with the given configuration. The initial state is set to zero.
    pub fn new(config: CommonFilterConfig<T>) -> Self {
        Self {
            k: Self::compute_gain(&config),
            state: t!(0),
            config,
        }
    }

    /// Returns the current state of the filter, which represents the output of the filter at the last applied input.
    pub fn state(&self) -> T {
        self.state
    }

    /// Returns the smoothing constant `k` in the formula `s_next = s + k * (input - s)`.
    ///
    /// If the PT1 filter is viewed as an exponential moving average, then `k = 1 - alpha` where
    /// `alpha` is the smoothing factor in the EMA formula: `s_next = alpha * input + (1 - alpha) * s`.
    pub fn smoothing_constant(&self) -> T {
        self.k
    }
}

impl<T: Float> Filter<T> for Pt1Filter<T> {
    /// Applies the filter to the input sample and updates the internal state. The output is the new state.
    fn apply(&mut self, input: T) -> T {
        self.state = self.state + self.k * (input - self.state);
        self.state
    }

    /// Resets the filter state to the specified value. Returns an error if the state is not finite.
    fn reset(&mut self, state: T) -> Result<(), Error> {
        if !state.is_finite() {
            return Err(Error::NonFiniteState);
        }
        self.state = state;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::CommonFilterConfigBuilder;

    use super::*;
    use approx::assert_relative_eq;

    // Helper to create a standard config
    fn standard_config() -> CommonFilterConfig<f64> {
        CommonFilterConfigBuilder::new()
            .cutoff_frequency_hz(100.0)
            .sample_frequency_hz(1000.0)
            .build()
            .unwrap()
    }

    #[test]
    fn test_initialization() {
        let config = standard_config();
        let filter = Pt1Filter::new(config);

        // state should start at zero
        assert_eq!(filter.state(), 0.0);

        // k calculation check:
        // RC = 1 / (2 * PI * 100) ≈ 0.0015915
        // dT = 1 / 1000 = 0.001
        // k = 0.001 / (0.0015915 + 0.001) ≈ 0.385869
        assert_relative_eq!(filter.smoothing_constant(), 0.385869, epsilon = 1e-6);
    }

    #[test]
    fn test_apply_step_response() {
        let mut filter = Pt1Filter::new(standard_config());
        let k = filter.smoothing_constant();

        // Step input of 1.0
        // y[1] = 0 + k * (1.0 - 0) = k
        let out1 = filter.apply(1.0);
        assert_relative_eq!(out1, k);

        // y[2] = y[1] + k * (1.0 - y[1])
        let out2 = filter.apply(1.0);
        let expected2 = out1 + k * (1.0 - out1);
        assert_relative_eq!(out2, expected2);
    }

    #[test]
    fn test_reset() {
        let mut filter = Pt1Filter::new(standard_config());
        filter.apply(100.0); // Move state away from zero

        // Reset to 50.0
        filter.reset(50.0).expect("Reset should succeed");
        assert_eq!(filter.state(), 50.0);

        // Test non-finite reset
        let err = filter.reset(f64::NAN).unwrap_err();
        assert_eq!(err, Error::NonFiniteState);
    }

    #[test]
    fn test_dynamic_cutoff_update() {
        let mut filter = Pt1Filter::new(standard_config());
        let initial_k = filter.smoothing_constant();

        // Increase cutoff: filter should become more "responsive" (higher k)
        filter.set_cutoff_frequency_hz(200.0).unwrap();
        assert!(filter.smoothing_constant() > initial_k);
        assert_eq!(filter.cutoff_frequency_hz(), 200.0);

        // Test error handling in setter
        let err = filter.set_cutoff_frequency_hz(-10.0).unwrap_err();
        assert_eq!(err, Error::NonPositiveCutoffFrequency);
    }

    #[test]
    fn test_dynamic_sample_rate_update() {
        let mut filter = Pt1Filter::new(standard_config());
        let initial_k = filter.smoothing_constant();

        // Increase sample rate: filter should become "slower" per sample (lower k)
        filter.set_sample_frequency_hz(2000.0).unwrap();
        assert!(filter.smoothing_constant() < initial_k);
        assert_eq!(filter.sample_frequency_hz(), 2000.0);
    }

    #[test]
    fn test_dc_gain_unity() {
        let mut filter = Pt1Filter::new(standard_config());

        // After many samples of a constant input, state should converge to input
        let input = 123.45;
        for _ in 0..100 {
            filter.apply(input);
        }

        assert_relative_eq!(filter.state(), input, epsilon = 1e-10);
    }

    // --- Nyquist tolerance tests ---
    // FO filters degrade gracefully above Nyquist (k stays in (0,1); filter remains stable).
    // No error is returned — enforcement is intentionally absent for this filter family.

    #[test]
    fn test_frequency_response_at_cutoff() {
        // Use a high f_s/f_c ratio (1000:1) so that the Euler-discretisation
        // error is < 0.2 %, keeping the measured gain within 1 % of -3 dB.
        let config = CommonFilterConfigBuilder::new()
            .cutoff_frequency_hz(100.0)
            .sample_frequency_hz(100_000.0)
            .build()
            .unwrap();
        let mut filter = Pt1Filter::new(config);

        let omega = 2.0 * core::f64::consts::PI * 100.0 / 100_000.0;
        let n_settle = 10_000; // > 60 time constants
        let n_measure = 2_000; // 2 full periods — peak is within 5e-6 of true amplitude

        for i in 0..n_settle {
            filter.apply((omega * i as f64).sin());
        }
        let mut peak = 0.0_f64;
        for i in n_settle..n_settle + n_measure {
            let out = filter.apply((omega * i as f64).sin()).abs();
            if out > peak {
                peak = out;
            }
        }

        // Gain at the cutoff frequency must be -3 dB (1/√2 ≈ 0.7071)
        assert_relative_eq!(peak, core::f64::consts::FRAC_1_SQRT_2, epsilon = 0.01);
    }

    #[test]
    fn test_nyquist_tolerance_build() {
        // f_c = 0.6 * f_s, well above Nyquist — must build without error
        assert!(CommonFilterConfigBuilder::new()
            .cutoff_frequency_hz(600.0)
            .sample_frequency_hz(1000.0)
            .build()
            .is_ok());
    }

    #[test]
    fn test_nyquist_tolerance_set_cutoff() {
        let mut filter = Pt1Filter::new(standard_config()); // 100Hz / 1000Hz

        // Raise cutoff past Nyquist — must succeed
        assert!(filter.set_cutoff_frequency_hz(600.0).is_ok());

        // k must remain in (0, 1): the filter is unconditionally stable
        let k = filter.smoothing_constant();
        assert!(k > 0.0 && k < 1.0, "k={k} is outside (0, 1)");
    }

    #[test]
    fn test_nyquist_tolerance_stability() {
        // Build directly with f_c above Nyquist
        let config = CommonFilterConfigBuilder::new()
            .cutoff_frequency_hz(600.0)
            .sample_frequency_hz(1000.0)
            .build()
            .unwrap();
        let mut filter = Pt1Filter::new(config);

        // Filter must still converge to a DC input
        let input = 1.0;
        let mut last_out = 0.0;
        for _ in 0..500 {
            last_out = filter.apply(input);
        }
        assert_relative_eq!(last_out, input, epsilon = 1e-6);
    }
}
