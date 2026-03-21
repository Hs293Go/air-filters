// SPDX-License-Identifier: MIT

//! A 1st-order low-pass filter (PT1/Proportional Time-Lag) implemented using the forward Euler
//! method for discretization.
//!
//! This filter is usable in highly constrained environments where not even libm is available, and
//! is the only filter available if this crate is built with `--no-default-features`, but not
//! `--features libm`.`
//!
//! This filter is mathematically equivalent to an exponential moving average (EMA) filter.

use num_traits::{float::FloatCore, FloatConst};

use crate::{
    internal::ConfigurableFilter, CommonConfigurableFilter, CommonFilterConfig, Error, Filter,
    FilterContext, FuncFilter,
};

/// A 1st-order low-pass filter implemented using the forward Euler method for discretization.
pub struct Pt1Filter<T: FloatCore> {
    k: T,
    state: T,
    config: CommonFilterConfig<T>,
}

impl<T: FloatCore + FloatConst> ConfigurableFilter<T> for Pt1Filter<T> {
    fn update_configuration(&mut self) -> Result<(), Error> {
        self.k = Self::compute_gain(&self.config);
        Ok(())
    }

    fn config_mut(&mut self) -> &mut CommonFilterConfig<T> {
        &mut self.config
    }
}

impl<T: FloatCore + FloatConst> CommonConfigurableFilter<T> for Pt1Filter<T> {
    fn config(&self) -> &CommonFilterConfig<T> {
        &self.config
    }
}

impl<T: FloatCore + FloatConst> Pt1Filter<T> {
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

    /// Returns the value returned by the most recent call to `apply`, and also the only internal
    /// state of the filter.
    pub fn last_output(&self) -> T {
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

impl<T: FloatCore> Filter<T> for Pt1Filter<T> {
    fn apply(&mut self, input: T) -> T {
        self.state = self.state + self.k * (input - self.state);
        self.state
    }

    fn reset(&mut self, steady_output: T) -> Result<(), Error> {
        if !steady_output.is_finite() {
            return Err(Error::NonFiniteState);
        }
        self.state = steady_output;
        Ok(())
    }
}

/// Container for the internal states of [`Pt1Filter`], used for stateless operation through
/// [`Pt1Filter::apply_stateless`].
#[derive(Debug, Clone, Copy)]
pub struct Pt1FilterContext<T: FloatCore>(T);

impl<T: FloatCore> Default for Pt1FilterContext<T> {
    /// Returns a zero-initialised context, suitable for a cold start.
    fn default() -> Self {
        Self(t!(0))
    }
}

impl<T: FloatCore> FilterContext<T> for Pt1FilterContext<T> {
    fn reset(&mut self, steady_output: T) -> Result<(), Error> {
        if !steady_output.is_finite() {
            return Err(Error::NonFiniteState);
        }
        self.0 = steady_output;
        Ok(())
    }

    fn last_output(&self) -> T {
        self.0
    }
}

impl<T: FloatCore> FuncFilter<T> for Pt1Filter<T> {
    type Context = Pt1FilterContext<T>;

    fn apply_stateless(&self, input: T, context: &Self::Context) -> (T, Self::Context) {
        let state = context.0 + self.k * (input - context.0);
        (state, Pt1FilterContext(state))
    }
}

#[cfg(test)]
mod tests {
    use crate::{iir::testing, CommonFilterConfigBuilder};

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
        assert_eq!(filter.last_output(), 0.0);

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
        assert_eq!(filter.last_output(), 50.0);

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

        filter
            .set_sample_loop_time(core::time::Duration::from_micros(500))
            .unwrap();
        assert!(filter.smoothing_constant() < initial_k);
        assert_eq!(filter.sample_frequency_hz(), 2000.0); // sample frequency should not change when setting loop time
    }

    #[test]
    fn test_dc_gain_unity() {
        let filter = Pt1Filter::new(standard_config());
        // After many samples of a constant input, state should converge to input
        let input = 123.45;
        testing::check_convergence_to_steady_state_input(filter, input);
    }

    #[test]
    fn test_frequency_response_at_cutoff() {
        let config = CommonFilterConfigBuilder::new()
            .cutoff_frequency_hz(100.0)
            .sample_frequency_hz(100_000.0)
            .build()
            .unwrap();
        testing::test_frequency_response_at_cutoff(Pt1Filter::new(config));
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
        testing::check_convergence_to_steady_state_input(Pt1Filter::new(config), 1.0);
    }

    #[test]
    fn test_functional_stateful_equivalence() {
        let config = standard_config();
        testing::test_functional_stateful_equivalence(
            Pt1Filter::new(config),
            Pt1Filter::new(config),
            Pt1FilterContext::default(),
        );
    }

    #[test]
    fn test_stateless_context_independence() {
        testing::test_stateless_context_independence(
            Pt1Filter::new(standard_config()),
            Pt1FilterContext::default(),
        );
    }

    #[test]
    fn test_stateless_reset() {
        testing::test_stateless_reset(
            Pt1Filter::new(standard_config()),
            Pt1FilterContext::default(),
        );
    }
}
