//! A second-order low-pass filter implemented as a cascade of two first-order filters.
use num_traits::{float::FloatCore, real::Real, FloatConst};

use crate::{
    internal::ConfigurableFilter, CommonConfigurableFilter, CommonFilterConfig, Error, Filter,
    FilterContext, FuncFilter,
};

/// A second-order low-pass filter implemented as a cascade of two first-order filters.
pub struct Pt2Filter<T: FloatCore + Real> {
    k: T,
    state: T,
    state1: T,
    config: CommonFilterConfig<T>,
}

impl<T: FloatCore + Real + FloatConst> Pt2Filter<T> {
    fn compute_gain(config: &CommonFilterConfig<T>) -> T {
        const ORDER: i32 = 2;
        let order_cutoff_correction = t!(1) / (t!(2).powf(t!(1) / t!(ORDER)) - t!(1)).sqrt();
        let rc = t!(1) / (order_cutoff_correction * T::TAU() * config.cutoff_frequency_hz);
        let sample_time = t!(1) / config.sample_frequency_hz;
        sample_time / (rc + sample_time)
    }

    /// Creates a new Pt2Filter with the given configuration. The initial states are set to zero.
    pub fn new(config: CommonFilterConfig<T>) -> Self {
        Self {
            k: Self::compute_gain(&config),
            state: t!(0),
            state1: t!(0),
            config,
        }
    }

    /// Returns the value returned by the most recent call to `apply`.
    ///
    /// # Details
    /// This is the output of the second (output-side) stage in the cascade.
    pub fn last_output(&self) -> T {
        self.state
    }

    /// Returns the smoothing constant `k`.
    pub fn smoothing_constant(&self) -> T {
        self.k
    }
}

impl<T: FloatCore + Real + FloatConst> ConfigurableFilter<T> for Pt2Filter<T> {
    fn update_configuration(&mut self) -> Result<(), Error> {
        self.k = Self::compute_gain(&self.config);
        Ok(())
    }

    fn config_mut(&mut self) -> &mut CommonFilterConfig<T> {
        &mut self.config
    }
}

impl<T: FloatCore + Real + FloatConst> CommonConfigurableFilter<T> for Pt2Filter<T> {
    fn config(&self) -> &CommonFilterConfig<T> {
        &self.config
    }
}

impl<T: FloatCore + Real> Filter<T> for Pt2Filter<T> {
    fn apply(&mut self, input: T) -> T {
        self.state1 = self.state1 + self.k * (input - self.state1);
        self.state = self.state + self.k * (self.state1 - self.state);
        self.state
    }

    fn reset(&mut self, steady_output: T) -> Result<(), Error> {
        if !steady_output.is_finite() {
            return Err(Error::NonFiniteState);
        }
        self.state = steady_output;
        self.state1 = steady_output;
        Ok(())
    }
}

/// Container for the internal states of [`Pt2Filter`], used for stateless operation through
/// [`Pt2Filter::apply_stateless`].
#[derive(Debug, Copy, Clone)]
pub struct Pt2FilterContext<T: FloatCore + Real> {
    state: T,  // The output of the second stage and the final output of the filter.
    state1: T, // The output of the first stage and the intermediate state between the two stages.
}

impl<T: FloatCore + Real> Default for Pt2FilterContext<T> {
    /// Returns a zero-initialised context, suitable for a cold start.
    fn default() -> Self {
        Self {
            state: t!(0),
            state1: t!(0),
        }
    }
}

impl<T: FloatCore + Real> FilterContext<T> for Pt2FilterContext<T> {
    fn reset(&mut self, value: T) -> Result<(), Error> {
        if !value.is_finite() {
            return Err(Error::NonFiniteState);
        }
        self.state = value;
        self.state1 = value;
        Ok(())
    }

    fn last_output(&self) -> T {
        self.state
    }
}

impl<T: FloatCore + Real> FuncFilter<T> for Pt2Filter<T> {
    type Context = Pt2FilterContext<T>;

    fn apply_stateless(&self, input: T, context: &Self::Context) -> (T, Self::Context) {
        let state1 = context.state1 + self.k * (input - context.state1);
        let state = context.state + self.k * (state1 - context.state);
        (state, Pt2FilterContext { state1, state })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{iir::testing, CommonFilterConfigBuilder};

    use super::*;

    fn standard_config() -> CommonFilterConfig<f64> {
        CommonFilterConfigBuilder::new()
            .cutoff_frequency_hz(100.0)
            .sample_frequency_hz(1000.0)
            .build()
            .unwrap()
    }

    #[test]
    fn test_initialization() {
        let filter = Pt2Filter::new(standard_config());

        // Initial states should be zero
        assert_eq!(filter.state, 0.0);
        assert_eq!(filter.state1, 0.0);

        // Expected k calculation for PT2:
        // correction = 1 / sqrt(2^(1/2) - 1) ≈ 1.553773974
        // rc = 1 / (1.553773974 * TAU * 100) ≈ 0.0010243
        // dT = 0.001
        // k = 0.001 / (0.0010243 + 0.001) ≈ 0.49399
        assert_relative_eq!(filter.smoothing_constant(), 0.49399, epsilon = 1e-5);
    }

    #[test]
    fn test_apply_step_response() {
        let mut filter = Pt2Filter::new(standard_config());
        let k = filter.smoothing_constant();

        // Step input of 1.0
        // s1 = 0 + k*(1-0) = k
        // s2 = 0 + k*(s1-0) = k*k
        let out1 = filter.apply(1.0);
        assert_relative_eq!(out1, k * k);

        // Second iteration
        // s1_next = k + k*(1-k)
        // s2_next = k*k + k*(s1_next - k*k)
        let s1_next = k + k * (1.0 - k);
        let expected_out2 = (k * k) + k * (s1_next - (k * k));
        let out2 = filter.apply(1.0);
        assert_relative_eq!(out2, expected_out2);
    }

    #[test]
    fn test_reset() {
        let mut filter = Pt2Filter::new(standard_config());
        filter.apply(100.0);

        filter.reset(25.0).unwrap();
        assert_eq!(filter.last_output(), 25.0);
        assert_eq!(filter.state1, 25.0);

        // Steady state check: apply 25.0 again, output should remain 25.0
        assert_relative_eq!(filter.apply(25.0), 25.0);

        // Reset to non-finite value should return an error and not change the state
        assert_eq!(
            filter.reset(f64::INFINITY).unwrap_err(),
            Error::NonFiniteState
        );
        assert_relative_eq!(filter.apply(25.0), 25.0);
    }

    #[test]
    fn test_dynamic_cutoff_update() {
        let mut filter = Pt2Filter::new(standard_config());
        let k_low = filter.smoothing_constant();

        // Increase cutoff -> responsiveness (k) must increase
        filter.set_cutoff_frequency_hz(200.0).unwrap();
        assert!(filter.smoothing_constant() > k_low);

        // Test error handling in setter
        let err = filter.set_cutoff_frequency_hz(-10.0).unwrap_err();
        assert_eq!(err, Error::NonPositiveCutoffFrequency);
    }

    #[test]
    fn test_dynamic_sample_rate_update() {
        let mut filter = Pt2Filter::new(standard_config());
        let initial_k = filter.smoothing_constant();

        // Increase sample rate: filter should become "slower" per sample (lower k)
        filter.set_sample_frequency_hz(2000.0).unwrap();
        assert!(filter.smoothing_constant() < initial_k);
        assert_eq!(filter.sample_frequency_hz(), 2000.0);
    }

    #[test]
    fn test_dc_gain_unity() {
        let filter = Pt2Filter::new(standard_config());
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
        testing::test_frequency_response_at_cutoff(Pt2Filter::new(config));
    }

    // --- Nyquist tolerance tests ---

    #[test]
    fn test_nyquist_tolerance_set_cutoff() {
        let mut filter = Pt2Filter::new(standard_config());

        assert!(filter.set_cutoff_frequency_hz(600.0).is_ok());

        let k = filter.smoothing_constant();
        assert!(k > 0.0 && k < 1.0, "k={k} is outside (0, 1)");
    }

    #[test]
    fn test_nyquist_tolerance_stability() {
        let config = CommonFilterConfigBuilder::new()
            .cutoff_frequency_hz(600.0)
            .sample_frequency_hz(1000.0)
            .build()
            .unwrap();
        testing::check_convergence_to_steady_state_input(Pt2Filter::new(config), 1.0);
    }

    #[test]
    fn test_functional_stateful_equivalence() {
        let config = standard_config();
        testing::test_functional_stateful_equivalence(
            Pt2Filter::new(config),
            Pt2Filter::new(config),
            Pt2FilterContext::default(),
        );
    }

    #[test]
    fn test_stateless_context_independence() {
        testing::test_stateless_context_independence(
            Pt2Filter::new(standard_config()),
            Pt2FilterContext::default(),
        );
    }

    #[test]
    fn test_stateless_reset() {
        testing::test_stateless_reset(
            Pt2Filter::new(standard_config()),
            Pt2FilterContext::default(),
        );
    }
}
