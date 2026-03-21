//! A third-order low-pass filter implemented as a cascade of three first-order filters.
use num_traits::{float::FloatCore, real::Real, FloatConst};

use crate::{
    internal::ConfigurableFilter, CommonConfigurableFilter, CommonFilterConfig, Error, Filter,
    FilterContext, FuncFilter,
};

/// A third-order low-pass filter implemented as a cascade of three first-order filters.
pub struct Pt3Filter<T: FloatCore + Real> {
    k: T,
    state: T,
    state1: T,
    state2: T,
    config: CommonFilterConfig<T>,
}

impl<T: FloatCore + Real + FloatConst> Pt3Filter<T> {
    fn compute_gain(config: &CommonFilterConfig<T>) -> T {
        const ORDER: i32 = 3;
        let order_cutoff_correction = t!(1) / (t!(2).powf(t!(1) / t!(ORDER)) - t!(1)).sqrt();
        let rc = t!(1) / (order_cutoff_correction * T::TAU() * config.cutoff_frequency_hz);
        let sample_time = t!(1) / config.sample_frequency_hz;
        sample_time / (rc + sample_time)
    }

    /// Creates a new Pt3Filter with the given configuration. The initial states are set to zero.
    pub fn new(config: CommonFilterConfig<T>) -> Self {
        Self {
            k: Self::compute_gain(&config),
            state: t!(0),
            state1: t!(0),
            state2: t!(0),
            config,
        }
    }

    /// Returns the value returned by the most recent call to `apply`.
    ///
    /// # Details
    /// This is the output of the third (output-side) stage in the cascade.
    pub fn last_output(&self) -> T {
        self.state
    }

    /// Returns the smoothing constant `k`.
    pub fn smoothing_constant(&self) -> T {
        self.k
    }
}

impl<T: FloatCore + Real + FloatConst> ConfigurableFilter<T> for Pt3Filter<T> {
    fn update_configuration(&mut self) -> Result<(), Error> {
        self.k = Self::compute_gain(&self.config);
        Ok(())
    }

    fn config_mut(&mut self) -> &mut CommonFilterConfig<T> {
        &mut self.config
    }
}

impl<T: FloatCore + Real + FloatConst> CommonConfigurableFilter<T> for Pt3Filter<T> {
    fn config(&self) -> &CommonFilterConfig<T> {
        &self.config
    }
}

impl<T: FloatCore + Real> Filter<T> for Pt3Filter<T> {
    /// Applies the filter to the input sample and updates the internal states. The output is the new state.
    fn apply(&mut self, input: T) -> T {
        self.state1 = self.state1 + self.k * (input - self.state1);
        self.state2 = self.state2 + self.k * (self.state1 - self.state2);
        self.state = self.state + self.k * (self.state2 - self.state);
        self.state
    }

    fn reset(&mut self, steady_output: T) -> Result<(), Error> {
        if !steady_output.is_finite() {
            return Err(Error::NonFiniteState);
        }
        self.state = steady_output;
        self.state1 = steady_output;
        self.state2 = steady_output;
        Ok(())
    }
}

/// Container for the internal states of a [`Pt3Filter`], used for stateless application.
///
/// Each field corresponds to one stage of the three-stage cascade, with `state1` closest
/// to the input and `state` being the final output stage.
#[derive(Debug, Copy, Clone)]
pub struct Pt3FilterContext<T: FloatCore + Real> {
    /// State of the first (input-side) cascade stage.
    state1: T,
    /// State of the second (middle) cascade stage.
    state2: T,
    /// State of the third (output-side) cascade stage.
    state: T,
}

impl<T: FloatCore + Real> Default for Pt3FilterContext<T> {
    /// Returns a zero-initialised context, suitable for a cold start.
    fn default() -> Self {
        Self {
            state1: t!(0),
            state2: t!(0),
            state: t!(0),
        }
    }
}

impl<T: FloatCore + Real> FilterContext<T> for Pt3FilterContext<T> {
    fn reset(&mut self, steady_output: T) -> Result<(), Error> {
        if !steady_output.is_finite() {
            return Err(Error::NonFiniteState);
        }
        self.state1 = steady_output;
        self.state2 = steady_output;
        self.state = steady_output;
        Ok(())
    }

    fn last_output(&self) -> T {
        self.state
    }
}

impl<T: FloatCore + Real> FuncFilter<T> for Pt3Filter<T> {
    type Context = Pt3FilterContext<T>;

    /// Applies the filter to `input` without mutating internal state, threading the stage states
    /// through [`Pt3FilterContext`] instead. The cascade is evaluated sequentially — each stage
    /// receives the updated output of the preceding stage — matching the behaviour of the
    /// stateful [`Filter::apply`].
    fn apply_stateless(&self, input: T, ctx: &Self::Context) -> (T, Self::Context) {
        let state1 = ctx.state1 + self.k * (input - ctx.state1);
        let state2 = ctx.state2 + self.k * (state1 - ctx.state2);
        let state = ctx.state + self.k * (state2 - ctx.state);
        (
            state,
            Pt3FilterContext {
                state1,
                state2,
                state,
            },
        )
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
        let filter = Pt3Filter::new(standard_config());

        // Expected k calculation for PT3:
        // correction = 1 / sqrt(2^(1/3) - 1) ≈ 1.961459
        // rc = 1 / (1.961459 * TAU * 100) ≈ 0.0008114
        // dT = 0.001
        // k = 0.001 / (0.0008114 + 0.001) ≈ 0.55206
        assert_relative_eq!(filter.smoothing_constant(), 0.55206, epsilon = 1e-5);
    }

    #[test]
    fn test_apply_step_response() {
        let mut filter = Pt3Filter::new(standard_config());
        let k = filter.smoothing_constant();

        // Step input of 1.0
        // s1 = k, s2 = k*k, s3 = k*k*k
        let out1 = filter.apply(1.0);
        assert_relative_eq!(out1, k * k * k);
    }

    #[test]
    fn test_reset() {
        let mut filter = Pt3Filter::new(standard_config());
        filter.apply(100.0);

        filter.reset(25.0).unwrap();
        assert_eq!(filter.last_output(), 25.0);
        // All three delay stages must be initialised to the reset value.
        assert_eq!(filter.state1, 25.0);
        assert_eq!(filter.state2, 25.0);

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
        let mut filter = Pt3Filter::new(standard_config());
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
        let mut filter = Pt3Filter::new(standard_config());
        let initial_k = filter.smoothing_constant();

        // Increase sample rate: filter should become "slower" per sample (lower k)
        filter.set_sample_frequency_hz(2000.0).unwrap();
        assert!(filter.smoothing_constant() < initial_k);
        assert_eq!(filter.sample_frequency_hz(), 2000.0);
    }

    #[test]
    fn test_dc_gain_unity() {
        let filter = Pt3Filter::new(standard_config());
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
        testing::test_frequency_response_at_cutoff(Pt3Filter::new(config));
    }

    // --- Nyquist tolerance tests ---

    #[test]
    fn test_nyquist_tolerance_build() {
        assert!(CommonFilterConfigBuilder::new()
            .cutoff_frequency_hz(600.0)
            .sample_frequency_hz(1000.0)
            .build()
            .is_ok());
    }

    #[test]
    fn test_nyquist_tolerance_set_cutoff() {
        let mut filter = Pt3Filter::new(standard_config());

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
        testing::check_convergence_to_steady_state_input(Pt3Filter::new(config), 1.0);
    }

    #[test]
    fn test_functional_stateful_equivalence() {
        let config = standard_config();
        testing::test_functional_stateful_equivalence(
            Pt3Filter::new(config),
            Pt3Filter::new(config),
            Pt3FilterContext::default(),
        );
    }

    #[test]
    fn test_stateless_context_independence() {
        testing::test_stateless_context_independence(
            Pt3Filter::new(standard_config()),
            Pt3FilterContext::default(),
        );
    }

    #[test]
    fn test_stateless_reset() {
        testing::test_stateless_reset(
            Pt3Filter::new(standard_config()),
            Pt3FilterContext::default(),
        );
    }
}
