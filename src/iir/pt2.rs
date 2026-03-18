use num_traits::{Float, FloatConst, Pow};

use crate::{
    internal::ConfigurableFilter, CommonConfigurableFilter, CommonFilterConfig, Error, Filter,
};

/// A second-order low-pass filter implemented as a cascade of two first-order filters.
pub struct Pt2Filter<T: Float> {
    k: T,
    state: T,
    state1: T,
    config: CommonFilterConfig<T>,
}

impl<T: Float + FloatConst> Pt2Filter<T>
where
    T: Pow<T, Output = T>,
{
    fn compute_gain(config: &CommonFilterConfig<T>) -> T {
        const ORDER: i32 = 2;
        let order_cutoff_correction = t!(1) / (t!(2).pow(t!(1) / t!(ORDER)) - t!(1)).sqrt();
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

    /// Returns the current state of the filter, which represents the output of the filter at the last applied input.
    pub fn state(&self) -> T {
        self.state
    }

    /// Returns the smoothing constant `k`.
    pub fn smoothing_constant(&self) -> T {
        self.k
    }
}

impl<T: Float + FloatConst> ConfigurableFilter<T> for Pt2Filter<T>
where
    T: Pow<T, Output = T>,
{
    fn update_configuration(&mut self) -> Result<(), Error> {
        self.k = Self::compute_gain(&self.config);
        Ok(())
    }

    fn config_mut(&mut self) -> &mut CommonFilterConfig<T> {
        &mut self.config
    }
}

impl<T: Float + FloatConst> CommonConfigurableFilter<T> for Pt2Filter<T>
where
    T: Pow<T, Output = T>,
{
    fn config(&self) -> &CommonFilterConfig<T> {
        &self.config
    }
}

impl<T: Float> Filter<T> for Pt2Filter<T> {
    /// Applies the filter to the input sample and updates the internal states. The output is the new state.
    fn apply(&mut self, input: T) -> T {
        self.state1 = self.state1 + self.k * (input - self.state1);
        self.state = self.state + self.k * (self.state1 - self.state);
        self.state
    }

    /// Resets the filter states to the specified value. Returns an error if the state is not finite.
    fn reset(&mut self, state: T) -> Result<(), Error> {
        if !state.is_finite() {
            return Err(Error::NonFiniteState);
        }
        self.state = state;
        self.state1 = state;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::CommonFilterConfigBuilder;

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
        assert_eq!(filter.state, 25.0);
        assert_eq!(filter.state1, 25.0);

        // Steady state check: apply 25.0 again, output should remain 25.0
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

    // --- Frequency-response test ---

    #[test]
    fn test_frequency_response_at_cutoff() {
        // The order-correction factor (1/√(√2−1) ≈ 1.554) is designed so that
        // the cascade of two first-order sections has its −3 dB point at exactly
        // f_c in continuous time.  With a 1000:1 f_s/f_c ratio the discretisation
        // error is < 0.4 %, keeping the measured gain within 1 % of 1/√2.
        let config = CommonFilterConfigBuilder::new()
            .cutoff_frequency_hz(100.0)
            .sample_frequency_hz(100_000.0)
            .build()
            .unwrap();
        let mut filter = Pt2Filter::new(config);

        let omega = 2.0 * core::f64::consts::PI * 100.0 / 100_000.0;
        let n_settle = 10_000;
        let n_measure = 2_000;

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

        assert_relative_eq!(peak, core::f64::consts::FRAC_1_SQRT_2, epsilon = 0.01);
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
        let mut filter = Pt2Filter::new(config);

        let input = 1.0;
        let mut last_out = 0.0;
        for _ in 0..500 {
            last_out = filter.apply(input);
        }
        assert_relative_eq!(last_out, input, epsilon = 1e-6);
    }
}
