// air-filters implements common digital filters inspired by implementations in betaflight
// Copyright © 2026 H S Helson Go
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
// OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::boxed::Box;

#[macro_use]
mod macros;

use core::time::Duration;

use num_traits::Float;

pub mod iir;

/// Errors that can occur during filter configuration and operation. These errors cover invalid
/// parameter values, such as non-positive or non-finite cutoff frequencies and sample frequencies,
/// as well as violations of the Nyquist theorem and non-finite internal states. Each error variant
/// is annotated with a descriptive message when the `std` feature is enabled, allowing for more
/// informative error handling in environments that support the standard library.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(thiserror::Error))]
pub enum Error {
    #[cfg_attr(feature = "std", error("Cutoff frequency must be positive"))]
    NonPositiveCutoffFrequency,
    #[cfg_attr(feature = "std", error("Sample frequency must be positive"))]
    NonPositiveSampleFrequency,
    #[cfg_attr(feature = "std", error("Quality factor must be positive"))]
    NonPositiveQualityFactor,
    #[cfg_attr(feature = "std", error("Cutoff frequency must be finite"))]
    NonFiniteCutoffFrequency,
    #[cfg_attr(feature = "std", error("Sample frequency must be finite"))]
    NonFiniteSampleFrequency,
    #[cfg_attr(feature = "std", error("Quality factor must be finite"))]
    NonFiniteQualityFactor,
    #[cfg_attr(feature = "std", error("Nyquist theorem violation: cutoff frequency must be less than half the sample frequency"))]
    NyquistTheoremViolation,
    #[cfg_attr(feature = "std", error("Filter internal state must remain finite"))]
    NonFiniteState,
}

/// Trait representing a generic filter that can be applied to input samples and stores internal state.
pub trait Filter<T> {
    /// Applies the filter to the input sample and updates the internal state. The output is the new state.
    fn apply(&mut self, input: T) -> T;

    /// Resets the filter state to the specified value. Returns an error if the state is not finite.
    fn reset(&mut self, state: T) -> Result<(), Error>;
}

impl<F, T, const N: usize> Filter<[T; N]> for [F; N]
where
    F: Filter<T>,
    T: Float,
{
    fn apply(&mut self, input: [T; N]) -> [T; N] {
        core::array::from_fn(|i| self[i].apply(input[i]))
    }

    fn reset(&mut self, state: [T; N]) -> Result<(), Error> {
        for (f, s) in self.iter_mut().zip(state) {
            f.reset(s)?;
        }
        Ok(())
    }
}

#[cfg(feature = "alloc")]
impl<T: Float> Filter<T> for Box<dyn Filter<T>> {
    fn apply(&mut self, input: T) -> T {
        (**self).apply(input)
    }

    fn reset(&mut self, state: T) -> Result<(), Error> {
        (**self).reset(state)
    }
}

#[cfg(feature = "alloc")]
impl<T: Float> Filter<T> for Box<dyn Filter<T> + Send> {
    fn apply(&mut self, input: T) -> T {
        (**self).apply(input)
    }

    fn reset(&mut self, state: T) -> Result<(), Error> {
        (**self).reset(state)
    }
}

#[cfg(feature = "alloc")]
impl<T: Float> Filter<T> for Box<dyn Filter<T> + Send + Sync> {
    fn apply(&mut self, input: T) -> T {
        (**self).apply(input)
    }

    fn reset(&mut self, state: T) -> Result<(), Error> {
        (**self).reset(state)
    }
}

mod internal {
    use super::*;

    pub trait ConfigurableFilter<T: Float> {
        fn config_mut(&mut self) -> &mut CommonFilterConfig<T>;

        fn update_configuration(&mut self) -> Result<(), super::Error>;
    }
}

pub trait CommonConfigurableFilter<T: Float>: Filter<T> + internal::ConfigurableFilter<T> {
    fn config(&self) -> &CommonFilterConfig<T>;

    /// Sets the cutoff frequency, in Hz, of the filter by deferring to
    /// [`CommonFilterConfig::cutoff_frequency_hz`] of the internal configuration object, then
    /// updates the filter configuration if successful.
    fn set_cutoff_frequency_hz(&mut self, cutoff_frequency_hz: T) -> Result<(), Error> {
        self.config_mut()
            .set_cutoff_frequency_hz(cutoff_frequency_hz)
            .and_then(|_| self.update_configuration())
    }

    /// Sets the sample frequency, in Hz, of the filter by deferring to
    /// [`CommonFilterConfig::sample_frequency_hz`] of the internal configuration object, then
    /// updates the filter configuration if successful
    fn set_sample_frequency_hz(&mut self, sample_frequency_hz: T) -> Result<(), Error> {
        self.config_mut()
            .set_sample_frequency_hz(sample_frequency_hz)
            .and_then(|_| self.update_configuration())
    }

    /// Sets the sample frequency based on the loop time by deferring to
    /// [`CommonFilterConfig::set_sample_loop_time`] of the internal configuration object, then
    /// updates the filter configuration if successful.
    fn set_sample_loop_time(&mut self, sample_loop_time: Duration) -> Result<(), Error> {
        self.config_mut()
            .set_sample_loop_time(sample_loop_time)
            .and_then(|_| self.update_configuration())
    }

    /// Returns the cutoff frequency in Hz.
    fn cutoff_frequency_hz(&self) -> T {
        self.config().cutoff_frequency_hz
    }

    /// Returns the sample frequency in Hz.
    fn sample_frequency_hz(&self) -> T {
        self.config().sample_frequency_hz
    }
}

/// Common configuration parameters for all filters, consisting of cutoff frequency_hz and sample
/// frequency_hz. This struct is used as a base for more specific filter configurations.
#[derive(Debug, Copy, Clone)]
pub struct CommonFilterConfig<T: Float> {
    cutoff_frequency_hz: T,
    sample_frequency_hz: T,
}

impl<T: Float> Default for CommonFilterConfig<T> {
    fn default() -> Self {
        Self {
            cutoff_frequency_hz: T::from(10.0).unwrap(),
            sample_frequency_hz: T::from(100.0).unwrap(),
        }
    }
}

impl<T: Float> CommonFilterConfig<T> {
    /// Sets the cutoff frequency in Hz in the configuration. Must be positive and finite.
    pub fn set_cutoff_frequency_hz(&mut self, cutoff_frequency_hz: T) -> Result<(), Error> {
        if !cutoff_frequency_hz.is_finite() {
            return Err(Error::NonFiniteCutoffFrequency);
        }

        if cutoff_frequency_hz <= t!(0) {
            return Err(Error::NonPositiveCutoffFrequency);
        }

        self.cutoff_frequency_hz = cutoff_frequency_hz;
        Ok(())
    }

    /// Sets the sample frequency in Hz in the configuration. Must be positive and finite.
    pub fn set_sample_frequency_hz(&mut self, sample_frequency_hz: T) -> Result<(), Error> {
        if !sample_frequency_hz.is_finite() {
            return Err(Error::NonFiniteSampleFrequency);
        }

        if sample_frequency_hz <= t!(0) {
            return Err(Error::NonPositiveSampleFrequency);
        }

        self.sample_frequency_hz = sample_frequency_hz;
        Ok(())
    }

    /// Sets the sample frequency based on the loop time. The sample frequency is calculated as the
    /// reciprocal of the loop time. The loop time must be positive and finite.
    pub fn set_sample_loop_time(&mut self, sample_loop_time: Duration) -> Result<(), Error> {
        let sample_frequency_hz = T::from(sample_loop_time.as_secs_f64().recip())
            .unwrap_or_else(|| unreachable!("f64 is always representable in T: Float"));
        self.set_sample_frequency_hz(sample_frequency_hz)
    }

    /// Returns the cutoff frequency in Hz.
    pub fn cutoff_frequency_hz(&self) -> T {
        self.cutoff_frequency_hz
    }

    /// Returns the sample frequency in Hz.
    pub fn sample_frequency_hz(&self) -> T {
        self.sample_frequency_hz
    }
}

/// Builder for [`CommonFilterConfig`] that allows for flexible construction with validation.
#[derive(Debug, Copy, Clone)]
pub struct CommonFilterConfigBuilder<T: Float> {
    cutoff_frequency_hz: Option<T>,
    sample_frequency_hz: Option<T>,
}

impl<T: Float> Default for CommonFilterConfigBuilder<T> {
    fn default() -> Self {
        Self {
            cutoff_frequency_hz: None,
            sample_frequency_hz: None,
        }
    }
}

impl<T: Float> CommonFilterConfigBuilder<T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Configures the cutoff frequency in Hz. If not set, defaults to 10.0 Hz.
    pub fn cutoff_frequency_hz(mut self, cutoff_frequency_hz: T) -> Self {
        self.cutoff_frequency_hz = Some(cutoff_frequency_hz);
        self
    }

    /// Configures the sample frequency in Hz. If not set, defaults to 100.0 Hz.
    pub fn sample_frequency_hz(mut self, sample_frequency_hz: T) -> Self {
        self.sample_frequency_hz = Some(sample_frequency_hz);
        self
    }

    /// Configures the sample frequency based on the loop time. The sample frequency is calculated as the
    /// reciprocal of the loop time.
    pub fn sample_loop_time(self, sample_loop_time: Duration) -> Self {
        let sample_frequency_hz = T::from(sample_loop_time.as_secs_f64().recip())
            .unwrap_or_else(|| unreachable!("f64 is always representable in T: Float"));
        self.sample_frequency_hz(sample_frequency_hz)
    }

    /// Validates the configuration and constructs a [`CommonFilterConfig`]. If any parameter is invalid, an error is returned.
    pub fn build(self) -> Result<CommonFilterConfig<T>, Error> {
        // 1. Resolve values (fallback to defaults if not set)
        let cutoff_frequency_hz = self
            .cutoff_frequency_hz
            .unwrap_or_else(|| T::from(10.0).unwrap());
        let sample_frequency_hz = self
            .sample_frequency_hz
            .unwrap_or_else(|| T::from(100.0).unwrap());

        // 2. Atomic validation
        if !cutoff_frequency_hz.is_finite() {
            return Err(Error::NonFiniteCutoffFrequency);
        }
        if !sample_frequency_hz.is_finite() {
            return Err(Error::NonFiniteSampleFrequency);
        }
        if cutoff_frequency_hz <= t!(0) {
            return Err(Error::NonPositiveCutoffFrequency);
        }
        if sample_frequency_hz <= t!(0) {
            return Err(Error::NonPositiveSampleFrequency);
        }

        Ok(CommonFilterConfig {
            cutoff_frequency_hz,
            sample_frequency_hz,
        })
    }
}

impl<T: Float> TryFrom<CommonFilterConfigBuilder<T>> for CommonFilterConfig<T> {
    type Error = Error;

    fn try_from(builder: CommonFilterConfigBuilder<T>) -> Result<Self, Self::Error> {
        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal pass-through filter: `apply` returns its input unchanged and
    /// stores it as state. Isolates blanket-impl tests from any concrete
    /// filter's arithmetic.
    struct Mock<T> {
        state: T,
    }

    impl<T: Float> Mock<T> {
        fn new() -> Self {
            Self { state: T::zero() }
        }
    }

    impl<T: Float> Filter<T> for Mock<T> {
        fn apply(&mut self, input: T) -> T {
            self.state = input;
            self.state
        }

        fn reset(&mut self, state: T) -> Result<(), Error> {
            if !state.is_finite() {
                return Err(Error::NonFiniteState);
            }
            self.state = state;
            Ok(())
        }
    }

    // ── ND filtering blanket impl ─────────────────────────────────────────────

    /// Axis i of the output equals the input applied to filter i.
    /// With pass-through semantics the expected output is trivially the input
    /// itself, so no filter arithmetic is needed to write the assertion.
    #[test]
    fn nd_apply_dispatches_per_axis() {
        let mut filters: [Mock<f64>; 3] = core::array::from_fn(|_| Mock::new());

        let out = filters.apply([1.0, 2.0, 3.0]);
        assert_eq!(out, [1.0, 2.0, 3.0]);

        // State of each filter reflects only its own axis.
        assert_eq!(filters[0].state, 1.0);
        assert_eq!(filters[1].state, 2.0);
        assert_eq!(filters[2].state, 3.0);
    }

    /// Inputs to one axis must not alter the state of any other axis.
    #[test]
    fn nd_apply_axes_are_independent() {
        let mut filters: [Mock<f64>; 3] = core::array::from_fn(|_| Mock::new());

        filters.apply([99.0, 0.0, 0.0]);

        assert_eq!(filters[1].state, 0.0, "axis 1 contaminated by axis 0");
        assert_eq!(filters[2].state, 0.0, "axis 2 contaminated by axis 0");
    }

    /// A successful vector reset writes each target state to its axis.
    #[test]
    fn nd_reset_distributes_to_all_axes() {
        let mut filters: [Mock<f64>; 3] = core::array::from_fn(|_| Mock::new());

        filters.reset([10.0, -20.0, 0.5]).unwrap();

        assert_eq!(filters[0].state, 10.0);
        assert_eq!(filters[1].state, -20.0);
        assert_eq!(filters[2].state, 0.5);
    }

    /// When reset fails on axis k, axes 0..k are mutated and axes k..N are not.
    /// This documents the known partial-mutation behaviour of the early-return
    /// in `Filter<[T; N]>::reset`.
    #[test]
    fn nd_reset_partial_failure_leaves_earlier_axes_mutated() {
        let mut filters: [Mock<f64>; 3] = core::array::from_fn(|_| Mock::new());
        filters.apply([1.0, 2.0, 3.0]);

        // Axis 0 valid, axis 1 NaN (fails), axis 2 never reached.
        let err = filters.reset([0.0, f64::NAN, 99.0]).unwrap_err();
        assert_eq!(err, Error::NonFiniteState);

        assert_eq!(filters[0].state, 0.0, "axis 0 should have been reset");
        assert_eq!(filters[1].state, 2.0, "axis 1 should be unchanged");
        assert_eq!(filters[2].state, 3.0, "axis 2 should be unchanged");
    }

    /// The blanket works for N = 1 (degenerate case).
    #[test]
    fn nd_apply_n1_works() {
        let mut filters: [Mock<f64>; 1] = [Mock::new()];
        assert_eq!(filters.apply([7.0]), [7.0]);
    }

    #[test]
    fn test_common_filter_configuration_builder() -> Result<(), Error> {
        // --- Existing Success Tests ---
        let config = CommonFilterConfigBuilder::new()
            .cutoff_frequency_hz(10.0)
            .sample_frequency_hz(100.0)
            .build()?;
        assert_eq!(config.cutoff_frequency_hz, 10.0);
        assert_eq!(config.sample_frequency_hz, 100.0);

        // --- Default and Fallback Tests ---
        let default_config = CommonFilterConfigBuilder::<f64>::new().build()?;
        assert_eq!(default_config.cutoff_frequency_hz, 10.0);
        assert_eq!(default_config.sample_frequency_hz, 100.0);

        // --- Negative Value Validations ---
        let err = CommonFilterConfigBuilder::new()
            .cutoff_frequency_hz(-1.0)
            .build()
            .unwrap_err();
        assert_eq!(err, Error::NonPositiveCutoffFrequency);

        let err = CommonFilterConfigBuilder::new()
            .sample_frequency_hz(0.0)
            .build()
            .unwrap_err();
        assert_eq!(err, Error::NonPositiveSampleFrequency);

        // --- Non-Finite Validations (NaN / Inf) ---
        let err = CommonFilterConfigBuilder::new()
            .cutoff_frequency_hz(f64::NAN)
            .build()
            .unwrap_err();
        assert_eq!(err, Error::NonFiniteCutoffFrequency);

        let err = CommonFilterConfigBuilder::new()
            .sample_frequency_hz(f64::INFINITY)
            .build()
            .unwrap_err();
        assert_eq!(err, Error::NonFiniteSampleFrequency);

        let err = CommonFilterConfigBuilder::<f64>::new()
            .sample_loop_time(Duration::ZERO)
            .build()
            .unwrap_err();
        assert_eq!(err, Error::NonFiniteSampleFrequency);

        // Just under half should pass
        let success = CommonFilterConfigBuilder::new()
            .cutoff_frequency_hz(49.99)
            .sample_frequency_hz(100.0)
            .build();
        assert!(success.is_ok());

        Ok(())
    }

    #[test]
    fn test_try_from_impl() {
        let builder = CommonFilterConfigBuilder::new()
            .cutoff_frequency_hz(5.0)
            .sample_frequency_hz(50.0);

        let config = CommonFilterConfig::try_from(builder);
        assert!(config.is_ok());
        assert_eq!(config.unwrap().cutoff_frequency_hz, 5.0);
    }
}
