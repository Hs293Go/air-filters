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

//! air-filters implements common digital filters inspired by implementations in betaflight.
//!
//! It includes common filters like the [`Pt1Filter`](crate::iir::pt1::Pt1Filter) low-pass filter
//! (equivalent to a single-pole RC filter/exponential moving average),
//! [`Pt2Filter`](crate::iir::pt2::Pt2Filter) and [`Pt3Filter`](crate::iir::pt3::Pt3Filter) low-pass
//! filters built from cascades of first order sections, and biquad filters for more complex
//! filtering needs.
//!
//! The filters are designed to be correct by construction, with configurations managed by dedicated
//! configuration structs such as [`CommonFilterConfig`] that enforce parameter validation. The
//! filters implement a common trait [`Filter`] that allows for dynamic dispatch and composition,
//! such as applying an array of filters to an array of inputs in a per-axis manner.

#![no_std]
#![warn(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::boxed::Box;

#[macro_use]
mod macros;

use core::time::Duration;

use num_traits::float::FloatCore;

pub mod iir;

/// Errors that can occur during filter configuration and operation. Each error variant
/// is annotated with a descriptive message via [`thiserror`] when the `std` feature is enabled.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(thiserror::Error))]
pub enum Error {
    /// Signals when a cutoff frequency is set to a non-positive value (<= 0).
    #[cfg_attr(feature = "std", error("Cutoff frequency must be positive"))]
    NonPositiveCutoffFrequency,
    /// Signals when a sample frequency is set to a non-positive value (<= 0).
    #[cfg_attr(feature = "std", error("Sample frequency must be positive"))]
    NonPositiveSampleFrequency,
    /// Signals when a quality factor is set to a non-positive value (<= 0).
    #[cfg_attr(feature = "std", error("Quality factor must be positive"))]
    NonPositiveQualityFactor,
    /// Signals when a cutoff frequency is set to a non-finite value (NaN or infinity).
    #[cfg_attr(feature = "std", error("Cutoff frequency must be finite"))]
    NonFiniteCutoffFrequency,
    /// Signals when a sample frequency is set to a non-finite value (NaN or infinity).
    #[cfg_attr(feature = "std", error("Sample frequency must be finite"))]
    NonFiniteSampleFrequency,
    /// Signals when a quality factor is set to a non-finite value (NaN or infinity).
    #[cfg_attr(feature = "std", error("Quality factor must be finite"))]
    NonFiniteQualityFactor,

    /// Signals when the cutoff frequency is greater than or equal to half the sample frequency,
    /// which violates the Nyquist theorem. This situation is mathematically fatal for
    /// [`BiquadFilter`](crate::iir::biquad), but not for [`Pt1Filter`](crate::iir::pt1::Pt1Filter),
    /// etc. cascaded filters.
    #[cfg(any(feature = "libm", feature = "std"))]
    #[cfg_attr(feature = "std", error("Nyquist theorem violation: cutoff frequency must be less than half the sample frequency"))]
    NyquistTheoremViolation,

    /// Signals when a filter's internal state is set to a non-finite value (NaN or infinity) during reset.
    #[cfg_attr(feature = "std", error("Filter internal state must remain finite"))]
    NonFiniteState,
}

/// Trait representing a generic filter that can be applied to input samples and stores internal state.
///
/// The trait minimally requires the filters to implement `apply` and `reset` methods. A trait
/// object of `Filter<T>` allows easy dynamic dispatch of different filter implementations to the
/// effect of betaflight's selection of different filters for IMU filtering.
///
/// # Example
///
/// ```rust
/// const ORDER: usize = 1;
/// let cfg = air_filters::CommonFilterConfig::new();
/// let mut filter: Box<dyn air_filters::Filter<f64>> = match ORDER {
///     1 => Box::new(air_filters::iir::pt1::Pt1Filter::new(cfg))
///         as Box<dyn air_filters::Filter<f64>>,
///     #[cfg(any(feature = "libm", feature = "std"))]
///     2 => Box::new(air_filters::iir::pt2::Pt2Filter::new(cfg)),
///     #[cfg(any(feature = "libm", feature = "std"))]
///     3 => Box::new(air_filters::iir::pt3::Pt3Filter::new(cfg)),
///     _ => panic!("Unsupported filter order"),
/// };
/// ```
pub trait Filter<T> {
    /// Applies the filter to the input sample, updates internal state, and returns the new output.
    ///
    /// This function is not expected to do error checking on the input value.
    fn apply(&mut self, input: T) -> T;

    /// Initializes the filter to the steady-state condition for a constant output of
    /// `steady_output`. After a successful reset, the next call to `apply` with the same value
    /// will return that value unchanged.
    ///
    /// Implementers should return [`Error::NonFiniteState`] if `steady_output` is not finite,
    /// leaving the filter state unmodified.
    fn reset(&mut self, steady_output: T) -> Result<(), Error>;
}

/// Trait for the external state container threaded through [`FuncFilter::apply_stateless`].
///
/// Every context must be [`Copy`] (cheap to snapshot or branch) and [`Default`] (zero-initialised
/// cold start). The [`FilterContext::reset`] method provides a warm start from an arbitrary
/// steady-state value.
pub trait FilterContext<T>: Copy + Default {
    /// Resets the context initialised to the steady-state condition for a constant output of
    /// `steady_output`. After a successful call, passing `steady_output` to
    /// [`FuncFilter::apply_stateless`] with this context will return that value unchanged.
    ///
    /// Returns [`Error::NonFiniteState`] if `steady_output` is not finite.
    fn reset(&mut self, steady_output: T) -> Result<(), Error>;

    /// Returns the last output value associated with this context.
    fn last_output(&self) -> T;
}

impl<C: FilterContext<T>, T: Copy, const N: usize> FilterContext<[T; N]> for [C; N]
where
    [C; N]: Default,
{
    fn reset(&mut self, steady_output: [T; N]) -> Result<(), Error> {
        for (ctx, y) in self.iter_mut().zip(steady_output) {
            ctx.reset(y)?;
        }
        Ok(())
    }

    fn last_output(&self) -> [T; N] {
        core::array::from_fn(|i| self[i].last_output())
    }
}

/// Trait for filters that support a functional style of programming by not using mutable state.
///
/// Instead of storing state internally, the filter's state is threaded through a [`Copy`]able
/// [`FuncFilter::Context`] value that is passed in and returned on each call to
/// [`apply_stateless`](FuncFilter::apply_stateless).
///
/// # Example
/// ```rust
/// use air_filters::iir::pt1::{Pt1Filter, Pt1FilterContext};
/// use air_filters::{CommonFilterConfigBuilder, FuncFilter};
///
/// let filter = Pt1Filter::new(
///     CommonFilterConfigBuilder::new()
///         .cutoff_frequency_hz(100.0)
///         .sample_frequency_hz(1000.0)
///         .build()
///         .unwrap(),
/// );
///
/// let mut ctx = Pt1FilterContext::default();
/// let (output, ctx) = filter.apply_stateless(1.0, &ctx);
/// ```
pub trait FuncFilter<T>: Filter<T> {
    /// External state container for this filter. Must satisfy [`FilterContext<T>`], which
    /// requires [`Copy`] (cheap to snapshot or branch across streams), [`Default`] (cold start),
    /// and [`FilterContext::reset`] (warm start from a steady-state value).
    type Context: FilterContext<T>;

    /// Applies the filter to `input` without modifying internal state. Returns the filtered
    /// output and the updated context; the original context is left unchanged.
    ///
    /// The returned context must not be discarded: dropping it silently loses the updated
    /// filter state.
    #[must_use]
    fn apply_stateless(&self, input: T, ctx: &Self::Context) -> (T, Self::Context);
}

impl<F: Filter<T>, T: Copy, const N: usize> Filter<[T; N]> for [F; N] {
    /// Applies each filter in the array to the corresponding element of the input array, returning
    /// an array of outputs. The state of each filter is updated independently based on its own
    /// input.
    fn apply(&mut self, input: [T; N]) -> [T; N] {
        core::array::from_fn(|i| self[i].apply(input[i]))
    }

    /// Resets each filter to the steady-state condition for the corresponding element of
    /// `steady_output`. Stops and returns an error on the first failure; filters after that
    /// index are not reset and the array may be left partially reset if implementers do not take
    /// care to avoid mutating state before validating the input.
    fn reset(&mut self, steady_output: [T; N]) -> Result<(), Error> {
        for (f, s) in self.iter_mut().zip(steady_output) {
            f.reset(s)?;
        }
        Ok(())
    }
}

impl<F: FuncFilter<T>, T: Copy, const N: usize> FuncFilter<[T; N]> for [F; N]
where
    [F::Context; N]: FilterContext<[T; N]>,
{
    type Context = [F::Context; N];

    fn apply_stateless(&self, input: [T; N], ctx: &[F::Context; N]) -> ([T; N], [F::Context; N]) {
        let mut outputs = input;
        let mut new_ctxs = *ctx;
        for (i, filt) in self.iter().enumerate() {
            let (out, c) = filt.apply_stateless(input[i], &ctx[i]);
            outputs[i] = out;
            new_ctxs[i] = c;
        }
        (outputs, new_ctxs)
    }
}

#[cfg(feature = "alloc")]
impl<T: FloatCore> Filter<T> for Box<dyn Filter<T>> {
    fn apply(&mut self, input: T) -> T {
        (**self).apply(input)
    }

    fn reset(&mut self, steady_output: T) -> Result<(), Error> {
        (**self).reset(steady_output)
    }
}

mod internal {
    use super::*;

    pub trait ConfigurableFilter<T: FloatCore> {
        fn config_mut(&mut self) -> &mut CommonFilterConfig<T>;

        fn update_configuration(&mut self) -> Result<(), super::Error>;
    }
}

/// Trait for filters that have common configurable parameters (cutoff frequency and sample frequency).
pub trait CommonConfigurableFilter<T: FloatCore>:
    Filter<T> + internal::ConfigurableFilter<T>
{
    /// Returns a reference to the internal configuration object
    fn config(&self) -> &CommonFilterConfig<T>;

    /// Sets the cutoff frequency, in Hz, of the filter by deferring to
    /// [`CommonFilterConfig::set_cutoff_frequency_hz`] of the internal configuration object, then
    /// updates the filter configuration if successful.
    fn set_cutoff_frequency_hz(&mut self, cutoff_frequency_hz: T) -> Result<(), Error> {
        self.config_mut()
            .set_cutoff_frequency_hz(cutoff_frequency_hz)
            .and_then(|_| self.update_configuration())
    }

    /// Sets the sample frequency, in Hz, of the filter by deferring to
    /// [`CommonFilterConfig::set_sample_frequency_hz`] of the internal configuration object, then
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
pub struct CommonFilterConfig<T: FloatCore> {
    cutoff_frequency_hz: T,
    sample_frequency_hz: T,
}

impl<T: FloatCore> Default for CommonFilterConfig<T> {
    fn default() -> Self {
        Self {
            cutoff_frequency_hz: T::from(10.0).unwrap(),
            sample_frequency_hz: T::from(100.0).unwrap(),
        }
    }
}

impl<T: FloatCore> CommonFilterConfig<T> {
    /// Creates a new `CommonFilterConfig` with default values for cutoff frequency (10.0 Hz) and sample frequency (100.0 Hz).
    pub fn new() -> Self {
        Self::default()
    }

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
            .unwrap_or_else(|| unreachable!("f64 is always representable in T: FloatCore"));
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
#[derive(Debug, Clone)]
pub struct CommonFilterConfigBuilder<T: FloatCore> {
    cutoff_frequency_hz: Option<T>,
    sample_frequency_hz: Option<T>,
}

impl<T: FloatCore> Default for CommonFilterConfigBuilder<T> {
    fn default() -> Self {
        Self {
            cutoff_frequency_hz: None,
            sample_frequency_hz: None,
        }
    }
}

impl<T: FloatCore> CommonFilterConfigBuilder<T> {
    /// Creates a new `CommonFilterConfigBuilder` with no parameters configured. If parameters are
    /// left unconfigured, they will follow defaults in [`CommonFilterConfig::default()`].
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
            .unwrap_or_else(|| unreachable!("f64 is always representable in T: FloatCore"));
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

impl<T: FloatCore> TryFrom<CommonFilterConfigBuilder<T>> for CommonFilterConfig<T> {
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

    impl<T: FloatCore> Mock<T> {
        fn new() -> Self {
            Self { state: T::zero() }
        }
    }

    impl<T: FloatCore> Filter<T> for Mock<T> {
        fn apply(&mut self, input: T) -> T {
            self.state = self.state + input;
            input
        }

        fn reset(&mut self, steady_output: T) -> Result<(), Error> {
            if !steady_output.is_finite() {
                return Err(Error::NonFiniteState);
            }
            self.state = steady_output;
            Ok(())
        }
    }

    /// Minimal context for `Mock`: accumulates all inputs since construction or last reset.
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct MockCtx<T: FloatCore>(T);

    impl<T: FloatCore> Default for MockCtx<T> {
        fn default() -> Self {
            Self(T::zero())
        }
    }

    impl<T: FloatCore> FilterContext<T> for MockCtx<T> {
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

    impl<T: FloatCore> FuncFilter<T> for Mock<T> {
        type Context = MockCtx<T>;

        fn apply_stateless(&self, input: T, ctx: &MockCtx<T>) -> (T, MockCtx<T>) {
            (input, MockCtx(ctx.0 + input))
        }
    }

    #[test]
    fn test_common_filter_config_default() {
        let config = CommonFilterConfig::<f64>::default();
        assert_eq!(config.cutoff_frequency_hz(), 10.0);
        assert_eq!(config.sample_frequency_hz(), 100.0);

        let config = CommonFilterConfig::<f64>::new();
        assert_eq!(config.cutoff_frequency_hz(), 10.0);
        assert_eq!(config.sample_frequency_hz(), 100.0);
    }

    #[test]
    fn test_common_filter_config_setters() {
        let mut config = CommonFilterConfig::<f64>::default();
        assert!(config.set_cutoff_frequency_hz(20.0).is_ok());
        assert_eq!(config.cutoff_frequency_hz(), 20.0);

        assert!(config.set_sample_frequency_hz(200.0).is_ok());
        assert_eq!(config.sample_frequency_hz(), 200.0);

        assert!(config
            .set_sample_loop_time(Duration::from_secs_f64(0.01))
            .is_ok());
        assert_eq!(config.sample_frequency_hz(), 100.0);

        // Invalid values
        assert_eq!(
            config.set_cutoff_frequency_hz(-5.0).unwrap_err(),
            Error::NonPositiveCutoffFrequency
        );
        assert_eq!(
            config.set_sample_frequency_hz(0.0).unwrap_err(),
            Error::NonPositiveSampleFrequency
        );
        assert_eq!(
            config.set_sample_loop_time(Duration::ZERO).unwrap_err(),
            Error::NonFiniteSampleFrequency
        );
        assert_eq!(
            config.set_cutoff_frequency_hz(f64::NAN).unwrap_err(),
            Error::NonFiniteCutoffFrequency
        );
        assert_eq!(
            config.set_sample_frequency_hz(f64::INFINITY).unwrap_err(),
            Error::NonFiniteSampleFrequency
        );
    }

    #[cfg(any(feature = "alloc", feature = "std"))]
    #[test]
    fn test_boxed_filter() {
        let mut filter: Box<dyn Filter<f64>> = Box::new(Mock::new());
        assert_eq!(filter.apply(42.0), 42.0);
        assert_eq!(filter.reset(100.0), Ok(()));
        assert_eq!(filter.apply(7.0), 7.0);
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

    #[test]
    fn test_filter_context_reset() {
        let mut ctx = MockCtx::<f64>::default();
        ctx.reset(42.0).unwrap();
        assert_eq!(ctx, MockCtx(42.0));

        assert_eq!(ctx.reset(f64::NAN).unwrap_err(), Error::NonFiniteState);
    }

    #[test]
    fn nd_func_apply_stateless_dispatches_per_axis() {
        let filters: [Mock<f64>; 3] = core::array::from_fn(|_| Mock::new());
        let ctx = [MockCtx(0.0_f64); 3];

        let (outputs, new_ctx) = filters.apply_stateless([1.0, 2.0, 3.0], &ctx);
        assert_eq!(outputs, [1.0, 2.0, 3.0]);
        assert_eq!(new_ctx, [MockCtx(1.0), MockCtx(2.0), MockCtx(3.0)]);
        assert_eq!(
            ctx.last_output(),
            [0.0, 0.0, 0.0],
            "original context should be unchanged"
        );
        assert_eq!(
            new_ctx.last_output(),
            [1.0, 2.0, 3.0],
            "original context should be unchanged"
        );
    }

    #[test]
    fn nd_func_axes_are_independent() {
        let filters: [Mock<f64>; 3] = core::array::from_fn(|_| Mock::new());
        let ctx = [MockCtx(0.0_f64); 3];

        let (_, ctx1) = filters.apply_stateless([1.0, 0.0, 0.0], &ctx);
        let (_, ctx2) = filters.apply_stateless([0.0, 2.0, 0.0], &ctx);

        // Each context tracks only its own stream.
        assert_eq!(ctx1, [MockCtx(1.0), MockCtx(0.0), MockCtx(0.0)]);
        assert_eq!(ctx2, [MockCtx(0.0), MockCtx(2.0), MockCtx(0.0)]);
    }

    #[test]
    fn nd_func_reset_context_distributes_to_all_axes() {
        let mut ctx = [MockCtx(0.0_f64); 3];
        ctx.reset([10.0, -20.0, 0.5]).unwrap();
        assert_eq!(ctx, [MockCtx(10.0), MockCtx(-20.0), MockCtx(0.5)]);
    }

    #[test]
    fn nd_func_reset_context_fails_on_non_finite() {
        let mut ctx = [MockCtx(0.0_f64); 3];
        let err = ctx.reset([0.0, f64::NAN, 99.0]).unwrap_err();
        assert_eq!(err, Error::NonFiniteState);
    }

    #[test]
    fn test_stateful_stateless_filter_equivalence() {
        let mut filter = Mock::new();
        let input = 3.0_f64;
        let ctx = MockCtx(2.0_f64);

        // Apply stateful filter
        let stateful_output = filter.apply(input);
        assert_eq!(stateful_output, input);
        assert_eq!(filter.state, input);

        // Apply stateless filter with the same input and context
        let (stateless_output, new_ctx) = filter.apply_stateless(input, &ctx);
        assert_eq!(stateless_output, input);
        assert_eq!(new_ctx, MockCtx(ctx.0 + input));
    }
}
