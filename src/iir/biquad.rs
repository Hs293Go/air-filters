//! Biquad filters with support for low-pass, notch, and band-pass types, and both Direct Form 1 and
//! Direct Form 2 Transposed topologies.
//!
//! Topology is encoded in the type of the filter and its configuration, allowing compile-time
//! selection of the difference equation and state storage.
//!
//! # Example
//!
//! [`BiquadFilterConfigBuilder`] offers a way to construct a [`BiquadFilter`] with a readable builder
//! pattern. After construction, usage is as simple a call to [`BiquadFilter::apply`] for each sample.
//!
//! ``` rust
//! use air_filters::iir::biquad::{BiquadFilter, BiquadFilterConfigBuilder, BiquadFilterType};
//! use air_filters::Filter;
//!
//! let mut filter = BiquadFilter::new(
//!     BiquadFilterConfigBuilder::direct_form_1()
//!         .cutoff_frequency_hz(50.0)
//!         .sample_frequency_hz(1000.0)
//!         .filter_type(BiquadFilterType::Notch)
//!         .q(5.0)
//!         .build()
//!         .unwrap(),
//! );
//!
//! let output = filter.apply(1.0);
//! ```
//!
//! Users perferring a stateful workflow can also directly construct the config object for their
//! desired topology
//!
//! ``` rust
//! use air_filters::iir::biquad::{BiquadFilterType, DF1BiquadFilter, DF1BiquadFilterConfig};
//! use air_filters::Filter;
//!
//! let mut config = DF1BiquadFilterConfig::new();
//!
//! // Setters return a Result to allow for validation (e.g. Nyquist enforcement), doesn't mutate
//! // the config if validation fails.
//! assert!(config.set_cutoff_frequency_hz(30.0).is_ok());
//! config.set_filter_type(BiquadFilterType::BandPass);
//! let mut filter = DF1BiquadFilter::new(config);
//!
//! let output = filter.apply(1.0);
//! ```

use core::{marker::PhantomData, time::Duration};

use crate::{
    internal::ConfigurableFilter, CommonConfigurableFilter, CommonFilterConfig,
    CommonFilterConfigBuilder, Error, Filter, FilterContext, FuncFilter,
};
use num_traits::{float::FloatCore, real::Real, FloatConst};

use crate::util::ring_buf::RingBuf;

/// Supported biquad filter types. Each type corresponds to a specific transfer
/// function and frequency response:
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BiquadFilterType {
    /// Attenuates frequencies above the cutoff. This variant is a 2nd-order Butterworth filter,
    /// fixing the quality factor as 1/sqrt(2) for maximally flat passband.
    LowPass,

    /// Attenuates frequencies around the cutoff, passing frequencies both above and below. The
    /// bandwidth of the notch is controlled by the quality factor `Q`. For dynamic notch filters,
    /// [`DirectForm1`] topology is recommended to avoid transients during coefficient updates.
    Notch,

    /// Attenuates frequencies far from the cutoff
    BandPass,
}

mod internal {
    use super::*;
    #[derive(Debug, Clone)]
    pub struct BiquadFilterCoefficients<T: FloatCore + Real> {
        pub a1: T,
        pub a2: T,
        pub b0: T,
        pub b1: T,
        pub b2: T,
    }

    /// Defines the topology of the biquad filter and manages the internal states.
    pub trait BiquadTopology<T: FloatCore + Real>: Default {
        /// Compute the output of the filter for the given input and coefficients, while updating the internal state.
        fn compute(&mut self, input: T, coeffs: &BiquadFilterCoefficients<T>) -> T;

        /// Reset the topology's internal delay-line to the steady-state corresponding
        /// to constant output `state`.
        ///
        /// Coefficients are supplied because the correct initial state depends on the
        /// filter's transfer function (topology alone does not know it).
        fn reset(&mut self, state: T, coeffs: &BiquadFilterCoefficients<T>) -> Result<(), Error>;
    }
}

/// Configuration for a biquad filter, holding both common parameters (cutoff frequency, sample
/// frequency) and biquad-specific parameters (filter type, Q). The filter topology is encoded
/// in the type parameter `P`.
#[derive(Debug, Copy, Clone)]
pub struct BiquadFilterConfig<T: FloatCore + Real, P: internal::BiquadTopology<T>> {
    base_config: CommonFilterConfig<T>,
    filter_type: BiquadFilterType,
    q: T,
    _topology: PhantomData<P>,
}

impl<T: FloatCore + Real + FloatConst> Default for BiquadFilterConfig<T, DirectForm1<T>> {
    /// Returns a default configuration for a biquad filter with [`DirectForm1`] topology, with
    /// cutoff frequency and sample frequency following [`CommonFilterConfig::default`] and filter
    /// type set to LowPass with Q fixed to 1/sqrt(2) for a Butterworth response.
    fn default() -> Self {
        Self {
            base_config: CommonFilterConfig::default(),
            filter_type: BiquadFilterType::LowPass,
            q: T::FRAC_1_SQRT_2(),
            _topology: PhantomData,
        }
    }
}

impl<T: FloatCore + Real + FloatConst> Default for BiquadFilterConfig<T, DirectForm2<T>> {
    /// Returns a default configuration for a biquad filter with [`DirectForm2`] topology, with
    /// cutoff frequency and sample frequency following [`CommonFilterConfig::default`] and filter
    /// type set to LowPass with Q fixed to 1/sqrt(2) for a Butterworth response.
    fn default() -> Self {
        Self {
            base_config: CommonFilterConfig::default(),
            filter_type: BiquadFilterType::LowPass,
            q: T::FRAC_1_SQRT_2(),
            _topology: PhantomData,
        }
    }
}

impl<T: FloatCore + Real + FloatConst> BiquadFilterConfig<T, DirectForm1<T>> {
    /// Creates a new `BiquadFilterConfig` with [`DirectForm1`] topology with default parameters
    /// following [`BiquadFilterConfig::default`].
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T: FloatCore + Real + FloatConst> BiquadFilterConfig<T, DirectForm2<T>> {
    /// Creates a new `BiquadFilterConfig` with [`DirectForm2`] topology with default parameters
    /// following [`BiquadFilterConfig::default`].
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T: FloatCore + Real + FloatConst, P: internal::BiquadTopology<T>> BiquadFilterConfig<T, P> {
    /// Returns the cutoff frequency in Hz.
    pub fn cutoff_frequency_hz(&self) -> T {
        self.base_config.cutoff_frequency_hz
    }

    /// Returns the sample frequency in Hz.
    pub fn sample_frequency_hz(&self) -> T {
        self.base_config.sample_frequency_hz
    }

    /// Returns the filter type (LowPass, Notch, BandPass).
    pub fn filter_type(&self) -> BiquadFilterType {
        self.filter_type
    }

    /// Returns the quality factor `Q`, which controls the filter's bandwidth and resonance. For
    /// low-pass filters, this is fixed to 1/sqrt(2) for a Butterworth response; for other filter
    /// types, it can be set by the user.
    pub fn quality_factor(&self) -> T {
        self.q
    }

    /// Sets the cutoff frequency in Hz. Must be less than half the sample frequency (Nyquist limit).
    pub fn set_cutoff_frequency_hz(&mut self, cutoff_frequency_hz: T) -> Result<(), Error> {
        if cutoff_frequency_hz >= self.sample_frequency_hz() * T::from(0.5).unwrap() {
            return Err(Error::NyquistTheoremViolation);
        }
        self.base_config
            .set_cutoff_frequency_hz(cutoff_frequency_hz)
    }

    /// Sets the sample frequency in Hz. Must be greater than twice the cutoff frequency (Nyquist limit).
    pub fn set_sample_frequency_hz(&mut self, sample_frequency_hz: T) -> Result<(), Error> {
        if self.cutoff_frequency_hz() >= sample_frequency_hz * T::from(0.5).unwrap() {
            return Err(Error::NyquistTheoremViolation);
        }
        self.base_config
            .set_sample_frequency_hz(sample_frequency_hz)
    }

    /// Sets the sample frequency by specifying the loop time. The sample frequency is the reciprocal of the loop time.
    pub fn set_sample_loop_time(
        &mut self,
        sample_loop_time: core::time::Duration,
    ) -> Result<(), Error> {
        let sample_frequency_hz = T::from(sample_loop_time.as_secs_f64().recip())
            .unwrap_or_else(|| unreachable!("f64 is always representable in T: FloatCore+Real"));
        self.set_sample_frequency_hz(sample_frequency_hz)
    }

    /// Sets the filter type (LowPass, Notch, BandPass). If the filter type is set to LowPass, the
    /// quality factor `Q` is overridden to 1/sqrt(2) for a Butterworth response.
    pub fn set_filter_type(&mut self, filter_type: BiquadFilterType) {
        self.filter_type = filter_type;
        if let BiquadFilterType::LowPass = filter_type {
            self.q = T::FRAC_1_SQRT_2();
        }
    }

    /// Sets the quality factor `Q`, which controls the filter's bandwidth and resonance. Must be
    /// positive and finite. For low-pass filters, this value is ignored and overridden to 1/sqrt(2)
    /// for a Butterworth response.
    pub fn set_quality_factor(&mut self, q: T) -> Result<(), Error> {
        if !q.is_finite() {
            return Err(Error::NonFiniteQualityFactor);
        }

        if q <= t!(0) {
            return Err(Error::NonPositiveQualityFactor);
        }

        self.q = if let BiquadFilterType::LowPass = self.filter_type {
            T::FRAC_1_SQRT_2()
        } else {
            q
        };
        Ok(())
    }
}

/// Builder for constructing a [`BiquadFilterConfig`] with either [`DirectForm1`] or [`DirectForm2`] topology.
#[derive(Debug, Clone)]
pub struct BiquadFilterConfigBuilder<T: FloatCore + Real, P: internal::BiquadTopology<T>> {
    base_config_builder: CommonFilterConfigBuilder<T>,
    filter_type: Option<BiquadFilterType>,
    q: Option<T>,
    _topology: PhantomData<P>,
}

impl<T: FloatCore + Real> BiquadFilterConfigBuilder<T, DirectForm1<T>> {
    /// Creates a new builder for a [`DirectForm1`] biquad filter.
    ///
    /// Choose DF1 when the filter's cutoff frequency or Q will be updated at runtime via
    /// [`CommonConfigurableFilter::set_cutoff_frequency_hz`] or similar. DF1 delay elements are
    /// raw signal samples, so coefficient changes take effect immediately without a transient
    /// glitch.
    pub fn direct_form_1() -> Self {
        Self {
            base_config_builder: CommonFilterConfigBuilder::default(),
            filter_type: None,
            q: None,
            _topology: PhantomData,
        }
    }
}

impl<T: FloatCore + Real> BiquadFilterConfigBuilder<T, DirectForm2<T>> {
    /// Creates a new builder for a [`DirectForm2`] (DF2T) biquad filter.
    ///
    /// Choose DF2T when the filter's coefficients are not likely to change during active
    /// operation. DF2T uses only two delay registers (vs. four for DF1), but those registers
    /// encode history through the feedback coefficients; an in-flight coefficient change causes a
    /// transient until the delay line flushes.
    pub fn direct_form_2() -> Self {
        Self {
            base_config_builder: CommonFilterConfigBuilder::default(),
            filter_type: None,
            q: None,
            _topology: PhantomData,
        }
    }
}

impl<T: FloatCore + Real + FloatConst, P: internal::BiquadTopology<T>>
    BiquadFilterConfigBuilder<T, P>
{
    /// Configures the cutoff frequency in Hz.
    pub fn cutoff_frequency_hz(mut self, cutoff_frequency_hz: T) -> Self {
        self.base_config_builder = self
            .base_config_builder
            .cutoff_frequency_hz(cutoff_frequency_hz);
        self
    }

    /// Configures the sample frequency in Hz.
    pub fn sample_frequency_hz(mut self, sample_frequency_hz: T) -> Self {
        self.base_config_builder = self
            .base_config_builder
            .sample_frequency_hz(sample_frequency_hz);
        self
    }

    /// Configures the filter type (LowPass, Notch, BandPass).
    pub fn filter_type(mut self, filter_type: BiquadFilterType) -> Self {
        self.filter_type = Some(filter_type);
        self
    }

    /// Configures the quality factor `Q`, which controls the filter's bandwidth and resonance.
    ///
    /// # Warning
    /// If the filter type is `LowPass`, this value is ignored and overridden to 1/sqrt(2) for a Butterworth response.
    pub fn q(mut self, q: T) -> Self {
        self.q = Some(q);
        self
    }

    /// Builds the `BiquadFilterConfig` from the provided parameters. Validates that the cutoff
    /// frequency is below the Nyquist limit and that the quality factor is positive and finite,
    /// while deferring to [`CommonFilterConfigBuilder::build`] for common parameter validation.
    /// Returns an error if any validation fails.
    pub fn build(self) -> Result<BiquadFilterConfig<T, P>, Error> {
        let base_config = self.base_config_builder.build()?;

        if base_config.cutoff_frequency_hz
            >= base_config.sample_frequency_hz * T::from(0.5).unwrap()
        {
            return Err(Error::NyquistTheoremViolation);
        }

        let filter_type = self.filter_type.unwrap_or(BiquadFilterType::LowPass);
        // For low-pass filters, the quality factor is fixed to 1/sqrt(2) for a Butterworth
        // response. For other filter types, use the provided Q value.
        let q = if let BiquadFilterType::LowPass = filter_type {
            T::FRAC_1_SQRT_2()
        } else {
            self.q.unwrap_or(T::FRAC_1_SQRT_2())
        };

        if !q.is_finite() {
            return Err(Error::NonFiniteQualityFactor);
        }

        if q <= t!(0) {
            return Err(Error::NonPositiveQualityFactor);
        }

        Ok(BiquadFilterConfig {
            base_config,
            filter_type,
            q,
            _topology: PhantomData,
        })
    }
}

/// Marker for the Direct Form II Transposed (DF2T) biquad topology.
///
/// Users likely don't need to interact with this type directly; see the type aliases
/// [`DF2BiquadFilterConfig`] and [`DF2BiquadFilter`] for convenient wrappers, or
/// [`BiquadFilterConfigBuilder::direct_form_2`] to select this topology when building a config.
///
/// # Details
///
/// Uses two delay elements `w[n-1]` and `w[n-2]`, where the intermediate variable
/// `w[n]` satisfies:
///
/// ```text
/// w[n]  =  x[n] − a1·w[n−1] − a2·w[n−2]
/// y[n]  =  b0·w[n] + b1·w[n−1] + b2·w[n−2]
/// ```
///
/// Because `w` is derived through the feedback coefficients `a1` and `a2`, the stored
/// delay elements encode input history *as seen through the current coefficients* — they
/// are not raw signal samples. Therefore, the functional separation of state and parameters
/// required for a [`FuncFilter`] implementation does not hold for this topology, and
/// `BiquadFilter<T, DirectForm2<T>>` does not implement `FuncFilter<T>`.
///
/// Furthermore, changing `a1` or `a2` between samples invalidates the stored values, causing a
/// transient glitch until the delay line flushes (two samples).
///
/// # Applications
///
/// Betaflight's `biquadFilterApply` (`filter.c`) uses this topology, noting:
/// *"higher precision but can't handle changes in coefficients"*.
///
/// Prefer DF2T for filters whose coefficients are set once at startup and never changed
/// during operation — a fixed gyroscope low-pass filter being the canonical example.
/// For filters that must update their cutoff or Q at runtime, use [`DirectForm1`].
#[derive(Debug, Clone)]
pub struct DirectForm2<T: FloatCore + Real> {
    x1: T,
    x2: T,
}

impl<T: FloatCore + Real> Default for DirectForm2<T> {
    fn default() -> Self {
        Self {
            x1: t!(0),
            x2: t!(0),
        }
    }
}

impl<T: FloatCore + Real> internal::BiquadTopology<T> for DirectForm2<T> {
    /// Evaluate one DF2T sample.
    ///
    /// `x1` and `x2` hold `w[n−1]` and `w[n−2]` — values of the intermediate variable
    /// that depend on both the signal and the feedback coefficients. They are valid only
    /// for the coefficient set that was active when they were last written. See the
    /// struct-level documentation for the consequences of mid-stream coefficient changes.
    fn compute(&mut self, input: T, coeffs: &internal::BiquadFilterCoefficients<T>) -> T {
        let result = coeffs.b0 * input + self.x1;

        self.x1 = coeffs.b1 * input - coeffs.a1 * result + self.x2;
        self.x2 = coeffs.b2 * input - coeffs.a2 * result;
        result
    }

    /// Reset the DF2T delay line to the steady-state values for constant output `state`.
    ///
    /// # Deviation from Betaflight
    ///
    /// Betaflight's `filter.c` does not implement a reset function for their DF2T biquad.  This
    /// implementation diverges by deriving the correct internal state from the current coefficients
    /// so that the filter produces `state` immediately on the next call to [`BiquadFilter::apply`]
    /// with input `state`.
    ///
    /// At steady state with constant input `u` and output `s = G·u` (`G = (b0+b1+b2)/(1+a1+a2)` is
    /// the DC gain):
    ///
    /// ```text
    /// x1 = s - b0·u  =  (1 - b0/G) · s x2 = b2·u - a2·s  =  (b2/G - a2) · s
    /// ```
    ///
    /// For unity-DC-gain filter types (LowPass, Notch) this simplifies to `x1 = (1-b0)·s` and `x2 =
    /// (b2-a2)·s`.
    ///
    /// # BandPass edge case
    ///
    /// BandPass filters have zero DC gain (`b0+b1+b2 = 0`), so no finite constant input produces a
    /// nonzero constant output.  Resetting to a nonzero `state` is therefore undefined; the delay
    /// line is zeroed and `Ok(())` is returned.
    fn reset(
        &mut self,
        state: T,
        coeffs: &internal::BiquadFilterCoefficients<T>,
    ) -> Result<(), Error> {
        if !state.is_finite() {
            return Err(Error::NonFiniteState);
        }

        let dc_den = coeffs.b0 + coeffs.b1 + coeffs.b2;

        if dc_den == t!(0) {
            // Zero DC gain (BandPass): steady state is undefined; zero the delay line.
            self.x1 = t!(0);
            self.x2 = t!(0);
            return Ok(());
        }

        // u: the constant input that produces constant output `state`.
        let dc_num = t!(1) + coeffs.a1 + coeffs.a2;
        let u = state * dc_num / dc_den;
        self.x1 = state - coeffs.b0 * u;
        self.x2 = coeffs.b2 * u - coeffs.a2 * state;
        Ok(())
    }
}

/// Type alias for the configuration of a biquad filter using the Direct Form 2 Transposed
/// topology.
///
/// This alias simplifies the construction of biquad filter configurations. See
/// [`DF1BiquadFilterConfig`] for an example of how these aliases may be used.
pub type DF2BiquadFilterConfig<T> = BiquadFilterConfig<T, DirectForm2<T>>;

/// Type alias for a biquad filter using the Direct Form 2 Transposed topology.
///
/// This alias simplifies the declaration of biquad filter types. See [`DF1BiquadFilter`] for an
/// example of how these aliases may be used.
pub type DF2BiquadFilter<T> = BiquadFilter<T, DirectForm2<T>>;

/// Marker for the Direct Form I (DF1) biquad topology.
///
/// Users likely don't need to interact with this type directly; see the type aliases
/// [`DF1BiquadFilterConfig`] and [`DF1BiquadFilter`] for convenient wrappers, or
/// [`BiquadFilterConfigBuilder::direct_form_1`] to select this topology when building a config.
///
/// # Details
///
/// Uses four delay elements — two past inputs (`x[n−1]`, `x[n−2]`) and two past
/// outputs (`y[n−1]`, `y[n−2]`):
///
/// ```text
/// y[n]  =  b0·x[n] + b1·x[n−1] + b2·x[n−2] − a1·y[n−1] − a2·y[n−2]
/// ```
///
/// Because the delay elements store actual signal values, not an intermediate quantity
/// derived from the coefficients, they remain valid across coefficient changes. Updating
/// `a1`, `a2`, `b0`, `b1`, or `b2` between samples is seamless: the next call to
/// [`BiquadFilter::apply`] evaluates the new difference equation against unchanged
/// historical data, producing no transient artifact.
///
/// The cost over [`DirectForm2`] is two extra delay registers (four instead of two).
///
/// # Applications
///
/// Betaflight's `biquadFilterApplyDF1` (`filter.c`) uses this topology for the
/// RPM-tracking notch filter, which calls `biquadFilterUpdate` on every gyro loop
/// iteration to track motor harmonics.
///
/// Prefer DF1 whenever the filter's cutoff frequency or Q must be updated at runtime.
/// For filters whose coefficients are fixed at startup, [`DirectForm2`] is sufficient
/// and uses half the delay-line registers.
#[derive(Debug, Clone)]
pub struct DirectForm1<T: FloatCore + Real> {
    x: RingBuf<T, 2>,
    y: RingBuf<T, 2>,
}

impl<T: FloatCore + Real> Default for DirectForm1<T> {
    fn default() -> Self {
        Self {
            x: RingBuf::new_filled(2, t!(0)),
            y: RingBuf::new_filled(2, t!(0)),
        }
    }
}

impl<T: FloatCore + Real> internal::BiquadTopology<T> for DirectForm1<T> {
    /// Evaluate one DF1 sample.
    ///
    /// `x1`, `x2` hold past input samples; `y1`, `y2` hold past output samples. All
    /// four are raw signal values independent of the coefficients, so a coefficient
    /// change between calls introduces no transient — the new coefficients are simply
    /// applied to the existing historical samples on the next evaluation.
    fn compute(&mut self, input: T, coeffs: &internal::BiquadFilterCoefficients<T>) -> T {
        let result = coeffs.b0 * input + coeffs.b1 * self.x[0] + coeffs.b2 * self.x[1]
            - coeffs.a1 * self.y[0]
            - coeffs.a2 * self.y[1];

        self.x.push_front(input);
        self.y.push_front(result);
        result
    }

    /// Reset all four delay elements to `state`.
    ///
    /// For unity-DC-gain filter types (LowPass, Notch) this produces the correct
    /// steady-state condition: applying input `state` immediately after reset
    /// yields output `state`.  Coefficients are accepted but unused; DF1 state
    /// initialisation does not require them because the delay elements directly
    /// represent past inputs and outputs.
    fn reset(
        &mut self,
        state: T,
        _coeffs: &internal::BiquadFilterCoefficients<T>,
    ) -> Result<(), Error> {
        if !state.is_finite() {
            return Err(Error::NonFiniteState);
        }
        self.x.fill(state);
        self.y.fill(state);
        Ok(())
    }
}

/// Type alias for the configuration of a biquad filter using the Direct Form 1 topology.
///
/// This alias simplifies the construction of biquad filter configurations, saving the user from
/// having to repeat the float type for both filter and topology.
///
/// ```rust
/// use air_filters::iir::biquad::{DF1BiquadFilter, DF1BiquadFilterConfig};
///
/// let config = DF1BiquadFilterConfig::<f32>::new();
/// let filter = DF1BiquadFilter::new(config);
/// ```
pub type DF1BiquadFilterConfig<T> = BiquadFilterConfig<T, DirectForm1<T>>;

/// Type alias for a biquad filter using the Direct Form I topology
///
/// This alias simplifies the declaration of biquad filter types, saving the user from having to
/// repeat the float type for both filter and topology.
///
/// ```rust
/// use air_filters::iir::biquad::DF1BiquadFilter;
///
/// struct Foo {
///     pub filter: DF1BiquadFilter<f32>, // equivalent to BiquadFilter<f32, DirectForm2<f32>>
/// }
/// ```
pub type DF1BiquadFilter<T> = BiquadFilter<T, DirectForm1<T>>;

/// A biquad IIR filter with configurable type (LowPass, Notch, BandPass) and topology (DF1 or DF2T).
#[derive(Debug, Clone)]
pub struct BiquadFilter<T: FloatCore + Real, P: internal::BiquadTopology<T>> {
    coeffs: internal::BiquadFilterCoefficients<T>,
    topology: P,
    config: BiquadFilterConfig<T, P>,
}

impl<T: FloatCore + Real + FloatConst, P: internal::BiquadTopology<T>> BiquadFilter<T, P> {
    fn compute_coeffs(config: &BiquadFilterConfig<T, P>) -> internal::BiquadFilterCoefficients<T> {
        let CommonFilterConfig {
            cutoff_frequency_hz,
            sample_frequency_hz,
        } = config.base_config;

        // Betaflight source has an additional 0.000001 factor because their sample frequency_hz is an
        // interval in microseconds
        let omega = t!(2) * T::PI() * cutoff_frequency_hz / sample_frequency_hz;
        let sn = omega.sin();
        let cs = omega.cos();

        let quality_factor = config.q;

        let alpha = sn / (t!(2) * quality_factor);
        let mut coeffs = internal::BiquadFilterCoefficients {
            a1: t!(0),
            a2: t!(0),
            b0: t!(0),
            b1: t!(0),
            b2: t!(0),
        };

        match config.filter_type {
            BiquadFilterType::LowPass => {
                // 2nd order Butterworth (with Q=1/sqrt(2)) / Butterworth biquad section with Q
                // described in http://www.ti.com/lit/an/slaa447/slaa447.pdf
                coeffs.b1 = t!(1) - cs;
                coeffs.b0 = coeffs.b1 * t!(0.5);
                coeffs.b2 = coeffs.b0;
                coeffs.a1 = -t!(2) * cs;
                coeffs.a2 = t!(1) - alpha;
            }
            BiquadFilterType::Notch => {
                coeffs.b0 = t!(1);
                coeffs.b1 = -t!(2) * cs;
                coeffs.b2 = t!(1);
                coeffs.a1 = coeffs.b1;
                coeffs.a2 = t!(1) - alpha;
            }
            BiquadFilterType::BandPass => {
                coeffs.b0 = alpha;
                coeffs.b1 = t!(0);
                coeffs.b2 = -alpha;
                coeffs.a1 = -t!(2) * cs;
                coeffs.a2 = t!(1) - alpha;
            }
        }
        let a0 = t!(1) + alpha;
        coeffs.a1 = coeffs.a1 / a0;
        coeffs.a2 = coeffs.a2 / a0;
        coeffs.b0 = coeffs.b0 / a0;
        coeffs.b1 = coeffs.b1 / a0;
        coeffs.b2 = coeffs.b2 / a0;
        coeffs
    }

    /// Creates a new `BiquadFilter` with the given configuration. The initial state is set to zero.
    pub fn new(config: BiquadFilterConfig<T, P>) -> Self {
        Self {
            coeffs: Self::compute_coeffs(&config),
            topology: P::default(),
            config,
        }
    }
}

impl<T: FloatCore + Real + FloatConst, P: internal::BiquadTopology<T>> ConfigurableFilter<T>
    for BiquadFilter<T, P>
{
    fn update_configuration(&mut self) -> Result<(), Error> {
        // Recompute coefficients based on the current config
        debug_assert!(
            self.config.cutoff_frequency_hz() < self.config.sample_frequency_hz() / t!(2)
        );

        self.coeffs = Self::compute_coeffs(&self.config);
        Ok(())
    }

    fn config_mut(&mut self) -> &mut CommonFilterConfig<T> {
        &mut self.config.base_config
    }
}

impl<T: FloatCore + Real + FloatConst, P: internal::BiquadTopology<T>> CommonConfigurableFilter<T>
    for BiquadFilter<T, P>
{
    /// Sets the cutoff frequency, in Hz, of the filter. Defers to
    /// [`BiquadFilterConfig::set_cutoff_frequency_hz`] for validation, including Nyquist
    /// enforcement. If the update is successful, the filter coefficients are recomputed to reflect
    /// the new cutoff frequency. Returns an error if validation fails.
    fn set_cutoff_frequency_hz(&mut self, f: T) -> Result<(), Error> {
        self.config.set_cutoff_frequency_hz(f)?;
        self.update_configuration()
    }

    /// Sets the sample frequency, in Hz, of the filter. Defers to
    /// [`BiquadFilterConfig::set_sample_frequency_hz`] for validation, including Nyquist
    /// enforcement. If the update is successful, the filter coefficients are recomputed to reflect
    /// the new sample frequency. Returns an error if validation fails.
    fn set_sample_frequency_hz(&mut self, f: T) -> Result<(), Error> {
        self.config.set_sample_frequency_hz(f)?;
        self.update_configuration()
    }

    /// Sets the sample frequency by specifying the loop time. The sample frequency is the reciprocal of the loop time. Defers to
    /// [`BiquadFilterConfig::set_sample_loop_time`] for validation, including Nyquist enforcement.
    /// If the update is successful, the filter coefficients are recomputed to reflect the new
    /// sample frequency. Returns an error if validation fails.
    fn set_sample_loop_time(&mut self, dt: Duration) -> Result<(), Error> {
        self.config.set_sample_loop_time(dt)?;
        self.update_configuration()
    }

    /// Returns the internal configuration of the filter.
    fn config(&self) -> &CommonFilterConfig<T> {
        &self.config.base_config
    }
}

impl<T: FloatCore + Real, P: internal::BiquadTopology<T>> Filter<T> for BiquadFilter<T, P> {
    fn apply(&mut self, input: T) -> T {
        self.topology.compute(input, &self.coeffs)
    }

    fn reset(&mut self, steady_output: T) -> Result<(), Error> {
        self.topology.reset(steady_output, &self.coeffs)
    }
}

/// Container for the internal states of a [`DF1BiquadFilter`], used for stateless application.
///
/// `x1`, `x2` hold past input samples; `y1`, `y2` hold past output samples.
#[derive(Debug, Copy, Clone)]
pub struct DF1BiquadFilterContext<T: FloatCore + Real> {
    /// `x[n−1]`: most recent past input sample.
    x1: T,
    /// `x[n−2]`: second most recent past input sample.
    x2: T,
    /// `y[n−1]`: most recent past output sample.
    y1: T,
    /// `y[n−2]`: second most recent past output sample.
    y2: T,
}

impl<T: FloatCore + Real> Default for DF1BiquadFilterContext<T> {
    /// Returns a zero-initialised context, suitable for a cold start.
    fn default() -> Self {
        Self {
            x1: t!(0),
            x2: t!(0),
            y1: t!(0),
            y2: t!(0),
        }
    }
}

impl<T: FloatCore + Real> FilterContext<T> for DF1BiquadFilterContext<T> {
    fn reset(&mut self, value: T) -> Result<(), Error> {
        if !value.is_finite() {
            return Err(Error::NonFiniteState);
        }

        self.x1 = value;
        self.x2 = value;
        self.y1 = value;
        self.y2 = value;
        Ok(())
    }

    fn last_output(&self) -> T {
        self.y1
    }
}

impl<T: FloatCore + Real> FuncFilter<T> for BiquadFilter<T, DirectForm1<T>> {
    type Context = DF1BiquadFilterContext<T>;

    fn apply_stateless(&self, input: T, ctx: &Self::Context) -> (T, Self::Context) {
        let result = self.coeffs.b0 * input + self.coeffs.b1 * ctx.x1 + self.coeffs.b2 * ctx.x2
            - self.coeffs.a1 * ctx.y1
            - self.coeffs.a2 * ctx.y2;
        (
            result,
            DF1BiquadFilterContext {
                x1: input,
                x2: ctx.x1,
                y1: result,
                y2: ctx.y1,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::iir::testing;

    use super::*;
    use approx::assert_relative_eq;

    // Helper for a standard 100Hz LPF on a 1kHz loop
    fn df1_config() -> BiquadFilterConfig<f64, DirectForm1<f64>> {
        BiquadFilterConfigBuilder::direct_form_1()
            .filter_type(BiquadFilterType::LowPass)
            .cutoff_frequency_hz(100.0)
            .sample_frequency_hz(1000.0)
            .build()
            .unwrap()
    }

    fn df2_config() -> BiquadFilterConfig<f64, DirectForm2<f64>> {
        BiquadFilterConfigBuilder::direct_form_2()
            .filter_type(BiquadFilterType::LowPass)
            .cutoff_frequency_hz(100.0)
            .sample_frequency_hz(1000.0)
            .build()
            .unwrap()
    }

    #[test]
    fn test_biquad_config_defaults() {
        let config = DF1BiquadFilterConfig::<f64>::default();
        assert_eq!(config.cutoff_frequency_hz(), 10.0);
        assert_eq!(config.sample_frequency_hz(), 100.0);
        assert_eq!(config.filter_type(), BiquadFilterType::LowPass);
        assert_relative_eq!(config.quality_factor(), 1.0 / 2f64.sqrt(), epsilon = 1e-10);

        let config = DF2BiquadFilterConfig::<f64>::default();
        assert_eq!(config.cutoff_frequency_hz(), 10.0);
        assert_eq!(config.sample_frequency_hz(), 100.0);
        assert_eq!(config.filter_type(), BiquadFilterType::LowPass);
        assert_relative_eq!(config.quality_factor(), 1.0 / 2f64.sqrt(), epsilon = 1e-10);

        let config = DF1BiquadFilterConfig::<f64>::new();
        assert_eq!(config.cutoff_frequency_hz(), 10.0);
        assert_eq!(config.sample_frequency_hz(), 100.0);
        assert_eq!(config.filter_type(), BiquadFilterType::LowPass);
        assert_relative_eq!(config.quality_factor(), 1.0 / 2f64.sqrt(), epsilon = 1e-10);

        let config = DF2BiquadFilterConfig::<f64>::new();
        assert_eq!(config.cutoff_frequency_hz(), 10.0);
        assert_eq!(config.sample_frequency_hz(), 100.0);
        assert_eq!(config.filter_type(), BiquadFilterType::LowPass);
        assert_relative_eq!(config.quality_factor(), 1.0 / 2f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_biquad_config_build_quality_factor_validation() {
        // Non-finite Q is rejected
        let result = BiquadFilterConfigBuilder::direct_form_1()
            .filter_type(BiquadFilterType::Notch)
            .q(f64::INFINITY)
            .build();
        assert!(matches!(result, Err(Error::NonFiniteQualityFactor)));

        // Non-positive Q is rejected
        let result = BiquadFilterConfigBuilder::direct_form_1()
            .filter_type(BiquadFilterType::Notch)
            .q(-1.0)
            .build();
        assert!(matches!(result, Err(Error::NonPositiveQualityFactor)));
    }

    #[test]
    fn test_biquad_config_quality_factor() {
        let mut config = DF1BiquadFilterConfig::<f64>::new();
        // Couldn't override Q for LowPass filter type
        assert!(config.set_quality_factor(1.0).is_ok());
        assert_relative_eq!(config.quality_factor(), 1.0 / 2f64.sqrt(), epsilon = 1e-10);
        // Could override Q for Notch filter type
        config.set_filter_type(BiquadFilterType::Notch);
        assert!(config.set_quality_factor(1.0).is_ok());
        assert_relative_eq!(config.quality_factor(), 1.0, epsilon = 1e-10);

        // Setting filter type back to LowPass should override Q again
        config.set_filter_type(BiquadFilterType::LowPass);
        assert_relative_eq!(config.quality_factor(), 1.0 / 2f64.sqrt(), epsilon = 1e-10);
    }
    #[test]
    fn test_biquad_config_invalid_quality_factor() {
        let mut config = DF1BiquadFilterConfig::<f64>::new();
        config.set_filter_type(BiquadFilterType::Notch);
        // Non-finite Q is rejected
        assert!(matches!(
            config.set_quality_factor(f64::INFINITY),
            Err(Error::NonFiniteQualityFactor)
        ));

        // Non-positive Q is rejected
        assert!(matches!(
            config.set_quality_factor(-1.0),
            Err(Error::NonPositiveQualityFactor)
        ));
    }
    #[test]
    fn test_biquad_config_builder_defaults() {
        let config = BiquadFilterConfigBuilder::<f64, _>::direct_form_1()
            .build()
            .unwrap();
        assert_eq!(config.cutoff_frequency_hz(), 10.0);
        assert_eq!(config.sample_frequency_hz(), 100.0);
        assert_eq!(config.filter_type(), BiquadFilterType::LowPass);
        assert_relative_eq!(config.quality_factor(), 1.0 / 2f64.sqrt(), epsilon = 1e-10);

        let config = BiquadFilterConfigBuilder::<f64, _>::direct_form_2()
            .build()
            .unwrap();
        assert_eq!(config.cutoff_frequency_hz(), 10.0);
        assert_eq!(config.sample_frequency_hz(), 100.0);
        assert_eq!(config.filter_type(), BiquadFilterType::LowPass);
        assert_relative_eq!(config.quality_factor(), 1.0 / 2f64.sqrt(), epsilon = 1e-10);

        // Default quality factor is still 1/sqrt(2) even if filter type is Notch and overriding is
        // lifted
        let config = BiquadFilterConfigBuilder::<f64, _>::direct_form_1()
            .filter_type(BiquadFilterType::Notch)
            .build()
            .unwrap();
        assert_eq!(config.cutoff_frequency_hz(), 10.0);
        assert_eq!(config.sample_frequency_hz(), 100.0);
        assert_eq!(config.filter_type(), BiquadFilterType::Notch);
        assert_relative_eq!(config.quality_factor(), 1.0 / 2f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_biquad_config_builder_invalid_common_config() {
        // Invalid cutoff frequency (negative)
        let result = BiquadFilterConfigBuilder::<f64, _>::direct_form_1()
            .cutoff_frequency_hz(-100.0)
            .build();
        assert!(matches!(result, Err(Error::NonPositiveCutoffFrequency)));

        // Invalid sample frequency (zero)
        let result = BiquadFilterConfigBuilder::<f64, _>::direct_form_1()
            .sample_frequency_hz(0.0)
            .build();
        assert!(matches!(result, Err(Error::NonPositiveSampleFrequency)));
    }

    #[test]
    fn test_biquad_lpf_coefficients() {
        let filter = BiquadFilter::new(df1_config());
        let c = &filter.coeffs;

        // For 100Hz/1000Hz LPF (Butterworth Q=0.7071):
        // omega = 2*PI*100/1000 = 0.628318
        // alpha = sin(omega)/(2*Q) = 0.587785 / 1.4142 = 0.41562
        // a0 = 1 + alpha = 1.41562
        // b1 = (1 - cos(omega)) / a0 = (1 - 0.809017) / 1.41562 ≈ 0.1349
        assert_relative_eq!(c.b1, 0.1349, epsilon = 1e-4);
    }
    #[test]
    fn test_touch_config_mut() {
        let mut filter = BiquadFilter::new(df1_config());
        assert!(filter.config_mut().set_cutoff_frequency_hz(1000.0).is_ok());
    }

    #[test]
    fn test_biquad_notch_null_response() {
        let config = BiquadFilterConfigBuilder::direct_form_1()
            .filter_type(BiquadFilterType::Notch)
            .cutoff_frequency_hz(250.0) // Center frequency
            .sample_frequency_hz(1000.0)
            .q(10.0) // Narrow notch
            .build()
            .unwrap();

        let mut filter = BiquadFilter::new(config);

        // A notch filter should significantly attenuate its center frequency.
        // We simulate a 250Hz sine wave (exactly 1/4 of fs)
        let mut max_output = 0.0f64;
        for i in 0..1000 {
            let input = (2.0 * core::f64::consts::PI * 0.25 * i as f64).sin();
            let out = filter.apply(input).abs();
            if i > 500 {
                max_output = out;
            }
        }
        // Output should be near zero (high attenuation)
        assert!(max_output < 0.05);
    }

    #[test]
    fn test_topology_equivalence() {
        let mut df1 = BiquadFilter::new(df1_config());
        let mut df2 = BiquadFilter::new(df2_config());

        // Apply same inputs to both topologies
        for i in 0..20 {
            let input = (i as f64 * 0.1).sin();
            let out1 = df1.apply(input);
            let out2 = df2.apply(input);
            // They should be identical within floating point precision
            assert_relative_eq!(out1, out2, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_biquad_reset_steady_state() {
        let mut filter = BiquadFilter::new(df1_config());

        // Prime the filter to 100.0
        filter.reset(100.0).expect("Reset failed");

        // The first apply of the same value should result in exactly that value
        let out = filter.apply(100.0);
        assert_relative_eq!(out, 100.0);
    }

    #[test]
    fn test_biquad_dynamic_reconfiguration() {
        let mut filter = BiquadFilter::new(df1_config());
        let initial_b0 = filter.coeffs.b0;

        // Changing cutoff should update coefficients via the Bridge pattern
        filter.set_cutoff_frequency_hz(200.0).unwrap();

        assert_ne!(filter.coeffs.b0, initial_b0);
        assert_eq!(filter.cutoff_frequency_hz(), 200.0);
    }

    #[test]
    fn test_lpf_unity_gain_dc() {
        let filter = BiquadFilter::new(df1_config());

        // Constant input should eventually result in constant output of the same value
        let input = 1.0;
        testing::check_convergence_to_steady_state_input(filter, input);
    }

    // --- Nyquist enforcement tests ---
    #[test]
    fn test_nyquist_rejected_by_config() {
        let mut config = DF1BiquadFilterConfig::new();
        assert_eq!(
            config.set_cutoff_frequency_hz(500.0).unwrap_err(),
            Error::NyquistTheoremViolation
        );

        let mut config = DF2BiquadFilterConfig::new();
        assert_eq!(
            config.set_cutoff_frequency_hz(500.0).unwrap_err(),
            Error::NyquistTheoremViolation
        );
    }

    #[test]
    fn test_nyquist_rejected_at_build() {
        // Exactly at Nyquist (f_c = f_s/2) must be rejected
        let err = BiquadFilterConfigBuilder::direct_form_1()
            .cutoff_frequency_hz(500.0)
            .sample_frequency_hz(1000.0)
            .build()
            .unwrap_err();
        assert_eq!(err, Error::NyquistTheoremViolation);

        // Above Nyquist (f_c > f_s/2) must be rejected
        let err = BiquadFilterConfigBuilder::direct_form_1()
            .cutoff_frequency_hz(600.0)
            .sample_frequency_hz(1000.0)
            .build()
            .unwrap_err();
        assert_eq!(err, Error::NyquistTheoremViolation);

        // Just below Nyquist must be accepted
        assert!(BiquadFilterConfigBuilder::direct_form_1()
            .cutoff_frequency_hz(499.9)
            .sample_frequency_hz(1000.0)
            .build()
            .is_ok());
    }

    #[test]
    fn test_nyquist_set_cutoff_rejected_and_state_preserved() {
        let mut filter = BiquadFilter::new(df1_config()); // 100Hz / 1000Hz
        let b0_before = filter.coeffs.b0;
        let b1_before = filter.coeffs.b1;
        let b2_before = filter.coeffs.b2;

        // Setting cutoff to Nyquist must fail
        assert_eq!(
            filter.set_cutoff_frequency_hz(500.0).unwrap_err(),
            Error::NyquistTheoremViolation
        );

        // Config and coefficients must be unchanged after the failed update
        assert_eq!(filter.cutoff_frequency_hz(), 100.0);
        assert_eq!(filter.coeffs.b0, b0_before);
        assert_eq!(filter.coeffs.b1, b1_before);
        assert_eq!(filter.coeffs.b2, b2_before);

        assert!(filter.set_cutoff_frequency_hz(499.9).is_ok());
    }

    #[test]
    fn test_nyquist_set_sample_frequency_rejected_and_state_preserved() {
        let mut filter = BiquadFilter::new(df1_config()); // 100Hz / 1000Hz
        let b0_before = filter.coeffs.b0;

        // Reducing f_s so that the current cutoff (100Hz) >= new f_s/2 (75Hz)
        assert_eq!(
            filter.set_sample_frequency_hz(150.0).unwrap_err(),
            Error::NyquistTheoremViolation
        );

        assert_eq!(filter.sample_frequency_hz(), 1000.0);
        assert_eq!(filter.coeffs.b0, b0_before);

        assert!(filter.set_sample_frequency_hz(2000.0).is_ok());
    }

    #[test]
    fn test_nyquist_set_sample_loop_time_rejected_and_state_preserved() {
        let mut filter = BiquadFilter::new(df1_config()); // 100Hz / 1000Hz
        let b0_before = filter.coeffs.b0;

        // 10ms loop time → f_s = 100Hz; cutoff 100Hz >= 50Hz → violation
        assert_eq!(
            filter
                .set_sample_loop_time(Duration::from_millis(10))
                .unwrap_err(),
            Error::NyquistTheoremViolation
        );

        assert_eq!(filter.sample_frequency_hz(), 1000.0);
        assert_eq!(filter.coeffs.b0, b0_before);

        assert!(filter
            .set_sample_loop_time(Duration::from_millis(1))
            .is_ok());
    }

    // --- DirectForm2 reset tests ---

    #[test]
    fn test_reset() {
        // After reset(s), the first apply(s) must return s exactly.
        let mut filter = BiquadFilter::new(df2_config());
        filter.reset(100.0).expect("DF2 reset failed");
        assert_relative_eq!(filter.apply(100.0), 100.0, epsilon = 1e-12);

        assert_eq!(filter.reset(f64::NAN).unwrap_err(), Error::NonFiniteState);

        let mut filter = BiquadFilter::new(df1_config());
        filter.reset(100.0).expect("DF1 reset failed");
        assert_relative_eq!(filter.apply(100.0), 100.0, epsilon = 1e-12);

        assert_eq!(filter.reset(f64::NAN).unwrap_err(), Error::NonFiniteState);
    }

    #[test]
    fn test_df2_reset_matches_df1() {
        // Both topologies, reset to the same value, must produce identical output
        // sequences for the same subsequent inputs.
        let mut df1 = BiquadFilter::new(df1_config());
        let mut df2 = BiquadFilter::new(df2_config());

        df1.reset(50.0).unwrap();
        df2.reset(50.0).unwrap();

        for i in 0..20 {
            let input = (i as f64 * 0.3).sin() * 50.0;
            assert_relative_eq!(df1.apply(input), df2.apply(input), epsilon = 1e-12);
        }
    }

    #[test]
    fn test_df2_reset_nonfinite_rejected() {
        let mut filter = BiquadFilter::new(df2_config());
        assert_eq!(filter.reset(f64::NAN).unwrap_err(), Error::NonFiniteState);
        assert_eq!(
            filter.reset(f64::INFINITY).unwrap_err(),
            Error::NonFiniteState
        );
    }

    #[test]
    fn test_df2_bandpass_reset_zeros_state_and_remains_operational() {
        // BandPass has zero DC gain; reset(s) zeros the delay line (defined behaviour)
        // and the filter must still process subsequent inputs without panicking.
        let config = BiquadFilterConfigBuilder::direct_form_2()
            .filter_type(BiquadFilterType::BandPass)
            .cutoff_frequency_hz(100.0)
            .sample_frequency_hz(1000.0)
            .q(1.0)
            .build()
            .unwrap();
        let mut filter = BiquadFilter::new(config);

        assert!(filter.reset(999.0).is_ok());

        // Delay line must have been zeroed: with zero initial state the first
        // output for a zero input is zero.
        assert_eq!(filter.apply(0.0), 0.0);
    }

    // --- Functional (stateless) tests ---

    #[test]
    fn test_df1_functional_stateful_equivalence() {
        let config = df1_config();
        testing::test_functional_stateful_equivalence(
            BiquadFilter::new(config.clone()),
            BiquadFilter::new(config),
            DF1BiquadFilterContext::default(),
        );
    }

    #[test]
    fn test_stateless_context_independence() {
        testing::test_stateless_context_independence(
            DF1BiquadFilter::new(df1_config()),
            DF1BiquadFilterContext::default(),
        );
    }

    #[test]
    fn test_stateless_reset() {
        testing::test_stateless_reset(
            DF1BiquadFilter::new(df1_config()),
            DF1BiquadFilterContext::default(),
        );
    }
}
