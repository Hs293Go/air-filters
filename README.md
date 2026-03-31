# air-filters

[![Rust CI & Publish](https://github.com/Hs293Go/air-filters/actions/workflows/rust.yml/badge.svg)](https://github.com/Hs293Go/air-filters/actions/workflows/rust.yml)
[![Crates.io](https://img.shields.io/crates/v/air-filters.svg)](https://crates.io/crates/air-filters)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Common digital **filters** for Unmanned **Aerial** Vehicles (UAVs) inspired by
[`betaflight`](https://github.com/betaflight/betaflight.git).

The following filters are implemented:

- **PT1**: A first-order low-pass filter for general use, also equivalent to a
  exponential moving average (EMA) filter, useful for smoothing noisy signals
  with minimal phase lag.
- **PT2**: A second-order low-pass filter composed of two cascaded PT1 filters,
  attenuates high-frequency noise more effectively than PT1, while introducing
  less phase lag than PT3.
- **PT3**: A third-order low-pass filter composed of three cascaded PT1 filters,
  useful for smoothing setpoints
- **biquad**:
  - In low-pass mode, effective for smoothing RPM feedback and the usual choice
    for filtering IMU data.
  - In notch filter mode, critical for rejecting motor/airframe harmonics.

## Features

- Inspired by Betaflight's `src/main/common/filter.c`, aiming to be a baseline
  for various filtering tasks in UAV applications.
- `no_std` compatible for deployment in embedded environments without the Rust
  standard library.
  - The `Pt1Filter` does **not** require a math library, making it suitable for
    barebones embedded targets.
  - Most of the remaining filters require floating-point math, so they require
    the `libm` feature

## Example

- Filtering a signal with a PT1 filter:

```rust
use air_filters::{iir::pt1::Pt1Filter, CommonFilterConfigBuilder, Filter};

fn main() {
    let config = CommonFilterConfigBuilder::new()
        .cutoff_frequency_hz(50.0)
        .sample_frequency_hz(1000.0)
        .build()
        .unwrap();
    let mut filter = Pt1Filter::new(config);

    let output = filter.apply(1.0);
    println!("Response: {output}");
}
```

- Filtering a signal with a biquad low-pass filter:

```rust
use air_filters::{
    self,
    iir::biquad::{BiquadFilter, BiquadFilterConfigBuilder, BiquadFilterType},
    Filter,
};

fn main() {
    let cfg = BiquadFilterConfigBuilder::direct_form_2()
        .cutoff_frequency_hz(40.0)
        .sample_frequency_hz(250.0)
        .filter_type(BiquadFilterType::LowPass)
        .build()
        .expect("Failed to build biquad filter config");

    let mut filter = BiquadFilter::new(cfg);

    let output = filter.apply(1.0);
    println!("Response: {output}");
}
```

## Code Examples

The basic application of filters can be demonstrated by running
`filter_imu_ulog` on IMU data in a PX4 flight log. An example command to filter
the gyroscope data between 25 and 30 seconds with a cutoff frequency of 60 Hz
is:

```bash
cargo run --example filter_imu_ulog -- flight_data.ulg --time-range 25 30 --cutoff-hz 30 --data-type accel --filter-type biquad
```

This generates a plot of the raw and filtered data, for example:

![Example of filtering IMU data from a PX4 flight log](https://raw.githubusercontent.com/Hs293Go/air-filters/refs/heads/main/imu_filtered.svg)

> [!NOTE]
>
> The particular log used in this example is
> [available here](https://review.px4.io/plot_app?log=4dea01a6-5981-4028-911c-ad2bb3f2a827).
>
> This log was presented in
> [a discussion on the PX4 forum](https://discuss.px4.io/t/high-vibration-in-cube-orange/32194/72)
> about how vibration on the Cube Orange autopilot worsened after upgrading to
> PX4 v1.14 and how the various filter and estimator parameters should be
> retuned to mitigate the issue. Downloading the ulog file, renaming it to
> `flight_data.ulg`, and running the above command exactly reproduces the plot
> shown above.
