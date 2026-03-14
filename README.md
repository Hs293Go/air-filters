# air-filters

Common digital **filters** for Unmanned **Aerial** Vehicles (UAVs) inspired by
[`betaflight`](https://github.com/betaflight/betaflight.git).

The following filters are implemented:

- **PT1**: A first-order low-pass filter for general use
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
  - Enable the `libm` feature for embedded targets without the stdlib.

## Example

```rust
  use air_filters::{iir::Pt1Filter, CommonFilterConfigBuilder, Filter};

  let config = CommonFilterConfigBuilder::new()
      .cutoff_frequency_hz(50.0)
      .sample_frequency_hz(1000.0)
      .build()?;
  let mut filter = Pt1Filter::new(config);
  let output = filter.apply(sensor_reading);
```
