//! Reads `sensor_combined` accelerometer data from a PX4 ULog file, applies a
//! biquad low-pass filter and a PT2 filter, then writes the raw and filtered
//! signals to a CSV on stdout.
//!
//! Usage:
//!   cargo run --example filter_accel_ulog -- path/to/flight.ulg
//!
//! Options:
//!   --cutoff-hz <f64>    Low-pass cutoff frequency in Hz (default: 30.0)
//!   --filter-type <str>  Filter type: biquad, pt1, pt2, or pt3 (default: biquad)
//!   --time-range <f64> <f64>  Time range in seconds to plot (default: entire log)
//!   --save-path <str>     Path to save the output plot image (default: ulog_accel_filtered.png)
mod ulog;

use clap::{Parser, ValueEnum};
use plotters::prelude::*;

use air_filters::{
    iir::{
        biquad::{BiquadFilter, BiquadFilterConfigBuilder, BiquadFilterType, DirectForm2},
        pt1::Pt1Filter,
        pt2::Pt2Filter,
        pt3::Pt3Filter,
    },
    CommonFilterConfigBuilder, Filter,
};

// ── Main ─────────────────────────────────────────────────────────────────────

fn stddev(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt()
}

#[derive(ValueEnum, Clone, Debug)]
enum FilterOptions {
    Pt1,
    Pt2,
    Pt3,
    Biquad,
}

#[derive(ValueEnum, Clone, Debug)]
enum DataType {
    Accel,
    Gyro,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the ULog file to read.
    file: String,

    /// Low-pass cutoff frequency in Hz.
    #[arg(short, long, default_value_t = 30.0)]
    cutoff_hz: f64,

    /// Filter type
    #[arg(short, long, value_enum, default_value_t = FilterOptions::Biquad)]
    filter_type: FilterOptions,

    /// Data type to plot (accel or gyro)
    #[arg(short, long, value_enum, default_value_t = DataType::Accel)]
    data_type: DataType,

    /// Time range in seconds to plot (e.g. `--time-range 30 40`), relative to the start of the log.
    #[arg(long, num_args = 2)]
    time_range: Vec<f64>,

    /// Path to save the output plot image.
    #[arg(short, long, default_value_t = String::from("imu_filtered.png"))]
    save_path: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let Args {
        file,
        cutoff_hz,
        filter_type,
        data_type,
        time_range,
        save_path,
    } = args;

    eprintln!("Parsing {file} …");

    let data = ulog::read_topic(&file, "sensor_combined")?;

    let ts = data.get("timestamp").ok_or("field 'timestamp' missing")?;
    let prefix = match data_type {
        DataType::Accel => "accelerometer_m_s2",
        DataType::Gyro => "gyro_rad",
    };
    let ax = data
        .get(&format!("{prefix}[0]"))
        .ok_or(format!("field '{prefix}[0]' missing").to_string())?;
    let ay = data
        .get(&format!("{prefix}[1]"))
        .ok_or(format!("field '{prefix}[1]' missing").to_string())?;
    let az = data
        .get(&format!("{prefix}[2]"))
        .ok_or(format!("field '{prefix}[2]' missing").to_string())?;

    let n = ts.len();
    let duration_s = (ts[n - 1] - ts[0]) / 1e6;

    // Median inter-sample interval — robust to dropout gaps in the log.
    let mut dts: Vec<f64> = ts.windows(2).map(|w| w[1] - w[0]).collect();
    dts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sample_hz = 1e6 / dts[dts.len() / 2];

    eprintln!("Data type: {data_type:?}");
    eprintln!("sensor_combined: {n} samples, {duration_s:.1} s, {sample_hz:.1} Hz");
    eprintln!("Cutoff: {cutoff_hz:.1} Hz");

    // ── Build filters ─────────────────────────────────────────────────────────
    //
    // One filter instance per axis per type — filters are stateful.

    let common_cfg = CommonFilterConfigBuilder::new()
        .cutoff_frequency_hz(cutoff_hz)
        .sample_frequency_hz(sample_hz)
        .build()
        .expect("filter config");

    let mut bq: [Box<dyn Filter<f64>>; 3] = std::array::from_fn(|_| match filter_type {
        FilterOptions::Biquad => Box::new(BiquadFilter::new(
            BiquadFilterConfigBuilder::<f64, DirectForm2<f64>>::direct_form_2()
                .cutoff_frequency_hz(cutoff_hz)
                .sample_frequency_hz(sample_hz)
                .filter_type(BiquadFilterType::LowPass)
                .build()
                .expect("filter config"),
        )) as Box<dyn Filter<f64>>,
        FilterOptions::Pt1 => Box::new(Pt1Filter::new(common_cfg)),
        FilterOptions::Pt2 => Box::new(Pt2Filter::new(common_cfg)),
        FilterOptions::Pt3 => Box::new(Pt3Filter::new(common_cfg)),
    });
    eprintln!("Using filter type: {filter_type:?}");

    // ── Filter sample-by-sample ───────────────────────────────────────────────

    let raw = [ax.as_slice(), ay.as_slice(), az.as_slice()];
    let mut bq_out: [Vec<f64>; 3] = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    for i in 0..n {
        for axis in 0..3 {
            bq_out[axis].push(bq[axis].apply(raw[axis][i]));
        }
    }

    // Zoom to a 10-second window during active flight.
    let (win_start, win_end) = if time_range.len() == 2 {
        (time_range[0], time_range[1])
    } else {
        (-f64::INFINITY, f64::INFINITY)
    };

    let t_s: Vec<f64> = ts.iter().map(|&v| (v - ts[0]) / 1e6).collect();
    let win: Vec<usize> = (0..n)
        .filter(|&i| t_s[i] >= win_start && t_s[i] <= win_end)
        .collect();
    if win.len() > 5000 {
        eprintln!(
            "Warning: large sample count ({}), plotting may be slow. \
             Consider using --time-range to zoom in.",
            win.len()
        );
    }

    let root = BitMapBackend::new(&save_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let raw_color = RGBColor(180, 200, 220);
    let bq_color = RGBColor(31, 119, 180);

    for ((panel, axis_label), axis_idx) in root
        .split_evenly((3, 1))
        .iter()
        .zip(match data_type {
            DataType::Accel => ["accel X (m/s²)", "accel Y (m/s²)", "accel Z (m/s²)"],
            DataType::Gyro => ["ωX (rad/s)", "ωY (rad/s)", "ωZ (rad/s)"],
        })
        .zip(0..3usize)
    {
        // Y-axis range: cover raw signal with a small margin.
        let raw_win: Vec<f64> = win.iter().map(|&i| raw[axis_idx][i]).collect();
        let y_min = raw_win.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = raw_win.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let margin = (y_max - y_min) * 0.05;

        let mut chart = ChartBuilder::on(panel)
            .margin(8)
            .x_label_area_size(if axis_idx == 2 { 30 } else { 10 })
            .y_label_area_size(55)
            .caption(
                format!(
                    "{filter_type:?} LPF on {data_type:?} data @ {cutoff_hz:.0} Hz  \
                     (raw σ={:.2}, filtered σ={:.2})",
                    stddev(raw[axis_idx]),
                    stddev(&bq_out[axis_idx]),
                ),
                ("sans-serif", 14),
            )
            .build_cartesian_2d(win_start..win_end, (y_min - margin)..(y_max + margin))?;

        chart
            .configure_mesh()
            .x_desc("Time (s)")
            .y_desc(axis_label)
            .x_labels(if axis_idx == 2 { 6 } else { 0 })
            .x_label_formatter(&|v| format!("{v:.0} s"))
            .y_labels(5)
            .light_line_style(WHITE.mix(0.3))
            .draw()?;

        // Raw (thin, light)
        chart
            .draw_series(LineSeries::new(
                win.iter().map(|&i| (t_s[i], raw[axis_idx][i])),
                raw_color.stroke_width(1),
            ))?
            .label("Raw")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 16, y)], raw_color));

        // LPF
        chart
            .draw_series(LineSeries::new(
                win.iter().map(|&i| (t_s[i], bq_out[axis_idx][i])),
                bq_color.stroke_width(1),
            ))?
            .label(format!("Biquad LPF {cutoff_hz:.0} Hz"))
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 16, y)], bq_color.stroke_width(1)));

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;
    }

    root.present()?;
    eprintln!("Saved {save_path}");

    Ok(())
}
