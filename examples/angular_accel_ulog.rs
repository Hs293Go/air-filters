//! Compares two approaches for computing angular acceleration from PX4 gyro
//! data:
//!
//! **Chain A** — classical DSP pipeline:
//!   gyro → Biquad LPF → backward difference → Biquad LPF → α (rad/s²)
//!
//! **Chain B** — Savitzky-Golay derivative filter (single pass):
//!   gyro → SG(s=1) → α (rad/s²)
//!
//! Both chains are applied to all three gyro axes.  The output plot shows the
//! raw gyro signal, both angular-acceleration estimates, and their σ values per
//! axis.
//!
//! Usage:
//!   cargo run --example angular_accel_ulog -- path/to/flight.ulg
//!
//! Options:
//!   --cutoff-hz <f64>     Biquad LPF cutoff frequency in Hz (default: 80.0)
//!   --sg-window <i32>     SG window size (odd, ≥ 3; default: 11)
//!   --time-range <f64> <f64>  Time range in seconds (default: entire log)
//!   --save-path <str>     Output SVG path (default: angular_accel.svg)
#[cfg(feature = "std")]
mod ulog;

#[cfg(feature = "std")]
mod example {
    use super::ulog;
    use air_filters::{
        fir::savitzky_golay::{SavitzkyGolayFilter, SgConfigBuilder},
        iir::biquad::{BiquadFilter, BiquadFilterConfigBuilder, BiquadFilterType, DirectForm2},
        CommonFilterConfigBuilder, Filter,
    };
    use clap::Parser;
    use plotters::prelude::*;

    // ── Helpers
    // ───────────────────────────────────────────────────────────────────

    fn stddev(v: &[f64]) -> f64 {
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt()
    }

    fn biquad_lpf(cutoff_hz: f64, sample_hz: f64) -> BiquadFilter<f64, DirectForm2<f64>> {
        BiquadFilter::new(
            BiquadFilterConfigBuilder::<f64, DirectForm2<f64>>::direct_form_2()
                .cutoff_frequency_hz(cutoff_hz)
                .sample_frequency_hz(sample_hz)
                .filter_type(BiquadFilterType::LowPass)
                .build()
                .expect("biquad config"),
        )
    }

    // ── CLI ───────────────────────────────────────────────────────────────────────

    #[derive(Parser, Debug)]
    #[command(version, about, long_about = None)]
    struct Args {
        /// Path to the ULog file.
        file: String,

        /// Biquad LPF cutoff frequency in Hz (applied before and after backward
        /// difference).
        #[arg(long, default_value_t = 80.0)]
        cutoff_hz: f64,

        /// Savitzky-Golay window size (odd integer ≥ 3).
        #[arg(long, default_value_t = 11)]
        sg_window: i32,

        /// Time range in seconds to plot (e.g. `--time-range 10 20`).
        #[arg(long, num_args = 2)]
        time_range: Vec<f64>,

        /// Output SVG path.
        #[arg(short, long, default_value_t = String::from("angular_accel.svg"))]
        save_path: String,
    }

    // ── Main ──────────────────────────────────────────────────────────────────────

    pub fn angular_accel_ulog() -> Result<(), Box<dyn std::error::Error>> {
        let args = Args::parse();
        let Args {
            file,
            cutoff_hz,
            sg_window,
            time_range,
            save_path,
        } = args;

        eprintln!("Parsing {file} …");
        let data = ulog::read_topic(&file, "sensor_combined")?;

        let ts = data.get("timestamp").ok_or("field 'timestamp' missing")?;
        let gyro: [&Vec<f64>; 3] = [
            data.get("gyro_rad[0]").ok_or("gyro_rad[0] missing")?,
            data.get("gyro_rad[1]").ok_or("gyro_rad[1] missing")?,
            data.get("gyro_rad[2]").ok_or("gyro_rad[2] missing")?,
        ];

        let n = ts.len();
        let duration_s = (ts[n - 1] - ts[0]) / 1e6;

        // Median inter-sample interval — robust to occasional gaps in the log.
        let mut dts: Vec<f64> = ts.windows(2).map(|w| w[1] - w[0]).collect();
        dts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let sample_hz = 1e6 / dts[dts.len() / 2];

        eprintln!("sensor_combined: {n} samples, {duration_s:.1} s, {sample_hz:.1} Hz");
        eprintln!("Biquad cutoff: {cutoff_hz:.1} Hz  |  SG window: {sg_window}");

        // ── Build Chain A: Biquad LPF → backward diff → Biquad LPF ──────────────

        let common_cfg = CommonFilterConfigBuilder::new()
            .cutoff_frequency_hz(cutoff_hz)
            .sample_frequency_hz(sample_hz)
            .build()
            .expect("common filter config");
        let _ = common_cfg; // used only for the biquad builder below

        let mut pre_lpf: [BiquadFilter<f64, DirectForm2<f64>>; 3] =
            std::array::from_fn(|_| biquad_lpf(cutoff_hz, sample_hz));
        let mut post_lpf_a: [BiquadFilter<f64, DirectForm2<f64>>; 3] =
            std::array::from_fn(|_| biquad_lpf(cutoff_hz, sample_hz));

        let mut post_lpf_b: [BiquadFilter<f64, DirectForm2<f64>>; 3] =
            std::array::from_fn(|_| biquad_lpf(cutoff_hz, sample_hz));

        // ── Build Chain B: SG derivative filter ──────────────────────────────────

        let sg_cfg = SgConfigBuilder::new()
            .window_size(sg_window)
            .order(2)
            .deriv_order(1)
            .sample_frequency_hz(sample_hz)
            .build()
            .map_err(|e| format!("SG config error: {e:?}"))?;

        let mut sg: [SavitzkyGolayFilter<f64>; 3] =
            std::array::from_fn(|_| SavitzkyGolayFilter::new(sg_cfg));

        // ── Filter sample-by-sample ───────────────────────────────────────────────

        let mut chain_a: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::with_capacity(n));
        let mut chain_b: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::with_capacity(n));
        let mut prev_pre: [f64; 3] = [0.0; 3];

        for i in 0..n {
            for axis in 0..3 {
                let raw = gyro[axis][i];
                let smoothed = pre_lpf[axis].apply(raw);

                // Chain A
                let diff = if i == 0 {
                    0.0
                } else {
                    (smoothed - prev_pre[axis]) * sample_hz
                };
                prev_pre[axis] = smoothed;
                let alpha_a = post_lpf_a[axis].apply(diff);
                chain_a[axis].push(alpha_a);

                // Chain B
                let alpha_b = sg[axis].apply(raw);
                let alpha_b = post_lpf_b[axis].apply(alpha_b);
                chain_b[axis].push(alpha_b);
            }
        }

        // ── Plot ──────────────────────────────────────────────────────────────────

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
                "Warning: {}) samples in window, plotting may be slow. \
                    Use --time-range to zoom in.",
                win.len()
            );
        }

        let root = SVGBackend::new(&save_path, (900, 750)).into_drawing_area();
        root.fill(&WHITE)?;

        let color_a = RGBColor(31, 119, 180); // blue  — biquad chain
        let color_b = RGBColor(214, 39, 40); // red   — SG chain

        let axis_labels = ["α X (rad/s²)", "α Y (rad/s²)", "α Z (rad/s²)"];

        for (panel, (axis_label, axis_idx)) in root
            .split_evenly((3, 1))
            .iter()
            .zip(axis_labels.iter().zip(0..3usize))
        {
            // Y-axis range — union of both signals to keep them on the same scale.
            let y_vals: Vec<f64> = win
                .iter()
                .flat_map(|&i| [chain_a[axis_idx][i], chain_b[axis_idx][i]])
                .collect();
            let y_min = y_vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let y_max = y_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let margin = ((y_max - y_min) * 0.05).max(1.0);

            let sigma_a = stddev(&chain_a[axis_idx]);
            let sigma_b = stddev(&chain_b[axis_idx]);

            let mut chart = ChartBuilder::on(panel)
                .margin(8)
                .x_label_area_size(if axis_idx == 2 { 30 } else { 10 })
                .y_label_area_size(70)
                .caption(
                    format!(
                    "{axis_label}  |  Biquad-BD-Biquad σ={sigma_a:.2}  SG(w={sg_window})-Biquad σ={sigma_b:.2}  rad/s²"
                ),
                    ("sans-serif", 13),
                )
                .build_cartesian_2d(
                    win_start.max(t_s[0])..win_end.min(t_s[n - 1]),
                    (y_min - margin)..(y_max + margin),
                )?;

            chart
                .configure_mesh()
                .x_desc("Time (s)")
                .y_desc(*axis_label)
                .x_labels(if axis_idx == 2 { 6 } else { 0 })
                .x_label_formatter(&|v| format!("{v:.1} s"))
                .y_labels(5)
                .light_line_style(WHITE.mix(0.3))
                .draw()?;

            // Chain A
            chart
                .draw_series(LineSeries::new(
                    win.iter().map(|&i| (t_s[i], chain_a[axis_idx][i])),
                    color_a.stroke_width(1),
                ))?
                .label(format!("Biquad LPF({cutoff_hz:.0} Hz) → BD → LPF"))
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 16, y)], color_a));

            // Chain B
            chart
                .draw_series(LineSeries::new(
                    win.iter().map(|&i| (t_s[i], chain_b[axis_idx][i])),
                    color_b.stroke_width(1),
                ))?
                .label(format!("SG(w={sg_window}, n=2, s=1) → LPF"))
                .legend(|(x, y)| {
                    PathElement::new(vec![(x, y), (x + 16, y)], color_b.stroke_width(1))
                });

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
}

#[cfg(feature = "std")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    example::angular_accel_ulog()
}

#[cfg(not(feature = "std"))]
fn main() {
    eprintln!("This example requires the 'std' feature for ULog parsing");
}
