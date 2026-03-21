use approx::assert_relative_eq;

use crate::{Filter, FilterContext, FuncFilter};

// The order-correction factor (1/√(∛2−1) ≈ 1.961) is designed so that
// the cascade of three first-order sections has its −3 dB point at exactly
// f_c in continuous time.  With a 1000:1 f_s/f_c ratio the discretisation
// error is < 0.4 %, keeping the measured gain within 1 % of 1/√2.
pub fn test_frequency_response_at_cutoff<F: Filter<f64>>(mut filter: F) {
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

pub fn check_convergence_to_steady_state_input<F: Filter<f64>>(mut filter: F, input: f64) {
    // Filter must still converge to a DC input
    let mut last_out = 0.0;
    for _ in 0..500 {
        last_out = filter.apply(input);
    }
    assert_relative_eq!(last_out, input, epsilon = 1e-6);
}

pub fn test_functional_stateful_equivalence<F: Filter<f64>, L: FuncFilter<f64>>(
    mut stateful: F,
    stateless: L,
    mut ctx: L::Context,
) {
    for &input in &[1.0_f64, 1.0, 1.0, 0.5, 0.0, -1.0, 0.0] {
        let stateful_out = stateful.apply(input);
        let (stateless_out, new_ctx) = stateless.apply_stateless(input, &ctx);
        ctx = new_ctx;
        assert_relative_eq!(stateful_out, stateless_out, epsilon = 1e-12);
        assert_relative_eq!(stateful_out, ctx.last_output(), epsilon = 1e-12);
    }
}

pub fn test_stateless_context_independence<L: FuncFilter<f64>>(filter: L, ctx_zero: L::Context) {
    // Verify that two independent contexts running on the same filter object do not
    // interfere with each other, and that the original context is not mutated.
    let (out_step, ctx_step) = filter.apply_stateless(1.0, &ctx_zero);
    let (out_zero, _) = filter.apply_stateless(0.0, &ctx_zero);

    // Step and zero inputs must produce different outputs
    assert!(out_step != out_zero);

    // The original context must be unmodified
    assert_eq!(ctx_zero.last_output(), 0.0);

    // Continued application from ctx_step should reflect its accumulated state
    let (out_step2, _) = filter.apply_stateless(1.0, &ctx_step);
    let (out_from_zero2, _) = filter.apply_stateless(1.0, &ctx_zero);
    assert!(out_step2 > out_from_zero2);
}

pub fn test_stateless_reset<L: FuncFilter<f64>>(filter: L, mut ctx: L::Context) {
    // Apply some inputs to move away from the initial state
    for _ in 0..10 {
        let (out, new_ctx) = filter.apply_stateless(1.0, &ctx);
        ctx = new_ctx;
        assert!(out > 0.0);
    }

    ctx.reset(0.5).unwrap(); // Reset the context to a steady output
    assert_eq!(ctx.last_output(), 0.5);

    ctx.reset(f64::INFINITY).unwrap_err(); // Reset with non-finite value should error
    assert_eq!(ctx.last_output(), 0.5); // State should remain unchanged after

    // After reset, the output should reflect the new steady state
    let (out_after_reset, _) = filter.apply_stateless(1.0, &ctx);
    assert!(out_after_reset > 0.5);
}
