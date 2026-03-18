//! Infinite Impulse Response (IIR) filters.
//!
//! IIR filters use feedback, meaning current output depends on both present/past inputs and
//! previous outputs. They can achieve sharper transition region rolloff than FIR
//! filters.
//!
//! This module contains implementations of various IIR filters, including biquad filters and first-, second-, and third-order low-pass filters.
pub mod biquad;
pub mod pt1;
pub mod pt2;
pub mod pt3;
