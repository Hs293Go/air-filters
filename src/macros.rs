macro_rules! t {
    (0) => {
        T::zero()
    };
    (1) => {
        T::one()
    };
    // Express common constants in terms of zero/one to avoid NumCast entirely.
    (2) => {
        T::one() + T::one()
    };
    (0.5) => {
        T::one() / (T::one() + T::one())
    };
    // General arm for compile-time constants (e.g. ORDER: i32) that cannot be
    // expressed with the arms above.  T::from() on any numeric constant returns
    // Some(_) for every T: Float; the unwrap path is therefore unreachable.
    ($val:expr) => {
        T::from($val).unwrap_or_else(|| unreachable!("numeric constant is not representable in T: Float"))
    };
}
