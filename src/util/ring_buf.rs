/// Fixed-capacity circular ring buffers used internally by filter
/// implementations.
///
/// Two implementations share the same public API:
/// - **no_std** — const-generic `[T; N]` stack array with a runtime
///   active-capacity field `cap ≤ N`, so a single type covers both the
///   always-full biquad delay line (`cap = N = 2`) and the variable-window SG
///   sample buffer (`cap = window ≤ N`).
/// - **std** — [`VecDeque<T>`](std::collections::VecDeque)-backed with runtime
///   capacity, suitable for large or dynamically-sized windows.
///
/// ## Push semantics
///
/// Both `push_front` and `push_back` maintain a fixed-size window by evicting
/// the element at the opposite end:
///
/// - **`push_back(val)`** — appends at the logical back; evicts the front.
///   After a sequence of `push_back` calls, `get(0)` / `buf[0]` is the
///   **oldest** sample and `get(cap - 1)` / `buf[cap - 1]` is the **newest**.
///   Used by the SG derivative filter, which needs oldest-first ordering for
///   the dot product.
///
/// - **`push_front(val)`** — prepends at the logical front; evicts the back.
///   After a sequence of `push_front` calls, `get(0)` / `buf[0]` is the
///   **newest** sample and `get(cap - 1)` / `buf[cap - 1]` is the **oldest**.
///   Used by the DF1 biquad delay line (`x[0]` = previous input, `x[1]` = one
///   before).
#[allow(dead_code)]
mod imp {
    use core::ops::Index;

    /// Fixed-capacity circular buffer backed by a const-generic stack array.
    ///
    /// `N` is the maximum capacity; the runtime `cap` (set at construction,
    /// `cap ≤ N`) is the number of active logical slots.
    #[derive(Clone, Debug)]
    pub struct RingBuf<T: Copy, const N: usize> {
        data: [T; N],
        /// Index of the logically *first* element (oldest for push_back,
        /// newest for push_front).
        head: usize,
        /// Number of active slots.  Invariant: `cap ≤ N`.
        cap: usize,
        primed: bool,
    }

    impl<T: Copy, const N: usize> RingBuf<T, N> {
        /// Creates a primed buffer with `cap` active slots all set to `val`.
        ///
        /// # Panics (debug only)
        /// Panics if `cap > N`.
        pub fn new_filled(cap: usize, val: T) -> Self {
            debug_assert!(cap <= N, "cap ({cap}) must be ≤ N ({N})");
            Self {
                data: [val; N],
                head: 0,
                cap,
                primed: true,
            }
        }

        /// Creates an **unprimed** buffer with `cap` active slots pre-loaded
        /// with `placeholder`.  [`fill`](Self::fill) must be called
        /// before trusting any read.
        ///
        /// # Panics (debug only)
        /// Panics if `cap > N`.
        pub fn new_empty(cap: usize, placeholder: T) -> Self {
            debug_assert!(cap <= N, "cap ({cap}) must be ≤ N ({N})");
            Self {
                data: [placeholder; N],
                head: 0,
                cap,
                primed: false,
            }
        }

        /// Overwrites all `cap` active slots with `val` and marks the buffer
        /// primed.
        pub fn fill(&mut self, val: T) {
            for slot in self.data.iter_mut().take(self.cap) {
                *slot = val;
            }
            self.head = 0;
            self.primed = true;
        }

        /// Appends `val` at the **back** (evicts the oldest element at the
        /// front).
        ///
        /// After `push_back`: `get(0)` = oldest, `get(cap-1)` = `val` (newest).
        pub fn push_back(&mut self, val: T) {
            self.data[self.head] = val;
            self.head = (self.head + 1) % self.cap;
            self.primed = true;
        }

        /// Prepends `val` at the **front** (evicts the oldest element at the
        /// back).
        ///
        /// After `push_front`: `get(0)` = `val` (newest), `get(cap-1)` =
        /// oldest.
        pub fn push_front(&mut self, val: T) {
            self.head = (self.head + self.cap - 1) % self.cap;
            self.data[self.head] = val;
            self.primed = true;
        }

        /// Returns the element at logical position `i`.
        #[inline]
        pub fn get(&self, i: usize) -> T {
            self.data[(self.head + i) % self.cap]
        }

        /// Returns `true` once the buffer has been primed.
        #[inline]
        pub fn is_primed(&self) -> bool {
            self.primed
        }

        /// Returns the number of active logical slots.
        #[inline]
        pub fn cap(&self) -> usize {
            self.cap
        }
    }

    impl<T: Copy, const N: usize> Index<usize> for RingBuf<T, N> {
        type Output = T;
        #[inline]
        fn index(&self, i: usize) -> &T {
            &self.data[(self.head + i) % self.cap]
        }
    }

    #[cfg(feature = "std")]
    extern crate std;
    #[cfg(feature = "std")]
    use std::collections::VecDeque;

    /// Fixed-capacity circular buffer backed by [`VecDeque`].
    ///
    /// Capacity is set at construction and is fixed thereafter.
    #[cfg(feature = "std")]
    #[derive(Clone, Debug)]
    pub struct GrowableRingBuf<T: Copy> {
        data: VecDeque<T>,
        primed: bool,
    }

    #[cfg(feature = "std")]
    impl<T: Copy> GrowableRingBuf<T> {
        /// Creates a primed buffer of `cap` slots all set to `val`.
        pub fn new_filled(cap: usize, val: T) -> Self {
            let data = core::iter::repeat_n(val, cap).collect();
            Self { data, primed: true }
        }

        /// Creates an unprimed buffer of `cap` slots pre-loaded with
        /// `placeholder`.
        pub fn new_empty(cap: usize, placeholder: T) -> Self {
            let data = core::iter::repeat_n(placeholder, cap).collect();
            Self {
                data,
                primed: false,
            }
        }

        /// Overwrites every slot with `val` and marks the buffer primed.
        pub fn fill(&mut self, val: T) {
            for slot in self.data.iter_mut() {
                *slot = val;
            }
            self.primed = true;
        }

        /// Appends `val` at the back, evicting the front element.
        ///
        /// After `push_back`: `[0]` = oldest, `[cap-1]` = `val` (newest).
        pub fn push_back(&mut self, val: T) {
            self.data.pop_front();
            self.data.push_back(val);
            self.primed = true;
        }

        /// Prepends `val` at the front, evicting the back element.
        ///
        /// After `push_front`: `[0]` = `val` (newest), `[cap-1]` = oldest.
        pub fn push_front(&mut self, val: T) {
            self.data.pop_back();
            self.data.push_front(val);
            self.primed = true;
        }

        /// Returns the element at logical position `i`.
        #[inline]
        pub fn get(&self, i: usize) -> T {
            self.data[i]
        }

        /// Returns `true` once the buffer has been primed.
        #[inline]
        pub fn is_primed(&self) -> bool {
            self.primed
        }

        /// Returns the capacity.
        #[inline]
        pub fn cap(&self) -> usize {
            self.data.len()
        }
    }

    #[cfg(feature = "std")]
    impl<T: Copy> Index<usize> for GrowableRingBuf<T> {
        type Output = T;
        #[inline]
        fn index(&self, i: usize) -> &T {
            &self.data[i]
        }
    }
}

#[cfg(feature = "std")]
pub(crate) use imp::GrowableRingBuf;
pub(crate) use imp::RingBuf;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #[cfg(feature = "std")]
    use crate::util::ring_buf::imp::GrowableRingBuf;

    #[cfg(not(feature = "std"))]
    use crate::util::ring_buf::imp::RingBuf;

    // ── push_back (SG / oldest-first) ─────────────────────────────────────────

    #[test]
    fn push_back_oldest_first() {
        #[cfg(not(feature = "std"))]
        let mut buf: RingBuf<i32, 3> = RingBuf::new_filled(3, 0);
        #[cfg(feature = "std")]
        let mut buf: GrowableRingBuf<i32> = GrowableRingBuf::new_filled(3, 0);

        buf.push_back(1);
        buf.push_back(2);
        buf.push_back(3);
        // After three pushes on a cap-3 buffer pre-filled with 0:
        // oldest = 1, middle = 2, newest = 3
        assert_eq!(buf[0], 1);
        assert_eq!(buf[1], 2);
        assert_eq!(buf[2], 3);
    }

    #[test]
    fn push_back_evicts_oldest() {
        #[cfg(not(feature = "std"))]
        let mut buf: RingBuf<i32, 3> = RingBuf::new_filled(3, 0);
        #[cfg(feature = "std")]
        let mut buf: GrowableRingBuf<i32> = GrowableRingBuf::new_filled(3, 0);

        buf.push_back(1);
        buf.push_back(2);
        buf.push_back(3);
        buf.push_back(4); // evicts 1
        assert_eq!(buf[0], 2);
        assert_eq!(buf[1], 3);
        assert_eq!(buf[2], 4);
    }

    // ── push_front (biquad / newest-first) ────────────────────────────────────

    #[test]
    fn push_front_newest_first() {
        #[cfg(not(feature = "std"))]
        let mut buf: RingBuf<i32, 2> = RingBuf::new_filled(2, 0);
        #[cfg(feature = "std")]
        let mut buf: GrowableRingBuf<i32> = GrowableRingBuf::new_filled(2, 0);

        buf.push_front(10);
        assert_eq!(buf[0], 10, "newest at [0]");
        assert_eq!(buf[1], 0, "evicted initial 0 to [1]");

        buf.push_front(20);
        assert_eq!(buf[0], 20, "newest at [0]");
        assert_eq!(buf[1], 10, "previous newest demoted to [1]");
    }

    // ── fill / is_primed ──────────────────────────────────────────────────────

    #[test]
    fn new_empty_is_unprimed() {
        #[cfg(not(feature = "std"))]
        let buf: RingBuf<f32, 4> = RingBuf::new_empty(4, 0.0);
        #[cfg(feature = "std")]
        let buf: GrowableRingBuf<f32> = GrowableRingBuf::new_empty(4, 0.0);
        assert!(!buf.is_primed());
    }

    #[test]
    fn fill_primes_and_sets_values() {
        #[cfg(not(feature = "std"))]
        let mut buf: RingBuf<f32, 4> = RingBuf::new_empty(4, 0.0);
        #[cfg(feature = "std")]
        let mut buf: GrowableRingBuf<f32> = GrowableRingBuf::new_empty(4, 0.0);

        buf.fill(7.0);
        assert!(buf.is_primed());
        for i in 0..4 {
            assert_eq!(buf[i], 7.0);
        }
    }

    // ── sub-capacity (no_std only: N > cap) ──────────────────────────────────

    #[cfg(not(feature = "std"))]
    #[test]
    fn sub_capacity_window() {
        // N=10 but only 3 active slots.
        let mut buf: RingBuf<i32, 10> = RingBuf::new_filled(3, 0);
        assert_eq!(buf.cap(), 3);
        buf.push_back(1);
        buf.push_back(2);
        buf.push_back(3);
        assert_eq!(buf[0], 1);
        assert_eq!(buf[1], 2);
        assert_eq!(buf[2], 3);
        buf.push_back(4); // evicts 1
        assert_eq!(buf[0], 2);
    }
}
