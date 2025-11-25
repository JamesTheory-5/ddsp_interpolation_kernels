# ddsp_interpolation_kernels
MODULE NAME:
ddsp_interpolation_kernels

DESCRIPTION:
Time-varying **fractional delay line** using high-quality interpolation kernels (linear, Catmull–Rom cubic, 3rd- and 5th-order Lagrange), implemented in pure functional JAX / GDSP style.
The module exposes `ddsp_interpolation_kernels_init`, `_update_state`, `_tick`, and `_process`, and can be used as a generic interpolation core (e.g., for physical modeling, chorus/flanger, or as a template for table readers).

INPUTS:

* x : input audio sample or audio buffer (shape `[...]` for tick, `(T,)` for process)
* params[0] = delay_samples : desired fractional delay in samples (scalar for tick, `(T,)` array for process)
* params[1] = interp_mode : integer interpolation mode

  * 0 = linear
  * 1 = Catmull–Rom cubic
  * 2 = 3rd-order Lagrange (4-tap)
  * 3 = 5th-order Lagrange (6-tap)
* params[2] = smooth_coef : 1-pole smoothing coefficient in (0, 1] applied to `delay_samples`

OUTPUTS:

* y : interpolated, delayed output sample / buffer

STATE VARIABLES (tuple):
(state_buffer, state_write_idx, state_delay_smooth)

* state_buffer : delay line buffer (1D array, shape `(N,)`)
* state_write_idx : current write index (scalar int32 in `[0, N)`)
* state_delay_smooth : smoothed delay value in samples (scalar float32)

EQUATIONS / MATH:

Let:

* `N` = buffer length (≥ `max_delay_samples + 6`)
* `b[n]` = delay buffer at sample `n`
* `w[n]` = write index at sample `n`
* `d_t[n]` = target delay from params (in samples)
* `d_s[n]` = smoothed delay (state)

**Smoothing:**
We use a 1-pole low-pass on the delay parameter:

* `d_s[n+1] = d_s[n] + α * (d_t[n] - d_s[n])`

  * `α = smooth_coef` in (0, 1]

**Write step:**

At each tick for input `x[n]`:

* `b[w[n]] ← x[n]` (modulo buffer length `N`)
* `w[n+1] = (w[n] + 1) mod N`

**Read position (fractional index in buffer):**

We read a sample delayed by `d_s[n+1]` (use smoothed value for stability):

* `r[n] = w[n] - d_s[n+1]`
* Wrap into `[0, N)` with floating modulo:

  * `r_wrap[n] = r[n] mod N`
* Split into integer + fractional parts:

  * `k[n] = floor(r_wrap[n])` (int)
  * `u[n] = r_wrap[n] - k[n]` (frac in `[0, 1)`)

**Neighbouring taps (ring buffer):**

Let `center = k[n]` and define integer indices:

* `i0 = (center - 2) mod N`
* `i1 = (center - 1) mod N`
* `i2 = (center + 0) mod N`
* `i3 = (center + 1) mod N`
* `i4 = (center + 2) mod N`
* `i5 = (center + 3) mod N`

Taps:

* `t0 = b[i0]`
* `t1 = b[i1]`
* `t2 = b[i2]`
* `t3 = b[i3]`
* `t4 = b[i4]`
* `t5 = b[i5]`

**Interpolation rules (in all cases, u = frac ∈ [0, 1)):**

1. **Linear** (between `t2` and `t3`):

   * `y_lin = t2 + u * (t3 - t2)`

2. **Cubic (Catmull–Rom)** using `p0=t1, p1=t2, p2=t3, p3=t4` as samples at x = −1, 0, 1, 2:

   Let `x = u`,

   * `p(x) = 0.5 * ( 2 p1 + (p2 - p0) x + (2p0 - 5p1 + 4p2 - p3) x^2 + (3p1 - p0 - 3p2 + p3) x^3 )`
   * `y_cubic = p(u)`

3. **3rd-order Lagrange (4 taps)** using `t1..t4` as positions 0..3, evaluated between t1 & t2:

   Let coordinates `x0=0, x1=1, x2=2, x3=3`, `x = 1+u`.
   With `u = frac`, the basis polynomials in terms of `u` are:

   * `ℓ0(u) = -(u)(u - 1)(u - 2) / 6`
   * `ℓ1(u) =  (1 + u)(u - 1)(u - 2) / 2`
   * `ℓ2(u) = -(1 + u)u(u - 2) / 2`
   * `ℓ3(u) =  (1 + u)u(u - 1) / 6`

   Output:

   * `y_lagr3 = t1*ℓ0(u) + t2*ℓ1(u) + t3*ℓ2(u) + t4*ℓ3(u)`

4. **5th-order Lagrange (6 taps)** using `t0..t5` as positions 0..5, evaluated between t2 & t3:

   Let `x = 2 + u`, `u = frac`. Define common factors:

   * `(x - 2) = u`
   * `(x - 3) = u - 1`
   * `(x - 4) = u - 2`
   * `(x - 5) = u - 3`
   * `(x - 0) = 2 + u`
   * `(x - 1) = 1 + u`

   Using precomputed denominators:

   * `D0 = -120, D1 = 24, D2 = -12, D3 = 12, D4 = -24, D5 = 120`

   Basis polynomials:

   * `ℓ0(x) = (x-1)(x-2)(x-3)(x-4)(x-5) / D0`
   * `ℓ1(x) = (x-0)(x-2)(x-3)(x-4)(x-5) / D1`
   * `ℓ2(x) = (x-0)(x-1)(x-3)(x-4)(x-5) / D2`
   * `ℓ3(x) = (x-0)(x-1)(x-2)(x-4)(x-5) / D3`
   * `ℓ4(x) = (x-0)(x-1)(x-2)(x-3)(x-5) / D4`
   * `ℓ5(x) = (x-0)(x-1)(x-2)(x-3)(x-4) / D5`

   Output:

   * `y_lagr5 = t0ℓ0 + t1ℓ1 + t2ℓ2 + t3ℓ3 + t4ℓ4 + t5ℓ5`

**Interpolation mode selection (no Python branching in jit):**

* `interp_mode_clipped = clip(interp_mode, 0, 3)`
* `y[n] = switch(interp_mode_clipped, [y_lin, y_cubic, y_lagr3, y_lagr5])`
  (implemented with `jax.lax.switch`)

through-zero rules:

* Delay can be time-varying and cross zero (if desired), but for a fractional delay line, `delay_samples` is typically constrained to `[0, max_delay_samples]`. Negative delays are undefined in this implementation and should be avoided.

phase wrapping rules:

* Buffer index wrapping is done via `jnp.mod(idx, N)` for both integer and float positions.

nonlinearities:

* None (module is linear in the input `x` for fixed `delay_samples`). Only nonlinearity is in parameter smoothing (linear filter).

interpolation rules:

* As above, polynomial interpolation of order 1, 3, or 5.

time-varying coefficient rules:

* Delay evolves as a smoothed version of the time-varying target using a 1-pole filter.
* This ensures stable, continuous interpolation even for fast delay modulation (e.g., flangers, choruses) while remaining fully differentiable.

NOTES:

* `smooth_coef` in `(0, 1]` controls smoothing speed: values near 1 track the target delay quickly; small values give slower parameter updates.
* `max_delay_samples` must be less than the buffer length chosen at init (`buffer_size >= max_delay_samples + 6`).
* For differentiable training, keeping `delay_samples` in a safe, positive range (e.g., `[1.0, max_delay_samples - 3]`) is recommended to avoid edge artifacts.
* `interp_mode` should be integer; gradients w.r.t. `interp_mode` are not meaningful (but gradients w.r.t. other parameters and inputs are preserved).

---

## FULL PYTHON FILE

```python
"""
ddsp_interpolation_kernels.py

Fully differentiable, JAX-based fractional delay line with multiple
interpolation kernels, in GDSP core style.

Public API:
    ddsp_interpolation_kernels_init(...)
    ddsp_interpolation_kernels_update_state(state, params)
    ddsp_interpolation_kernels_tick(x, state, params)
    ddsp_interpolation_kernels_process(x, state, params)

STATE:
    state = (buffer, write_idx, delay_smooth)

PARAMS (tuple, no dicts):
    params = (delay_samples, interp_mode, smooth_coef)

    delay_samples : desired delay in samples
        - scalar for tick()
        - array of shape (T,) for process()
    interp_mode : integer
        0 = linear
        1 = Catmull–Rom cubic
        2 = 3rd-order Lagrange (4-tap)
        3 = 5th-order Lagrange (6-tap)
    smooth_coef : scalar in (0, 1] controlling parameter smoothing

DSP MATH:
    See top-of-file documentation for detailed equations.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax


# ---------------------------------------------------------------------------
# Interpolation kernels (pure, stateless, all JAX)
# ---------------------------------------------------------------------------

def _linear_interp(y0: jnp.ndarray,
                   y1: jnp.ndarray,
                   frac: jnp.ndarray) -> jnp.ndarray:
    """Linear interpolation between y0 and y1."""
    return y0 + frac * (y1 - y0)


def _cubic_catmull_rom(p0: jnp.ndarray,
                       p1: jnp.ndarray,
                       p2: jnp.ndarray,
                       p3: jnp.ndarray,
                       frac: jnp.ndarray) -> jnp.ndarray:
    """
    Catmull–Rom cubic interpolation.

    p0, p1, p2, p3 are samples at x = -1, 0, 1, 2 respectively.
    frac in [0, 1) -> evaluation between p1 and p2.
    """
    x = frac
    x2 = x * x
    x3 = x2 * x

    a = 2.0 * p1
    b = p2 - p0
    c = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
    d = 3.0 * p1 - p0 - 3.0 * p2 + p3

    return 0.5 * (a + b * x + c * x2 + d * x3)


def _lagrange3_interp(t0: jnp.ndarray,
                      t1: jnp.ndarray,
                      t2: jnp.ndarray,
                      t3: jnp.ndarray,
                      frac: jnp.ndarray) -> jnp.ndarray:
    """
    3rd-order Lagrange interpolation (4 taps).

    t0..t3 correspond to x=0..3.
    Evaluates between t1 and t2 with x = 1 + frac, frac in [0, 1).
    """
    u = frac

    c0 = -u * (u - 1.0) * (u - 2.0) * (1.0 / 6.0)
    c1 = (1.0 + u) * (u - 1.0) * (u - 2.0) * 0.5
    c2 = -(1.0 + u) * u * (u - 2.0) * 0.5
    c3 = (1.0 + u) * u * (u - 1.0) * (1.0 / 6.0)

    return t0 * c0 + t1 * c1 + t2 * c2 + t3 * c3


def _lagrange5_interp(t0: jnp.ndarray,
                      t1: jnp.ndarray,
                      t2: jnp.ndarray,
                      t3: jnp.ndarray,
                      t4: jnp.ndarray,
                      t5: jnp.ndarray,
                      frac: jnp.ndarray) -> jnp.ndarray:
    """
    5th-order Lagrange interpolation (6 taps).

    t0..t5 correspond to x=0..5.
    Evaluates between t2 and t3 with x = 2 + frac, frac in [0, 1).
    """
    u = frac

    # Common factors:
    a0 = u          # (x - 2)
    a1 = u - 1.0    # (x - 3)
    a2 = u - 2.0    # (x - 4)
    a3 = u - 3.0    # (x - 5)
    b0 = 2.0 + u    # (x - 0)
    b1 = 1.0 + u    # (x - 1)

    # Denominators:
    # D0 = -120, D1 = 24, D2 = -12, D3 = 12, D4 = -24, D5 = 120
    l0 = -(b1 * a0 * a1 * a2 * a3) / 120.0
    l1 = (b0 * a0 * a1 * a2 * a3) / 24.0
    l2 = -(b0 * b1 * a1 * a2 * a3) / 12.0
    l3 = (b0 * b1 * a0 * a2 * a3) / 12.0
    l4 = -(b0 * b1 * a0 * a1 * a3) / 24.0
    l5 = (b0 * b1 * a0 * a1 * a2) / 120.0

    return (
        t0 * l0
        + t1 * l1
        + t2 * l2
        + t3 * l3
        + t4 * l4
        + t5 * l5
    )


# ---------------------------------------------------------------------------
# Core GDSP-style API
# ---------------------------------------------------------------------------

def ddsp_interpolation_kernels_init(
    max_delay_samples: int,
    *,
    buffer_size: int | None = None,
    initial_delay: float = 0.0,
    interp_mode: int = 2,
    smooth_coef: float = 0.01,
    dtype=jnp.float32,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
           Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Initialize the fractional delay interpolation module.

    Args:
        max_delay_samples: maximum expected delay in samples (int).
        buffer_size: optional override for delay buffer length.
            If None, uses max_delay_samples + 6.
        initial_delay: starting delay in samples (float).
        interp_mode: initial interpolation mode (0..3).
        smooth_coef: 1-pole smoothing coefficient in (0, 1].
        dtype: numeric dtype for buffer and delay.

    Returns:
        state: (buffer, write_idx, delay_smooth)
        params: (delay_samples, interp_mode, smooth_coef)
            - delay_samples is initialized to initial_delay.
    """
    if buffer_size is None:
        buffer_size = int(max_delay_samples) + 6

    # Buffer allocated outside jit (allowed).
    buffer = jnp.zeros((buffer_size,), dtype=dtype)
    write_idx = jnp.array(0, dtype=jnp.int32)
    delay_smooth = jnp.array(initial_delay, dtype=dtype)

    state = (buffer, write_idx, delay_smooth)

    delay_samples = jnp.array(initial_delay, dtype=dtype)
    interp_mode_arr = jnp.array(interp_mode, dtype=jnp.int32)
    smooth_coef_arr = jnp.array(smooth_coef, dtype=dtype)

    params = (delay_samples, interp_mode_arr, smooth_coef_arr)
    return state, params


def ddsp_interpolation_kernels_update_state(
    state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Optional state-only update (e.g., to advance parameter smoothing
    without processing audio).

    This applies one smoothing step to the delay parameter using the
    same rule as in tick().

    Args:
        state: (buffer, write_idx, delay_smooth)
        params: (delay_samples, interp_mode, smooth_coef)
            delay_samples must be scalar here (no time axis).

    Returns:
        new_state with updated delay_smooth.
    """
    buffer, write_idx, delay_smooth = state
    delay_samples, _interp_mode, smooth_coef = params

    # Ensure scalar-like delay for this usage:
    delay_target = delay_samples

    # One smoothing step:
    delay_next = delay_smooth + smooth_coef * (delay_target - delay_smooth)

    return buffer, write_idx, delay_next


@jax.jit
def ddsp_interpolation_kernels_tick(
    x: jnp.ndarray,
    state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Process one sample with a time-varying fractional delay line.

    Args:
        x: input sample (scalar array).
        state: (buffer, write_idx, delay_smooth)
        params: (delay_samples, interp_mode, smooth_coef)
            delay_samples: desired delay in samples (scalar).
            interp_mode: 0=linear, 1=cubic, 2=Lagrange3, 3=Lagrange5.
            smooth_coef: smoothing coefficient in (0, 1].

    Returns:
        y: output sample (scalar).
        new_state: updated state tuple.
    """
    buffer, write_idx, delay_smooth = state
    delay_samples, interp_mode, smooth_coef = params

    # Cast input to buffer dtype.
    x = jnp.asarray(x, dtype=buffer.dtype)
    delay_target = jnp.asarray(delay_samples, dtype=buffer.dtype)
    smooth_coef = jnp.asarray(smooth_coef, dtype=buffer.dtype)

    # One-pole smoothing on delay parameter.
    delay_next = delay_smooth + smooth_coef * (delay_target - delay_smooth)

    buffer_len = buffer.shape[0]
    buffer_len_f = jnp.asarray(buffer_len, dtype=buffer.dtype)

    # Wrap write index.
    write_idx_wrapped = jnp.mod(write_idx, buffer_len).astype(jnp.int32)

    # Write current sample into buffer via dynamic_update_slice.
    x_slice = x.reshape((1,))
    buffer = lax.dynamic_update_slice(
        buffer,
        x_slice,
        (write_idx_wrapped,)
    )

    # Compute fractional read position: w - delay (using smoothed delay).
    write_idx_f = jnp.asarray(write_idx_wrapped, dtype=buffer.dtype)
    read_pos = write_idx_f - delay_next

    # Wrap fractional index into [0, buffer_len).
    read_pos_wrapped = jnp.mod(read_pos, buffer_len_f)

    idx_base_f = jnp.floor(read_pos_wrapped)
    idx_base = idx_base_f.astype(jnp.int32)
    frac = read_pos_wrapped - idx_base_f  # in [0, 1)

    # Center index for taps: idx_base.
    center = idx_base

    # Compute tap indices (center-2 .. center+3) with wrapping.
    def wrap_int(i: jnp.ndarray) -> jnp.ndarray:
        return jnp.mod(i, buffer_len).astype(jnp.int32)

    i0 = wrap_int(center - 2)
    i1 = wrap_int(center - 1)
    i2 = wrap_int(center + 0)
    i3 = wrap_int(center + 1)
    i4 = wrap_int(center + 2)
    i5 = wrap_int(center + 3)

    t0 = buffer[i0]
    t1 = buffer[i1]
    t2 = buffer[i2]
    t3 = buffer[i3]
    t4 = buffer[i4]
    t5 = buffer[i5]

    # Interpolation mode selection using lax.switch (no Python branching).
    mode = jnp.clip(interp_mode, 0, 3)

    vals = (t0, t1, t2, t3, t4, t5, frac)

    def _interp_linear(v):
        t0_, t1_, t2_, t3_, t4_, t5_, f_ = v
        return _linear_interp(t2_, t3_, f_)

    def _interp_cubic(v):
        t0_, t1_, t2_, t3_, t4_, t5_, f_ = v
        # Use t1..t4 as p0..p3
        return _cubic_catmull_rom(t1_, t2_, t3_, t4_, f_)

    def _interp_lagr3(v):
        t0_, t1_, t2_, t3_, t4_, t5_, f_ = v
        # Use t1..t4 as taps for positions 0..3
        return _lagrange3_interp(t1_, t2_, t3_, t4_, f_)

    def _interp_lagr5(v):
        t0_, t1_, t2_, t3_, t4_, t5_, f_ = v
        return _lagrange5_interp(t0_, t1_, t2_, t3_, t4_, t5_, f_)

    y = lax.switch(
        mode,
        (
            _interp_linear,
            _interp_cubic,
            _interp_lagr3,
            _interp_lagr5,
        ),
        vals,
    )

    # Update write index.
    write_idx_next = jnp.mod(write_idx_wrapped + 1, buffer_len).astype(jnp.int32)

    new_state = (buffer, write_idx_next, delay_next)
    return y, new_state


@jax.jit
def ddsp_interpolation_kernels_process(
    x: jnp.ndarray,
    state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Process a buffer of samples using lax.scan.

    Args:
        x: input buffer, shape (T,)
        state: initial state (buffer, write_idx, delay_smooth)
        params: (delay_samples, interp_mode, smooth_coef)
            delay_samples:
                - if shape (T,), a per-sample delay curve is used.
                - if scalar, a constant delay is used.
            interp_mode: scalar integer
            smooth_coef: scalar

    Returns:
        y: output buffer, shape (T,)
        final_state: updated state tuple after processing.
    """
    buffer, write_idx, delay_smooth = state
    delay_samples, interp_mode, smooth_coef = params

    x = jnp.asarray(x, dtype=buffer.dtype)

    # Handle delay_samples which can be scalar or (T,)
    delay_samples = jnp.asarray(delay_samples, dtype=buffer.dtype)

    # If delay_samples is scalar, broadcast to match x.
    delay_samples = jnp.broadcast_to(delay_samples, x.shape)

    # interp_mode & smooth_coef are treated as scalars (no time axis).
    interp_mode = jnp.asarray(interp_mode, dtype=jnp.int32)
    smooth_coef = jnp.asarray(smooth_coef, dtype=buffer.dtype)

    init_state = (buffer, write_idx, delay_smooth)

    def scan_body(carry, xs):
        state_t = carry
        x_t, d_t = xs
        params_t = (d_t, interp_mode, smooth_coef)
        y_t, state_next = ddsp_interpolation_kernels_tick(x_t, state_t, params_t)
        return state_next, y_t

    final_state, y = lax.scan(scan_body, init_state, (x, delay_samples))
    return y, final_state


# ---------------------------------------------------------------------------
# Smoke test, plot, listen
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as onp
    import matplotlib.pyplot as plt

    try:
        import sounddevice as sd
        HAVE_SD = True
    except Exception:
        HAVE_SD = False

    # Basic smoke test: modulated delay on a sine.
    sample_rate = 48000
    duration = 1.0
    t = jnp.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)

    freq = 440.0
    x = jnp.sin(2.0 * jnp.pi * freq * t)

    max_delay = int(0.05 * sample_rate)  # 50 ms
    state, params = ddsp_interpolation_kernels_init(
        max_delay_samples=max_delay,
        initial_delay=0.5 * max_delay,
        interp_mode=3,      # 5th-order Lagrange
        smooth_coef=0.01,
        dtype=jnp.float32,
    )

    # Time-varying delay: gentle modulation (flanger-style).
    lfo_freq = 0.5  # Hz
    delay_center = 0.5 * max_delay
    delay_depth = 0.4 * max_delay

    delay_curve = delay_center + delay_depth * jnp.sin(
        2.0 * jnp.pi * lfo_freq * t
    )

    params_tv = (delay_curve, params[1], params[2])

    y, state_out = ddsp_interpolation_kernels_process(x, state, params_tv)

    # Convert to NumPy for plotting / playback.
    x_np = onp.asarray(x)
    y_np = onp.asarray(y)

    # Plot input vs output.
    plt.figure(figsize=(10, 4))
    n_plot = min(2000, x_np.shape[0])
    plt.plot(x_np[:n_plot], label="input", alpha=0.7)
    plt.plot(y_np[:n_plot], label="delayed (interp)", alpha=0.7)
    plt.title("ddsp_interpolation_kernels: fractional delay with 5th-order Lagrange")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Listen (if sounddevice is available).
    if HAVE_SD:
        print("Playing dry, then wet...")
        sd.play(x_np, samplerate=sample_rate, blocking=True)
        sd.play(y_np, samplerate=sample_rate, blocking=True)
        print("Done.")
    else:
        print("sounddevice not available; skipping audio playback.")
```

---

### Next things you might want to do

* Split out the interpolation kernels into a shared `interpolation_core.py` to be reused by:

  * wavetable/table_core readers
  * oscillators (saw/square/pulse) as fractional table readers
  * physical models (e.g., Karplus–Strong using this delay core)

* Add BLIT/BLEP/BLAMP-based discontinuity handling to build:

  * band-limited saw / square / pulse oscillators
  * triangle / parabolic oscillators using BLAMP

* Wrap similar GDSP-style cores for:

  * `table_core` (phasor-driven wavetable lookup using these kernels)
  * `phasor_core` integration (phase accumulation + interpolation lookup)
