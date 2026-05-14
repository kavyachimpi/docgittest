#multiply.py
PRACTICE TESTING NOW
CHANGES MADE TO DEVELOP BRANCH FOR TESTING
import numpy as np
import logging
from math import ceil, log2
import time

# FFT backend selection
_fft_backend = 'scipy'
try:
    import pyfftw
    from pyfftw.interfaces.numpy_fft import fft as fftw_fft, ifft as fftw_ifft
    _fft_backend = 'pyfftw'
    def fft(x):
        return fftw_fft(x)
    def ifft(x):
        return fftw_ifft(x)
except ImportError:
    from scipy.fft import fft, ifft
    def fft(x):
        return fft_orig(x)
    def ifft(x):
        return ifft_orig(x)
    fft_orig = fft
    ifft_orig = ifft

class NewlineFormatter(logging.Formatter):
    def format(self, record):
        return super().format(record) + '\n'

handler = logging.StreamHandler()
handler.setFormatter(NewlineFormatter('%(asctime)s - %(levelname)s - %(message)s'))
root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)

class FFTMultiplier:
    def __init__(self, chunk_bits=None, log_level=logging.INFO, enable_benchmark=False, max_retries=2, prefer_high_precision=False, fft_cache_enabled=False, fft_threads=1):
        self.chunk_bits = chunk_bits
        self.max_chunk = None
        self._set_log_level(log_level)
        self.enable_benchmark = enable_benchmark
        self.max_retries = max_retries
        self.prefer_high_precision = prefer_high_precision
        self.fft_cache_enabled = fft_cache_enabled
        self.fft_threads = fft_threads
        self._fft_cache = {} if fft_cache_enabled else None
        logging.info(f"Initialized FFTMultiplier (FFT backend: {_fft_backend}, threads: {fft_threads})")

    def _set_log_level(self, level):
        logging.getLogger().setLevel(level)


    def multiply(self, a, b, chunk_bits_override=None):
        self._validate_inputs(a, b)
        logging.info("Starting multiplication process")
        if self.enable_benchmark:
            overall_start = time.perf_counter()

        # Track and apply sign
        sign = 1
        if a < 0:
            sign *= -1
            a = -a
        if b < 0:
            sign *= -1
            b = -b

        # Hybrid: Use Karatsuba for small numbers
        if a.bit_length() < 2048 and b.bit_length() < 2048:
            logging.info("Using Karatsuba for small numbers")
            result = self._karatsuba(a, b)
            if self.enable_benchmark:
                elapsed = time.perf_counter() - overall_start
                logging.info(f"Karatsuba multiplication completed in {elapsed:.6f} seconds")
            return sign * result

        # FFT-based for large numbers, with error profiling and chunk tuning
        min_chunk = 8
        max_chunk = 14
        best_error = float('inf')
        best_result = None
        best_chunk = None
        best_dtype = None
        best_error_stats = None
        chunk_bits_list = [max_chunk] if chunk_bits_override is not None else list(range(max_chunk, min_chunk - 1, -1))
        for chunk_bits in chunk_bits_list:
            try:
                start_time = time.perf_counter()
                if a == 0 or b == 0:
                    logging.info("One operand is zero, returning 0")
                    return 0
                self.chunk_bits = chunk_bits
                self.max_chunk = (1 << self.chunk_bits) - 1
                logging.info(f"Trying chunk size: {self.chunk_bits} bits, FFT backend: {_fft_backend}")

                a_chunks = self._int_to_chunks(a)
                b_chunks = self._int_to_chunks(b)
                logging.debug(f"Chunk counts - a: {len(a_chunks)}, b: {len(b_chunks)}")

                min_fft_size = len(a_chunks) + len(b_chunks) - 1
                fft_size = 1 << ceil(log2(min_fft_size)) if min_fft_size > 0 else 1
                logging.debug(f"FFT size determined: {fft_size}")

                # Try float64, complex128, longdouble, clongdouble, complex256 if available
                dtype_list = [np.float64, np.complex128]
                if hasattr(np, 'longdouble'):
                    dtype_list.append(np.longdouble)
                if hasattr(np, 'clongdouble'):
                    dtype_list.append(np.clongdouble)
                if hasattr(np, 'complex256'):
                    dtype_list.append(np.complex256)
                for dtype in dtype_list:
                    a_padded = self._pad_chunks(a_chunks, fft_size).astype(dtype)
                    b_padded = self._pad_chunks(b_chunks, fft_size).astype(dtype)

                    # FFT result caching (optional, only for repeated operands)
                    fft_a = fft_b = None
                    cache_key_a = cache_key_b = None
                    if self.fft_cache_enabled:
                        cache_key_a = (tuple(a_padded), self.chunk_bits, fft_size, str(dtype))
                        cache_key_b = (tuple(b_padded), self.chunk_bits, fft_size, str(dtype))
                        fft_a = self._fft_cache.get(cache_key_a)
                        fft_b = self._fft_cache.get(cache_key_b)
                    if fft_a is None:
                        if _fft_backend == 'pyfftw':
                            fft_a = fft(a_padded, threads=self.fft_threads)
                        else:
                            fft_a = fft(a_padded)
                        if self.fft_cache_enabled:
                            self._fft_cache[cache_key_a] = fft_a
                    if fft_b is None:
                        if _fft_backend == 'pyfftw':
                            fft_b = fft(b_padded, threads=self.fft_threads)
                        else:
                            fft_b = fft(b_padded)
                        if self.fft_cache_enabled:
                            self._fft_cache[cache_key_b] = fft_b

                    fft_product = fft_a * fft_b
                    if _fft_backend == 'pyfftw':
                        c_padded = ifft(fft_product, threads=self.fft_threads).real
                    else:
                        c_padded = ifft(fft_product).real
                    logging.debug(f"FFT processing completed (dtype={dtype})")

                    # Floating-point error bounds propagation (FFT step)
                    c_coeffs = np.rint(c_padded).astype(np.int64)
                    abs_errors = np.abs(c_padded - c_coeffs)
                    min_err, max_err, mean_err = np.min(abs_errors), np.max(abs_errors), np.mean(abs_errors)
                    logging.info(f"[FFT error bounds] chunk_bits={chunk_bits}, dtype={dtype}: min={min_err:.6g}, max={max_err:.6g}, mean={mean_err:.6g}")

                    try:
                        self._check_precision(c_padded, c_coeffs)
                        # Vectorized carry handling for efficiency
                        result_chunks = self._handle_carries_vectorized(c_coeffs)
                        # Floating-point error bounds propagation (carry step)
                        carry_arr = np.array(result_chunks, dtype=np.float64)
                        carry_errs = np.abs(carry_arr - np.round(carry_arr))
                        min_carry_err, max_carry_err, mean_carry_err = np.min(carry_errs), np.max(carry_errs), np.mean(carry_errs)
                        logging.info(f"[Carry error bounds] chunk_bits={chunk_bits}, dtype={dtype}: min={min_carry_err:.6g}, max={max_carry_err:.6g}, mean={mean_carry_err:.6g}")
                        result = self._chunks_to_int(result_chunks)
                        elapsed = time.perf_counter() - start_time
                        logging.info(f"FFT multiplication completed successfully in {elapsed:.6f} seconds (dtype={dtype})")
                        if self.enable_benchmark:
                            overall_elapsed = time.perf_counter() - overall_start
                            logging.info(f"Total multiplication time: {overall_elapsed:.6f} seconds")
                        # Error profiling: keep best result (lowest max error)
                        if max_err < best_error:
                            best_error = max_err
                            best_result = result
                            best_chunk = chunk_bits
                            best_dtype = dtype
                            best_error_stats = (min_err, max_err, mean_err, min_carry_err, max_carry_err, mean_carry_err)
                    except ArithmeticError as e:
                        logging.warning(f"Precision loss detected with dtype={dtype}: {e}. Trying higher precision or smaller chunk size.")
                        continue
            except Exception as e:
                logging.error(f"FFT computation failed: {e}")
                continue

        if best_result is not None:
            logging.info(f"Selected chunk_bits={best_chunk}, dtype={best_dtype} with max FFT error={best_error:.6g}")
            if best_error_stats:
                logging.info(f"[Selected error stats] FFT min={best_error_stats[0]:.6g}, max={best_error_stats[1]:.6g}, mean={best_error_stats[2]:.6g}; Carry min={best_error_stats[3]:.6g}, max={best_error_stats[4]:.6g}, mean={best_error_stats[5]:.6g}")
            return sign * best_result

        # Error-correcting recovery: final re-computation with strictest settings before CRT
        logging.warning("Attempting error-correcting recovery: strictest settings before CRT fallback.")
        try:
            strict_chunk = min_chunk
            a_chunks = self._int_to_chunks(a)
            b_chunks = self._int_to_chunks(b)
            min_fft_size = len(a_chunks) + len(b_chunks) - 1
            fft_size = 1 << ceil(log2(min_fft_size)) if min_fft_size > 0 else 1
            dtype_list = []
            if hasattr(np, 'clongdouble'):
                dtype_list.append(np.clongdouble)
            if hasattr(np, 'longdouble'):
                dtype_list.append(np.longdouble)
            if hasattr(np, 'complex256'):
                dtype_list.append(np.complex256)
            dtype_list += [np.complex128, np.float64]
            for dtype in dtype_list:
                logging.info(f"Error-correcting retry: chunk_bits={strict_chunk}, dtype={dtype}")
                a_padded = self._pad_chunks(a_chunks, fft_size).astype(dtype)
                b_padded = self._pad_chunks(b_chunks, fft_size).astype(dtype)
                if _fft_backend == 'pyfftw':
                    fft_a = fft(a_padded, threads=self.fft_threads)
                    fft_b = fft(b_padded, threads=self.fft_threads)
                    fft_product = fft_a * fft_b
                    c_padded = ifft(fft_product, threads=self.fft_threads).real
                else:
                    fft_a = fft(a_padded)
                    fft_b = fft(b_padded)
                    fft_product = fft_a * fft_b
                    c_padded = ifft(fft_product).real
                c_coeffs = np.rint(c_padded).astype(np.int64)
                try:
                    self._check_precision(c_padded, c_coeffs)
                    result_chunks = self._handle_carries_vectorized(c_coeffs)
                    result = self._chunks_to_int(result_chunks)
                    logging.info("Error-correcting recovery succeeded.")
                    return sign * result
                except ArithmeticError as e:
                    logging.error(f"Error-correcting recovery failed: {e}")
                    continue
        except Exception as e:
            logging.error(f"Error-correcting recovery encountered an exception: {e}")

        # Fallback: Multi-modulus (CRT) if all FFT attempts and recovery fail
        logging.warning("Falling back to multi-modulus (CRT) multiplication.")
        return sign * self._crt_multiply(a, b)

        # Error-correcting recovery: final re-computation with strictest settings before CRT
        logging.warning("Attempting error-correcting recovery: strictest settings before CRT fallback.")
        try:
            strict_chunk = 8
            a_chunks = self._int_to_chunks(a)
            b_chunks = self._int_to_chunks(b)
            min_fft_size = len(a_chunks) + len(b_chunks) - 1
            fft_size = 1 << ceil(log2(min_fft_size)) if min_fft_size > 0 else 1
            dtype_list = []
            if hasattr(np, 'clongdouble'):
                dtype_list.append(np.clongdouble)
            if hasattr(np, 'longdouble'):
                dtype_list.append(np.longdouble)
            if hasattr(np, 'complex256'):
                dtype_list.append(np.complex256)
            dtype_list += [np.complex128, np.float64]
            for dtype in dtype_list:
                logging.info(f"Error-correcting retry: chunk_bits={strict_chunk}, dtype={dtype}")
                a_padded = self._pad_chunks(a_chunks, fft_size).astype(dtype)
                b_padded = self._pad_chunks(b_chunks, fft_size).astype(dtype)
                if _fft_backend == 'pyfftw':
                    fft_a = fft(a_padded, threads=self.fft_threads)
                    fft_b = fft(b_padded, threads=self.fft_threads)
                    fft_product = fft_a * fft_b
                    c_padded = ifft(fft_product, threads=self.fft_threads).real
                else:
                    fft_a = fft(a_padded)
                    fft_b = fft(b_padded)
                    fft_product = fft_a * fft_b
                    c_padded = ifft(fft_product).real
                c_coeffs = np.rint(c_padded).astype(np.int64)
                try:
                    self._check_precision(c_padded, c_coeffs)
                    result_chunks = self._handle_carries_vectorized(c_coeffs)
                    result = self._chunks_to_int(result_chunks)
                    logging.info("Error-correcting recovery succeeded.")
                    return sign * result
                except ArithmeticError as e:
                    logging.error(f"Error-correcting recovery failed: {e}")
                    continue
        except Exception as e:
            logging.error(f"Error-correcting recovery encountered an exception: {e}")

        # Fallback: Multi-modulus (CRT) if all FFT attempts and recovery fail
        logging.warning("Falling back to multi-modulus (CRT) multiplication.")
        return sign * self._crt_multiply(a, b)
    def _handle_carries_vectorized(self, coeffs):
        # Vectorized carry propagation for efficiency
        coeffs = np.array(coeffs, dtype=np.int64)
        carry = np.zeros_like(coeffs)
        result = np.zeros_like(coeffs)
        chunk_mask = self.max_chunk
        carry_val = 0
        for i in range(len(coeffs)):
            total = coeffs[i] + carry_val
            result[i] = total & chunk_mask
            carry_val = total >> self.chunk_bits
        # Handle remaining carry
        while carry_val > 0:
            result = np.append(result, carry_val & chunk_mask)
            carry_val >>= self.chunk_bits
        return result.tolist()
    def _handle_carries_kahan_checked(self, coeffs):
        # Kahan summation for carry propagation with explicit overflow check
        carry = 0
        c = 0.0
        result = []
        max_chunk_val = self.max_chunk
        for idx, coeff in enumerate(coeffs):
            y = coeff + carry - c
            t = int(round(y))
            # Explicit overflow check
            if abs(t) > (1 << 63) - 1:
                logging.error(f"Overflow detected in intermediate result at index {idx}: {t}")
                raise ArithmeticError("Overflow in carry propagation.")
            carry = t >> self.chunk_bits
            chunk = t & max_chunk_val
            result.append(chunk)
            c = (t - chunk) - (carry << self.chunk_bits)
        while carry > 0:
            chunk = carry & max_chunk_val
            result.append(chunk)
            carry >>= self.chunk_bits
        return result

    def _karatsuba(self, x, y):
        # Standard Karatsuba algorithm for small numbers
        if x < 10 or y < 10:
            return x * y
        m = max(x.bit_length(), y.bit_length()) // 2
        high1, low1 = x >> m, x & ((1 << m) - 1)
        high2, low2 = y >> m, y & ((1 << m) - 1)
        z0 = self._karatsuba(low1, low2)
        z1 = self._karatsuba((low1 + high1), (low2 + high2))
        z2 = self._karatsuba(high1, high2)
        return (z2 << (2 * m)) + ((z1 - z2 - z0) << m) + z0

    def _crt_multiply(self, a, b):
        # Multi-modulus FFT/CRT fallback for catastrophic precision loss
        # Use three large primes (pairwise coprime, < 2^30 for float safety)
        primes = [1073741789, 1073741827, 1073741831]
        results = []
        for p in primes:
            res = self._modular_fft_multiply(a, b, p)
            results.append(res)
        # Recombine using CRT
        x, m1, m2, m3 = results[0], primes[0], primes[1], primes[2]
        y, z = results[1], results[2]
        M = m1 * m2 * m3
        n1, n2, n3 = m2 * m3, m1 * m3, m1 * m2
        inv1 = pow(n1, -1, m1)
        inv2 = pow(n2, -1, m2)
        inv3 = pow(n3, -1, m3)
        result = (x * n1 * inv1 + y * n2 * inv2 + z * n3 * inv3) % M
        logging.info("CRT recombination completed.")
        return result

    def _modular_fft_multiply(self, a, b, mod):
        # Modular FFT-based multiplication for CRT fallback
        a_digits = self._int_to_chunks_mod(a, mod)
        b_digits = self._int_to_chunks_mod(b, mod)
        n = 1 << ceil(log2(len(a_digits) + len(b_digits) - 1))
        a_pad = np.array(a_digits + [0] * (n - len(a_digits)), dtype=np.float64)
        b_pad = np.array(b_digits + [0] * (n - len(b_digits)), dtype=np.float64)
        fa = fft(a_pad)
        fb = fft(b_pad)
        fc = fa * fb
        c = np.rint(ifft(fc).real).astype(np.int64) % mod
        # Standard carry handling
        carry = 0
        for i in range(len(c)):
            c[i] = (c[i] + carry) % mod
            carry = c[i] // mod
            c[i] = c[i] % mod
        # Convert back to integer
        res = 0
        for i, v in enumerate(c):
            res = (res + v * pow(1 << 16, i, mod)) % mod
        return res

    def _int_to_chunks_mod(self, x, mod):
        # Use 16-bit chunks for modular FFT
        mask = (1 << 16) - 1
        chunks = []
        while x > 0:
            chunks.append(x & mask)
            x >>= 16
        return chunks or [0]

    def _int_to_chunks(self, x):
        chunks = []
        while x > 0:
            chunks.append(x & self.max_chunk)
            x >>= self.chunk_bits
        return chunks or [0]

    def _pad_chunks(self, chunks, size):
        return np.array(chunks + [0] * (size - len(chunks)), dtype=np.float64)

    def _handle_carries_kahan(self, coeffs):
        # Kahan summation for carry propagation
        carry = 0
        c = 0.0
        result = []
        for coeff in coeffs:
            y = coeff + carry - c
            t = int(round(y))
            carry = t >> self.chunk_bits
            chunk = t & self.max_chunk
            result.append(chunk)
            c = (t - chunk) - (carry << self.chunk_bits)
        while carry > 0:
            chunk = carry & self.max_chunk
            result.append(chunk)
            carry >>= self.chunk_bits
        return result

    def _chunks_to_int(self, chunks):
        result = 0
        for i, chunk in enumerate(chunks):
            result += chunk << (self.chunk_bits * i)
        return result

    def _validate_inputs(self, a, b):
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError("Inputs must be integers")
        if isinstance(a, bool) or isinstance(b, bool):
            raise TypeError("Inputs must be integers, not bool")
        if abs(a) > 2**1024 or abs(b) > 2**1024:
            logging.warning("Input is extremely large; results may be unreliable due to float64/complex128 limits.")

    def _check_precision(self, computed, rounded):
        max_deviation = np.max(np.abs(computed - rounded))
        mean_deviation = np.mean(np.abs(computed - rounded))
        logging.info(f"Maximum floating-point deviation: {max_deviation:.6f}, Mean deviation: {mean_deviation:.6f}")
        if max_deviation > 0.5:
            logging.error("Significant precision loss detected! Consider using smaller chunk size or multi-modulus FFT.")
            raise ArithmeticError("Significant precision loss detected.")
        elif max_deviation > 0.1:
            logging.warning("Moderate precision deviation observed")
        if np.any(np.isnan(computed)) or np.any(np.isinf(computed)):
            logging.error("NaN or Inf detected in FFT results!")
            raise ArithmeticError("FFT produced NaN or Inf values.")

# Example usage
if __name__ == "__main__":
    multiplier = FFTMultiplier()

    # Accept user input for a and b, re-prompting on invalid input (now supports signed integers)
    while True:
        try:
            a = int(input("Enter the first integer (a): "))
            if isinstance(a, bool):
                raise ValueError("Input must not be a boolean.")
            break
        except Exception as e:
            print(f"Invalid input: {e}. Please enter a valid integer.")

    while True:
        try:
            b = int(input("Enter the second integer (b): "))
            if isinstance(b, bool):
                raise ValueError("Input must not be a boolean.")
            break
        except Exception as e:
            print(f"Invalid input: {e}. Please enter a valid integer.")

    print(f"Calculating {a} * {b}\n")
    try:
        result = multiplier.multiply(a, b)
        print(f"Result: {result}\n")
        assert result == a * b, "Test failed!"
    except Exception as e:
        print(f"Multiplication failed: {e}")