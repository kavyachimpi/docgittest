# Testcase.py
## MODIFIED THE COMMENTS TO TEST FROM USER2
## removing the chnages now
from multiply import FFTMultiplier
import pytest

## added details as part of final test

class TestFFTMultiplier:
    
    # Positive test cases
    def test_small_positive(self):
        m = FFTMultiplier()
        assert m.multiply(12, 34) == 408
        
    def test_commutative_property(self):
        m = FFTMultiplier()
        a = 123456789
        b = 987654321
        assert m.multiply(a, b) == m.multiply(b, a) == a * b

    def test_large_positive(self):
        m = FFTMultiplier()
        a = 12345678901234567890
        b = 98765432109876543210
        assert m.multiply(a, b) == a * b

    def test_zero_a(self):
        m = FFTMultiplier()
        assert m.multiply(0, 123456) == 0

    def test_zero_b(self):
        m = FFTMultiplier()
        assert m.multiply(123456, 0) == 0

    def test_one_a(self):
        m = FFTMultiplier()
        assert m.multiply(1, 987654321) == 987654321

    def test_one_b(self):
        m = FFTMultiplier()
        assert m.multiply(987654321, 1) == 987654321

    def test_negative_a(self):
        m = FFTMultiplier()
        assert m.multiply(-12345, 6789) == -12345 * 6789

    def test_negative_b(self):
        m = FFTMultiplier()
        assert m.multiply(12345, -6789) == 12345 * -6789

    def test_both_negative(self):
        m = FFTMultiplier()
        assert m.multiply(-12345, -6789) == 12345 * 6789

    # Negative test cases
    def test_non_integer_a(self):
        m = FFTMultiplier()
        with pytest.raises(TypeError):
            m.multiply('abc', 123)

    def test_non_integer_b(self):
        m = FFTMultiplier()
        with pytest.raises(TypeError):
            m.multiply(123, 4.56)

    def test_bool_input(self):
        m = FFTMultiplier()
        with pytest.raises(TypeError):
            m.multiply(True, 123)

    def test_extremely_large_input_warning(self):
        m = FFTMultiplier()
        a = 2**1025
        b = 2
        # Should not raise, but logs a warning; result is still correct
        assert m.multiply(a, b) == a * b
