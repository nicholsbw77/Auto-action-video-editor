import pytest
from gui.widgets import parse_timecode, format_timecode


class TestTimecode:
    def test_parse_timecode(self):
        assert parse_timecode("00:01:30.50") == 90.5

    def test_parse_timecode_hours(self):
        assert parse_timecode("01:00:00.00") == 3600.0

    def test_parse_timecode_zero(self):
        assert parse_timecode("00:00:00.00") == 0.0

    def test_format_timecode(self):
        assert format_timecode(90.5) == "00:01:30.50"

    def test_format_timecode_hours(self):
        assert format_timecode(3661.25) == "01:01:01.25"

    def test_roundtrip(self):
        for val in [0.0, 1.5, 90.5, 3600.0, 7261.99]:
            assert abs(parse_timecode(format_timecode(val)) - val) < 0.01

    def test_format_timecode_centisecond_overflow(self):
        """Regression: 59.999 should become 01:00:00.00, not 00:00:59.100"""
        result = format_timecode(59.999)
        assert result == "00:01:00.00"

    def test_format_timecode_boundary(self):
        """Test edge case at exact minute boundary"""
        result = format_timecode(59.995)
        assert result == "00:01:00.00"

    def test_format_timecode_sub_overflow(self):
        """Test that 3599.999 rolls over to 01:00:00.00"""
        result = format_timecode(3599.999)
        assert result == "01:00:00.00"
