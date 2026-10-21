from datetime import datetime, time
import unittest

from modules.attendance_rules import day_bounds, status_for


class AttendanceRulesTest(unittest.TestCase):
    def test_checkin_and_checkout_statuses(self):
        checkin_deadline = time(9, 0)
        checkout_deadline = time(18, 0)

        self.assertEqual(
            status_for(
                "checkin",
                datetime(2026, 10, 20, 8, 59),
                checkin_deadline,
                checkout_deadline,
            ),
            "present",
        )
        self.assertEqual(
            status_for(
                "checkin",
                datetime(2026, 10, 20, 9, 0),
                checkin_deadline,
                checkout_deadline,
            ),
            "late",
        )
        self.assertEqual(
            status_for(
                "checkout",
                datetime(2026, 10, 20, 18, 0),
                checkin_deadline,
                checkout_deadline,
            ),
            "present",
        )
        self.assertEqual(
            status_for(
                "checkout",
                datetime(2026, 10, 20, 18, 1),
                checkin_deadline,
                checkout_deadline,
            ),
            "overtime",
        )

    def test_invalid_check_type(self):
        with self.assertRaises(ValueError):
            status_for("other", datetime.now(), time(9), time(18))

    def test_day_bounds(self):
        start, end = day_bounds(datetime(2026, 10, 20, 14, 30))
        self.assertEqual(start, datetime(2026, 10, 20, 0, 0))
        self.assertEqual(end, datetime(2026, 10, 21, 0, 0))
