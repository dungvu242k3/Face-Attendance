"""Pure attendance rules that are easy to test without a database."""

from __future__ import annotations

from datetime import datetime, time, timedelta

from modules.attendance_service_constants import CHECKIN, CHECKOUT


def status_for(
    check_type: str,
    check_time: datetime,
    checkin_deadline: time,
    checkout_deadline: time,
) -> str:
    if check_type == CHECKIN:
        return "present" if check_time.time() < checkin_deadline else "late"
    if check_type == CHECKOUT:
        return "present" if check_time.time() <= checkout_deadline else "overtime"
    raise ValueError("Loại chấm công không hợp lệ")


def day_bounds(day: datetime | None = None) -> tuple[datetime, datetime]:
    current = day or datetime.now()
    start = datetime.combine(current.date(), time.min)
    return start, start + timedelta(days=1)
