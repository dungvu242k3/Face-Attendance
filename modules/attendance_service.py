"""Attendance business rules independent from HTTP and SQL details."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from config import Settings
from modules.attendance_rules import day_bounds, status_for
from modules.attendance_service_constants import VALID_CHECK_TYPES
from modules.database import Database
from modules.repositories.attendance_repository import AttendanceRepository


@dataclass(frozen=True)
class AttendanceResult:
    log_id: int
    check_time: datetime
    status: str
    duplicate: bool = False


class AttendanceService:
    def __init__(
        self, database: Database, repository: AttendanceRepository, settings: Settings
    ):
        self.database = database
        self.repository = repository
        self.settings = settings

    def record(
        self,
        employee_id: int,
        person_name: str,
        check_type: str,
        image_path: str | None = None,
        now: datetime | None = None,
    ) -> AttendanceResult:
        if check_type not in VALID_CHECK_TYPES:
            raise ValueError("Loại chấm công không hợp lệ")
        if employee_id is None:
            raise ValueError("Nhân viên chưa được liên kết với embedding")

        check_time = now or self.now()
        status = self.status_for(check_type, check_time)
        duplicate_since = check_time - timedelta(
            seconds=max(0, self.settings.duplicate_window_seconds)
        )

        with self.database.transaction() as connection:
            self.repository.lock_for_attendance(connection, employee_id, check_type)
            recent = self.repository.find_recent(
                connection, employee_id, check_type, duplicate_since
            )
            if recent:
                return AttendanceResult(
                    log_id=int(recent[0]),
                    check_time=recent[1],
                    status=recent[2],
                    duplicate=True,
                )

            log_id = self.repository.insert(
                connection,
                employee_id=employee_id,
                person_name=person_name,
                check_time=check_time,
                status=status,
                check_type=check_type,
                image_path=image_path,
            )
            return AttendanceResult(log_id, check_time, status)

    def status_for(self, check_type: str, check_time: datetime) -> str:
        return status_for(
            check_type,
            check_time,
            checkin_deadline=time(
                self.settings.checkin_deadline_hour,
                self.settings.checkin_deadline_minute,
            ),
            checkout_deadline=time(
                self.settings.checkout_deadline_hour,
                self.settings.checkout_deadline_minute,
            ),
        )

    def now(self) -> datetime:
        return datetime.now(ZoneInfo(self.settings.timezone_name)).replace(tzinfo=None)

    def day_bounds(self, day: datetime | None = None) -> tuple[datetime, datetime]:
        return day_bounds(day or self.now())
