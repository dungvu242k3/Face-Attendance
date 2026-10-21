-- Run this once for databases created by an older version of the project.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE INDEX IF NOT EXISTS idx_employees_name ON employees (name);
CREATE INDEX IF NOT EXISTS idx_attendance_logs_check_time ON attendance_logs_512 (check_time DESC);
CREATE INDEX IF NOT EXISTS idx_attendance_logs_employee_type
    ON attendance_logs_512 (employee_id, check_type, check_time DESC);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_employee_id
    ON face_embeddings_512 (employee_id);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'attendance_status_valid'
    ) THEN
        ALTER TABLE attendance_logs_512
            ADD CONSTRAINT attendance_status_valid
            CHECK (status IN ('present', 'late', 'overtime')) NOT VALID;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'attendance_check_type_valid'
    ) THEN
        ALTER TABLE attendance_logs_512
            ADD CONSTRAINT attendance_check_type_valid
            CHECK (check_type IN ('checkin', 'checkout')) NOT VALID;
    END IF;
END $$;
