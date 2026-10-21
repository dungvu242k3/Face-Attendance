-- PostgreSQL must have the pgvector extension installed before this script runs.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS employees (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT,
    phone TEXT,
    position TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS face_embeddings_512 (
    id SERIAL PRIMARY KEY,
    employee_id INT REFERENCES employees(id) ON DELETE CASCADE,
    person_name TEXT NOT NULL,
    embedding vector(512) NOT NULL,
    filename TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS attendance_logs_512 (
    id SERIAL PRIMARY KEY,
    employee_id INT REFERENCES employees(id) ON DELETE SET NULL,
    person_name TEXT,
    check_time TIMESTAMP NOT NULL DEFAULT NOW(),
    status TEXT NOT NULL CHECK (status IN ('present', 'late', 'overtime')),
    image_path TEXT,
    check_type TEXT NOT NULL CHECK (check_type IN ('checkin', 'checkout'))
);

CREATE INDEX IF NOT EXISTS idx_employees_name ON employees (name);
CREATE INDEX IF NOT EXISTS idx_attendance_logs_check_time ON attendance_logs_512 (check_time DESC);
CREATE INDEX IF NOT EXISTS idx_attendance_logs_employee_type
    ON attendance_logs_512 (employee_id, check_type, check_time DESC);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_employee_id
    ON face_embeddings_512 (employee_id);

-- For larger datasets, add a pgvector HNSW/IVFFlat index after validating the
-- installed pgvector version and the expected query volume.
