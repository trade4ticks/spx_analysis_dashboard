-- Run against spx_interpolated (main DB)
CREATE TABLE IF NOT EXISTS research_followups (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id     UUID NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
    question   TEXT NOT NULL,
    answer     TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
