-- Research runner tables — run once against the main (spx_interpolated) database

CREATE TABLE IF NOT EXISTS research_runs (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    name         TEXT        NOT NULL,
    question     TEXT        NOT NULL,
    config       JSONB       NOT NULL,
    status       TEXT        NOT NULL DEFAULT 'running',  -- running | complete | error
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    ai_summary   TEXT,
    error_msg    TEXT
);

CREATE TABLE IF NOT EXISTS research_results (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id        UUID        NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
    ticker        TEXT,                   -- NULL for cross-sectional queries
    x_col         TEXT        NOT NULL,
    y_col         TEXT        NOT NULL,
    analysis_type TEXT        NOT NULL,   -- correlation | decile | yearly_consistency | equity_curve | regression
    result        JSONB       NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_rr_run  ON research_results(run_id);
CREATE INDEX IF NOT EXISTS idx_rr_tick ON research_results(ticker);
CREATE INDEX IF NOT EXISTS idx_rr_type ON research_results(analysis_type);

CREATE TABLE IF NOT EXISTS research_series (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id      UUID        NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
    ticker      TEXT,
    x_col       TEXT        NOT NULL,
    y_col       TEXT,
    series_name TEXT        NOT NULL,   -- equity_curve_top | equity_curve_bottom | rolling_corr
    data        JSONB       NOT NULL,   -- [{date, value}, ...]
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_rs_run ON research_series(run_id);

CREATE TABLE IF NOT EXISTS research_charts (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id      UUID        NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
    ticker      TEXT,
    x_col       TEXT,
    y_col       TEXT,
    chart_type  TEXT        NOT NULL,
    title       TEXT,
    png_data    BYTEA,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_rc_run ON research_charts(run_id);
