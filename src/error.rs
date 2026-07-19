/// Domain-level errors for API boundaries. Internal functions still use
/// `anyhow::Result` for propagation; this enum is for callers that need to
/// branch on error kind (parse_calls, Model::load, tool dispatch).
#[derive(Debug)]
pub enum StingError {
    /// Model output was not valid JSON (or not a tool-call structure).
    ParseFailed { message: String, raw: String },
    /// Model checkpoint or config file missing / unreadable.
    MissingModel(String),
    /// Tool config and model disagree (e.g. retrieval head mismatch).
    ConfigMismatch(String),
}

impl std::fmt::Display for StingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseFailed { message, .. } => write!(f, "parse failed: {message}"),
            Self::MissingModel(msg) => write!(f, "missing model: {msg}"),
            Self::ConfigMismatch(msg) => write!(f, "config mismatch: {msg}"),
        }
    }
}
