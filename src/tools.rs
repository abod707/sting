//! Tool schema handling: loading the tools config, converting MCP-style
//! JSON Schema to needle's compact parameter style, snake_case name
//! normalization (port of needle/model/run.py::normalize_tools), and
//! building the compact tools JSON the model was trained on.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::{json, Map, Value};

/// One executable tool: the schema shown to the model + how to run it.
// [Rust Book Ch. 5] Struct with owned Strings — the config file's JSON is
// parsed once and each tool OWNS its data (no lifetimes to juggle).
pub struct Tool {
    pub name: String,
    pub description: String,
    /// needle-style parameters: {"arg": {"type","description","required"}}
    pub parameters: Value,
    /// None = model-only tool (we print the call but can't execute it)
    pub exec: Option<ExecSpec>,
}

#[derive(Clone, serde::Deserialize)]
pub struct ExecSpec {
    /// argv[0], e.g. "termux-battery-status". Never run through a shell.
    pub cmd: String,
    /// argument template, e.g. [{"lit":"-t"},{"arg":"title"}]
    #[serde(default)]
    pub args: Vec<ArgTemplate>,
}

#[derive(Clone, serde::Deserialize)]
#[serde(untagged)]
pub enum ArgTemplate {
    /// {"lit": "-d"} — literal argv token
    Lit { lit: String },
    /// {"arg": "title"} — required value; {"arg": "x", "default": "1000"} —
    /// value with fallback; {"arg": "lang", "flag": "-l"} — optional flag+value
    /// pair included only when supplied; {"arg": "file", "optional": true} —
    /// optional positional, skipped when absent.
    Arg {
        arg: String,
        #[serde(default)]
        flag: Option<String>,
        #[serde(default)]
        default: Option<String>,
        #[serde(default)]
        optional: bool,
    },
}

pub struct ToolSet {
    pub tools: Vec<Tool>,
}

impl ToolSet {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading tools config {}", path.display()))?;
        let root: Value = serde_json::from_str(&raw)?;
        // accept either {"tools": [...]} or a bare [...]
        let arr = root
            .get("tools")
            .and_then(|t| t.as_array())
            .or_else(|| root.as_array())
            .context("tools config must be a list or {\"tools\": [...]}")?
            .clone();

        let mut tools = Vec::new();
        for item in arr {
            let name = item["name"].as_str().context("tool missing name")?.to_string();
            let description = item["description"].as_str().unwrap_or("").to_string();
            let parameters = normalize_parameters(item.get("parameters").unwrap_or(&json!({})));
            let exec = match item.get("exec") {
                Some(e) => Some(serde_json::from_value(e.clone())?),
                None => None,
            };
            tools.push(Tool { name, description, parameters, exec });
        }
        Ok(Self { tools })
    }

    /// Compact tools JSON for the model prompt, with snake_case names.
    /// Returns (json, map from snake_case back to the original names).
    pub fn to_model_json(&self) -> (String, HashMap<String, String>) {
        let all: Vec<usize> = (0..self.tools.len()).collect();
        self.model_json_for(&all)
    }

    /// Same, but only for the tools at `indices` (retrieval shortlist).
    pub fn model_json_for(&self, indices: &[usize]) -> (String, HashMap<String, String>) {
        let mut name_map = HashMap::new();
        let mut arr = Vec::new();
        for &i in indices {
            let t = &self.tools[i];
            let snake = to_snake_case(&t.name);
            name_map.insert(snake.clone(), t.name.clone());
            arr.push(json!({
                "name": snake,
                "description": t.description,
                "parameters": t.parameters,
            }));
        }
        (Value::Array(arr).to_string(), name_map)
    }

    pub fn get(&self, name: &str) -> Option<&Tool> {
        // [Rust Book Ch. 13] iterator chain instead of a hand-rolled loop
        self.tools.iter().find(|t| t.name == name)
    }
}

/// One tool's compact schema JSON — the text embedded for retrieval.
/// Matches the string the contrastive head was trained on (full tool object,
/// compact separators, snake_case name).
pub fn single_tool_model_json(tool: &Tool) -> String {
    json!({
        "name": to_snake_case(&tool.name),
        "description": tool.description,
        "parameters": tool.parameters,
    })
    .to_string()
}

/// Accept both needle-style parameters and MCP/JSON-Schema style
/// ({"type":"object","properties":{...},"required":[...]}) — normalize to
/// needle-style, which is what the model was finetuned on.
pub fn normalize_parameters(params: &Value) -> Value {
    let obj = match params.as_object() {
        Some(o) => o,
        None => return json!({}),
    };
    // MCP/JSON-Schema detection: has "properties" and type == "object"
    if let Some(props) = obj.get("properties").and_then(|p| p.as_object()) {
        let required: Vec<&str> = obj
            .get("required")
            .and_then(|r| r.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        let mut out = Map::new();
        for (key, spec) in props {
            let mut entry = Map::new();
            entry.insert(
                "type".into(),
                spec.get("type").cloned().unwrap_or(json!("string")),
            );
            if let Some(d) = spec.get("description") {
                entry.insert("description".into(), d.clone());
            }
            entry.insert("required".into(), json!(required.contains(&key.as_str())));
            out.insert(key.clone(), Value::Object(entry));
        }
        return Value::Object(out);
    }
    params.clone()
}

/// Port of needle's to_snake_case (dataset/tokenizer.py).
pub fn to_snake_case(name: &str) -> String {
    // replace non-alphanumeric/underscore with underscores
    let mut s: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    // insert _ before uppercase following lowercase/digit
    let mut out = String::with_capacity(s.len() + 4);
    let chars: Vec<char> = s.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if c.is_ascii_uppercase() && i > 0 {
            let prev = chars[i - 1];
            let next_lower = chars.get(i + 1).map(|n| n.is_ascii_lowercase()).unwrap_or(false);
            if prev.is_ascii_lowercase() || prev.is_ascii_digit() {
                out.push('_');
            } else if prev.is_ascii_uppercase() && next_lower {
                out.push('_');
            }
        }
        out.push(c);
    }
    s = out.to_lowercase();
    // collapse multiple underscores, trim edges
    let mut collapsed = String::with_capacity(s.len());
    let mut prev_us = false;
    for c in s.chars() {
        if c == '_' {
            if !prev_us {
                collapsed.push('_');
            }
            prev_us = true;
        } else {
            collapsed.push(c);
            prev_us = false;
        }
    }
    collapsed.trim_matches('_').to_string()
}

/// A parsed tool call from the model's output.
#[derive(Debug)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Map<String, Value>,
}

/// Parse the model's output JSON into calls, mapping snake_case names back
/// to the originals.
pub fn parse_calls(output: &str, name_map: &HashMap<String, String>) -> Vec<ToolCall> {
    let parsed: Value = match serde_json::from_str(output.trim()) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let arr = match parsed {
        Value::Array(a) => a,
        Value::Object(_) => vec![parsed],
        _ => return Vec::new(),
    };
    let mut calls = Vec::new();
    for item in arr {
        let snake = item["name"].as_str().unwrap_or("").to_string();
        if snake.is_empty() {
            continue;
        }
        let name = name_map.get(&snake).cloned().unwrap_or(snake);
        let arguments = item["arguments"].as_object().cloned().unwrap_or_default();
        calls.push(ToolCall { name, arguments });
    }
    calls
}
