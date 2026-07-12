//! Dispatcher: turn a parsed ToolCall into a real termux-* command invocation.
//!
//! Safety model:
//!   - only tools with an `exec` entry in the config can run at all
//!   - argv is built directly (std::process::Command) — never a shell,
//!     so there is no injection surface
//!   - by default the user confirms each command; --yes skips, --dry-run
//!     never executes

use std::io::Write;
use std::process::Command;

use anyhow::{bail, Context, Result};
use serde_json::Value;

use crate::tools::{ArgTemplate, Tool, ToolCall};

pub enum Outcome {
    Executed { argv: Vec<String>, stdout: String, status: i32 },
    DryRun { argv: Vec<String> },
    Declined,
    NoExec,
}

fn value_to_arg(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Build argv from the tool's exec template + the model's arguments.
pub fn build_argv(tool: &Tool, call: &ToolCall) -> Result<Option<Vec<String>>> {
    // [Rust Book Ch. 6] Option in, Option out: None simply means
    // "not an executable tool", which the caller renders differently
    // from an error.
    let exec = match &tool.exec {
        Some(e) => e,
        None => return Ok(None),
    };
    let mut argv = vec![exec.cmd.clone()];
    for tpl in &exec.args {
        match tpl {
            ArgTemplate::Lit { lit } => argv.push(lit.clone()),
            ArgTemplate::Arg { arg, flag, default, optional } => {
                match (call.arguments.get(arg), flag, default) {
                    // model supplied the arg
                    (Some(v), Some(f), _) => {
                        argv.push(f.clone());
                        argv.push(value_to_arg(v));
                    }
                    (Some(v), None, _) => argv.push(value_to_arg(v)),
                    // absent + default
                    (None, flag, Some(d)) => {
                        if let Some(f) = flag {
                            argv.push(f.clone());
                        }
                        argv.push(d.clone());
                    }
                    // absent + optional flag pair: skip entirely
                    (None, Some(_), None) => {}
                    // absent positional: skip when optional, else it's a
                    // config/schema mismatch worth surfacing
                    (None, None, None) => {
                        if !*optional {
                            bail!("tool '{}' missing required argument '{}'", tool.name, arg)
                        }
                    }
                }
            }
        }
    }
    Ok(Some(argv))
}

pub fn dispatch(tool: &Tool, call: &ToolCall, dry_run: bool, assume_yes: bool) -> Result<Outcome> {
    let argv = match build_argv(tool, call)? {
        Some(a) => a,
        None => return Ok(Outcome::NoExec),
    };

    if dry_run {
        return Ok(Outcome::DryRun { argv });
    }

    if !assume_yes {
        eprint!("  run `{}`? [Y/n] ", argv.join(" "));
        std::io::stderr().flush().ok();
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).ok();
        let ans = line.trim().to_lowercase();
        if !(ans.is_empty() || ans == "y" || ans == "yes") {
            return Ok(Outcome::Declined);
        }
    }

    // [Rust Book Ch. 9] `?` + context: every failure path says what it was doing
    let out = Command::new(&argv[0])
        .args(&argv[1..])
        .output()
        .with_context(|| format!("spawning {} (is Termux:API installed?)", argv[0]))?;

    Ok(Outcome::Executed {
        argv,
        stdout: String::from_utf8_lossy(&out.stdout).trim().to_string(),
        status: out.status.code().unwrap_or(-1),
    })
}
