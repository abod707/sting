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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Map;

    fn make_exec(args: Vec<ArgTemplate>) -> Option<crate::tools::ExecSpec> {
        Some(crate::tools::ExecSpec { cmd: "test-cmd".into(), args })
    }

    fn lit(s: &str) -> ArgTemplate {
        ArgTemplate::Lit { lit: s.into() }
    }

    fn arg(name: &str) -> ArgTemplate {
        ArgTemplate::Arg { arg: name.into(), flag: None, default: None, optional: false }
    }

    fn arg_default(name: &str, default: &str) -> ArgTemplate {
        ArgTemplate::Arg { arg: name.into(), flag: None, default: Some(default.into()), optional: false }
    }

    fn arg_flag(name: &str, flag: &str) -> ArgTemplate {
        ArgTemplate::Arg { arg: name.into(), flag: Some(flag.into()), default: None, optional: false }
    }

    fn arg_flag_default(name: &str, flag: &str, default: &str) -> ArgTemplate {
        ArgTemplate::Arg { arg: name.into(), flag: Some(flag.into()), default: Some(default.into()), optional: false }
    }

    fn arg_optional(name: &str) -> ArgTemplate {
        ArgTemplate::Arg { arg: name.into(), flag: None, default: None, optional: true }
    }

    fn tool(name: &str, exec: Option<crate::tools::ExecSpec>) -> crate::tools::Tool {
        crate::tools::Tool {
            name: name.into(),
            description: String::new(),
            parameters: Default::default(),
            exec,
        }
    }

    #[test]
    fn build_argv_no_exec() {
        let t = tool("test", None);
        let call = crate::tools::ToolCall { name: "test".into(), arguments: Map::new() };
        assert!(build_argv(&t, &call).unwrap().is_none());
    }

    #[test]
    fn build_argv_simple_lit() {
        let t = tool("test", make_exec(vec![lit("on")]));
        let call = crate::tools::ToolCall { name: "test".into(), arguments: Map::new() };
        let argv = build_argv(&t, &call).unwrap().unwrap();
        assert_eq!(argv, vec!["test-cmd", "on"]);
    }

    #[test]
    fn build_argv_supplied_arg() {
        let mut args = Map::new();
        args.insert("action".into(), serde_json::Value::String("on".into()));
        let t = tool("test", make_exec(vec![arg("action")]));
        let call = crate::tools::ToolCall { name: "test".into(), arguments: args };
        let argv = build_argv(&t, &call).unwrap().unwrap();
        assert_eq!(argv, vec!["test-cmd", "on"]);
    }

    #[test]
    fn build_argv_flag_arg() {
        let mut args = Map::new();
        args.insert("title".into(), serde_json::Value::String("hello".into()));
        let t = tool("test", make_exec(vec![arg_flag("title", "-t")]));
        let call = crate::tools::ToolCall { name: "test".into(), arguments: args };
        let argv = build_argv(&t, &call).unwrap().unwrap();
        assert_eq!(argv, vec!["test-cmd", "-t", "hello"]);
    }

    #[test]
    fn build_argv_missing_required() {
        let t = tool("test", make_exec(vec![arg("action")]));
        let call = crate::tools::ToolCall { name: "test".into(), arguments: Map::new() };
        let err = build_argv(&t, &call).unwrap_err();
        assert!(err.to_string().contains("missing required argument"));
    }

    #[test]
    fn build_argv_default_value() {
        let t = tool("test", make_exec(vec![arg_default("timeout", "30")]));
        let call = crate::tools::ToolCall { name: "test".into(), arguments: Map::new() };
        let argv = build_argv(&t, &call).unwrap().unwrap();
        assert_eq!(argv, vec!["test-cmd", "30"]);
    }

    #[test]
    fn build_argv_supplied_overrides_default() {
        let mut args = Map::new();
        args.insert("timeout".into(), serde_json::Value::String("60".into()));
        let t = tool("test", make_exec(vec![arg_default("timeout", "30")]));
        let call = crate::tools::ToolCall { name: "test".into(), arguments: args };
        let argv = build_argv(&t, &call).unwrap().unwrap();
        assert_eq!(argv, vec!["test-cmd", "60"]);
    }

    #[test]
    fn build_argv_flag_with_default() {
        let t = tool("test", make_exec(vec![arg_flag_default("lang", "-l", "en")]));
        let call = crate::tools::ToolCall { name: "test".into(), arguments: Map::new() };
        let argv = build_argv(&t, &call).unwrap().unwrap();
        assert_eq!(argv, vec!["test-cmd", "-l", "en"]);
    }

    #[test]
    fn build_argv_optional_skipped() {
        let t = tool("test", make_exec(vec![arg_optional("file")]));
        let call = crate::tools::ToolCall { name: "test".into(), arguments: Map::new() };
        let argv = build_argv(&t, &call).unwrap().unwrap();
        assert_eq!(argv, vec!["test-cmd"]);
    }

    #[test]
    fn build_argv_optional_supplied() {
        let mut args = Map::new();
        args.insert("file".into(), serde_json::Value::String("test.txt".into()));
        let t = tool("test", make_exec(vec![arg_optional("file")]));
        let call = crate::tools::ToolCall { name: "test".into(), arguments: args };
        let argv = build_argv(&t, &call).unwrap().unwrap();
        assert_eq!(argv, vec!["test-cmd", "test.txt"]);
    }

    #[test]
    fn build_argv_flag_optional_pair_absent() {
        let t = tool("test", make_exec(vec![ArgTemplate::Arg {
            arg: "lang".into(),
            flag: Some("-l".into()),
            default: None,
            optional: true,
        }]));
        let call = crate::tools::ToolCall { name: "test".into(), arguments: Map::new() };
        let argv = build_argv(&t, &call).unwrap().unwrap();
        assert_eq!(argv, vec!["test-cmd"]);
    }

    #[test]
    fn value_to_arg_string() {
        assert_eq!(value_to_arg(&serde_json::Value::String("hello".into())), "hello");
    }

    #[test]
    fn value_to_arg_number() {
        assert_eq!(value_to_arg(&serde_json::json!(42)), "42");
    }
}
