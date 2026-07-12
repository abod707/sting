//! sting — a tiny on-device function-calling assistant for Termux.
//!
//! Pipeline: natural-language request -> Needle-26M (finetuned) picks a tool
//! and fills its arguments -> dispatcher runs the matching termux-* command.
//!
//! Usage:
//!   sting "turn on the flashlight"
//!   sting --dry-run "set brightness to 180"
//!   sting --tools my_tools.json --repl
//!   sting verify-tokenizer spec_parity.jsonl      (dev: tokenizer parity test)
//!
//! Model files are looked up in --model-dir, $STING_HOME, or ./model.

mod dispatch;
mod generate;
mod model;
mod retrieval;
mod tokenizer;
mod tools;

use std::io::{BufRead, Write};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use candle_core::Device;

use crate::dispatch::Outcome;
use crate::model::Model;
use crate::tokenizer::Tokenizer;
use crate::tools::ToolSet;

// [Rust Book Ch. 5] Plain struct for CLI options; hand-rolled parsing keeps
// the dependency tree small and the code readable end to end.
struct Opts {
    query: Option<String>,
    tools_path: Option<PathBuf>,
    /// dev/power-user: use the file's JSON string verbatim as the model's
    /// tools list (no re-serialization, no snake_case mapping)
    tools_raw_path: Option<PathBuf>,
    model_dir: Option<PathBuf>,
    repl: bool,
    dry_run: bool,
    assume_yes: bool,
    raw: bool,
    no_constrain: bool,
    timing: bool,
    /// shortlist size for tool retrieval; 0 disables retrieval
    top_k: usize,
    verify_tokenizer: Option<PathBuf>,
}

fn usage() -> ! {
    eprintln!(
        "sting — tiny on-device tool calling (Needle 26M)

USAGE:
  sting [FLAGS] \"your request\"
  sting --repl [FLAGS]
  sting verify-tokenizer <parity.jsonl>

FLAGS:
  --tools <file>      tools config JSON (default: $STING_HOME/tools.json or ./tools.json)
  --model-dir <dir>   model directory (default: $STING_HOME/model or ./model)
  --repl              interactive mode
  --dry-run           print the command instead of running it
  -y, --yes           execute without confirmation
  --raw               print the model's JSON output only (no dispatch)
  --no-constrain      disable grammar-constrained decoding
  --top-k <n>         retrieval shortlist size (default 6; 0 = all tools)
  --time              print prefill/decode timing"
    );
    std::process::exit(2);
}

fn parse_args() -> Opts {
    let mut opts = Opts {
        query: None,
        tools_path: None,
        tools_raw_path: None,
        model_dir: None,
        repl: false,
        dry_run: false,
        assume_yes: false,
        raw: false,
        no_constrain: false,
        timing: false,
        top_k: 6,
        verify_tokenizer: None,
    };
    let mut args = std::env::args().skip(1);
    // [Rust Book Ch. 8] while let + iterator: consume args by hand so flags
    // with values (--tools X) can pull their argument off the iterator.
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tools" => opts.tools_path = args.next().map(PathBuf::from),
            "--tools-raw" => opts.tools_raw_path = args.next().map(PathBuf::from),
            "--model-dir" => opts.model_dir = args.next().map(PathBuf::from),
            "--repl" => opts.repl = true,
            "--dry-run" => opts.dry_run = true,
            "-y" | "--yes" => opts.assume_yes = true,
            "--raw" => opts.raw = true,
            "--no-constrain" => opts.no_constrain = true,
            "--top-k" => {
                opts.top_k = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("--top-k needs a number");
                        usage();
                    })
            }
            "--time" => opts.timing = true,
            "verify-tokenizer" => opts.verify_tokenizer = args.next().map(PathBuf::from),
            "-h" | "--help" => usage(),
            s if s.starts_with('-') => {
                eprintln!("unknown flag: {s}");
                usage();
            }
            _ => {
                // [Rust Book Ch. 4] `arg` is MOVED into the Option here — no
                // clone. Each loop iteration owns its String and may give it away.
                opts.query = Some(arg);
            }
        }
    }
    opts
}

fn resolve(base: Option<PathBuf>, env_sub: &str, fallback: &str) -> PathBuf {
    if let Some(p) = base {
        return p;
    }
    if let Ok(home) = std::env::var("STING_HOME") {
        return PathBuf::from(home).join(env_sub);
    }
    PathBuf::from(fallback)
}

fn main() -> Result<()> {
    let opts = parse_args();

    if let Some(parity_path) = &opts.verify_tokenizer {
        return verify_tokenizer(parity_path);
    }

    let model_dir = resolve(opts.model_dir.clone(), "model", "model");
    let tools_path = resolve(opts.tools_path.clone(), "tools.json", "tools.json");

    let dev = Device::Cpu;
    let t0 = std::time::Instant::now();
    let tok = Tokenizer::from_spec_file(&model_dir.join("tokenizer_spec.json"))?;
    let model = Model::load(&model_dir, &dev)?;

    // --tools-raw bypasses config parsing entirely (dev / parity testing)
    let (toolset, raw_tools_json) = match &opts.tools_raw_path {
        Some(p) => {
            let raw = std::fs::read_to_string(p)
                .with_context(|| format!("reading {}", p.display()))?;
            (ToolSet { tools: Vec::new() }, Some(raw.trim().to_string()))
        }
        None => (ToolSet::load(&tools_path)?, None),
    };
    if opts.timing {
        eprintln!("[load {} ms]", t0.elapsed().as_millis());
    }

    let use_retrieval = raw_tools_json.is_none()
        && opts.top_k > 0
        && toolset.tools.len() > opts.top_k
        && model.has_retrieval_head();
    let mut retriever = use_retrieval.then(|| retrieval::Retriever::new(&tools_path, &model_dir));

    // [Rust Book Ch. 13] a closure capturing its environment: model and
    // tokenizer by shared borrow, the retriever by MUTABLE borrow (its
    // embedding cache updates) — which is why run_one must be `mut`.
    let mut run_one = |query: &str| -> Result<()> {
        // pick which tools this query's prompt will contain
        let (tools_json, name_map) = match (&raw_tools_json, retriever.as_mut()) {
            (Some(raw), _) => (raw.clone(), Default::default()),
            (None, Some(r)) => {
                let keep = r.shortlist(&model, &tok, &toolset, query, opts.top_k, &dev)?;
                if opts.timing {
                    let names: Vec<&str> =
                        keep.iter().map(|&i| toolset.tools[i].name.as_str()).collect();
                    eprintln!("[retrieval: {}]", names.join(", "));
                }
                toolset.model_json_for(&keep)
            }
            (None, None) => toolset.to_model_json(),
        };

        let (out, stats) = generate::generate(
            &model,
            &tok,
            query,
            &tools_json,
            !opts.no_constrain,
            &dev,
        )?;
        if opts.timing {
            eprintln!(
                "[prefill {} tok / {} ms | decode {} tok / {} ms ({:.1} tok/s)]",
                stats.prefill_tokens,
                stats.prefill_ms,
                stats.generated_tokens,
                stats.decode_ms,
                stats.generated_tokens as f64 / (stats.decode_ms.max(1) as f64 / 1000.0),
            );
        }

        if opts.raw {
            println!("{out}");
            return Ok(());
        }

        let calls = tools::parse_calls(&out, &name_map);
        if calls.is_empty() {
            let shown = if out.trim().is_empty() { "<empty>" } else { out.trim() };
            println!("(no tool call — model output: {shown})");
            return Ok(());
        }

        for call in &calls {
            let args_json = serde_json::Value::Object(call.arguments.clone());
            println!("→ {}({})", call.name, args_json);
            let tool = match toolset.get(&call.name) {
                Some(t) => t,
                None => {
                    println!("  (model chose a tool not in the config — skipping)");
                    continue;
                }
            };
            match dispatch::dispatch(tool, call, opts.dry_run, opts.assume_yes)? {
                Outcome::Executed { argv, stdout, status } => {
                    if status != 0 {
                        eprintln!("  exit {status}: {}", argv.join(" "));
                    }
                    if !stdout.is_empty() {
                        println!("{stdout}");
                    }
                }
                Outcome::DryRun { argv } => println!("  would run: {}", argv.join(" ")),
                Outcome::Declined => println!("  skipped."),
                Outcome::NoExec => println!("  (no exec mapping for this tool — call printed only)"),
            }
        }
        Ok(())
    };

    if opts.repl {
        eprintln!("sting repl — type a request, or 'q' to quit");
        let stdin = std::io::stdin();
        loop {
            eprint!("» ");
            std::io::stderr().flush().ok();
            let mut line = String::new();
            if stdin.lock().read_line(&mut line)? == 0 {
                break;
            }
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if line == "q" || line == "quit" || line == "exit" {
                break;
            }
            if let Err(e) = run_one(line) {
                eprintln!("error: {e:#}");
            }
        }
        return Ok(());
    }

    match &opts.query {
        Some(q) => run_one(q),
        None => usage(),
    }
}

/// Dev tool: verify the pure-Rust tokenizer against Python SentencePiece.
/// Input JSONL rows: {"text": "...", "ids": [..], "decoded": "..."} from Python.
fn verify_tokenizer(parity_path: &std::path::Path) -> Result<()> {
    let spec_env =
        std::env::var("STING_SPEC").unwrap_or_else(|_| "model/tokenizer_spec.json".into());
    let tok = Tokenizer::from_spec_file(std::path::Path::new(&spec_env))?;

    let file = std::fs::File::open(parity_path)
        .with_context(|| format!("opening {}", parity_path.display()))?;
    let reader = std::io::BufReader::new(file);

    let mut total = 0u64;
    let mut enc_fail = 0u64;
    let mut dec_fail = 0u64;
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: serde_json::Value = serde_json::from_str(&line)?;
        let text = row["text"].as_str().context("row missing text")?;
        let want: Vec<u32> = row["ids"]
            .as_array()
            .context("row missing ids")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as u32)
            .collect();
        total += 1;

        let got = tok.encode(text);
        if got != want {
            enc_fail += 1;
            if enc_fail <= 5 {
                eprintln!("ENC MISMATCH on {text:?}\n  want {want:?}\n  got  {got:?}");
            }
        }
        if let Some(dec_want) = row["decoded"].as_str() {
            let dec_got = tok.decode(&want);
            if dec_got != dec_want {
                dec_fail += 1;
                if dec_fail <= 5 {
                    eprintln!("DEC MISMATCH: want {dec_want:?} got {dec_got:?}");
                }
            }
        }
    }
    println!("tokenizer parity: {total} rows, {enc_fail} encode mismatches, {dec_fail} decode mismatches");
    if enc_fail > 0 || dec_fail > 0 {
        bail!("tokenizer parity failed");
    }
    Ok(())
}
