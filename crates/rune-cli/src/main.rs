use anyhow::{bail, Result};
use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use rune::SpannedError as _;

fn compile(source: &str) -> rune::Result<st::Unit> {
    let unit = rune::parse_all::<rune::ast::File>(&source)?;
    Ok(unit.encode()?)
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let mut args = env::args();
    args.next();

    let mut path = None;
    let mut trace = false;
    let mut dump_unit = false;
    let mut dump_vm = false;
    let mut dump_functions = false;
    let mut dump_types = false;
    let mut help = false;

    for arg in args {
        match arg.as_str() {
            "--" => continue,
            "--trace" => {
                trace = true;
            }
            "--dump" => {
                dump_unit = true;
                dump_vm = true;
                dump_functions = true;
                dump_types = true;
            }
            "--dump-unit" => {
                dump_unit = true;
            }
            "--dump-vm" => {
                dump_vm = true;
            }
            "--dump-functions" => {
                dump_functions = true;
            }
            "--dump-types" => {
                dump_types = true;
            }
            "--help" => {
                help = true;
            }
            other if !other.starts_with("-") => {
                path = Some(PathBuf::from(other));
            }
            other => {
                println!("Unrecognized option: {}", other);
                help = true;
            }
        }
    }

    const USAGE: &str = "rune-cli [--trace] <file>";

    if help {
        println!("Usage: {}", USAGE);
        println!();
        println!("  --trace          - Provide detailed tracing for each instruction executed.");
        println!("  --dump           - Dump all forms of diagnostic.");
        println!("  --dump-unit      - Dump diagnostics on the unit generated from the file.");
        println!("  --dump-vm        - Dump diagnostics on VM state. If combined with `--trace`, does so afte each instruction.");
        println!("  --dump-functions - Dump available functions.");
        println!("  --dump-types     - Dump available types.");
        return Ok(());
    }

    let path = match path {
        Some(path) => PathBuf::from(path),
        None => {
            bail!("Invalid usage: {}", USAGE);
        }
    };

    let source = fs::read_to_string(&path)?;

    let unit = match compile(&source) {
        Ok(unit) => unit,
        Err(e) => {
            let span = e.span();
            let thing = &source[span.start..span.end];
            let (line, col) = span.line_col(&source);

            println!(
                "{} at {}:{}:{} {}",
                e,
                path.display(),
                line + 1,
                col + 1,
                thing
            );

            let mut i = 0;
            let mut e = &e as &dyn Error;

            while let Some(err) = e.source() {
                println!("#{}: {}", i, err);
                i += 1;
                e = err;
            }

            return Ok(());
        }
    };

    let mut context = st::Context::with_default_packages()?;
    context.install(st_http::module()?)?;
    context.install(st_json::module()?)?;

    if dump_functions {
        println!("# functions");

        for (i, (hash, f)) in context.iter_functions().enumerate() {
            println!("{:04} = {} ({})", i, f, hash);
        }
    }

    if dump_types {
        println!("# types");

        for (i, (hash, ty)) in context.iter_types().enumerate() {
            println!("{:04} = {} ({})", i, ty, hash);
        }
    }

    if dump_unit {
        use std::io::Write as _;

        println!("# instructions:");

        let mut first_function = true;

        for (n, inst) in unit.iter_instructions().enumerate() {
            let out = std::io::stdout();
            let mut out = out.lock();

            let debug = unit.debug_info_at(n);

            if let Some((hash, function)) = unit.function_at(n) {
                if first_function {
                    first_function = false;
                } else {
                    writeln!(out)?;
                }

                writeln!(out, "fn {} ({}):", function.signature, hash)?;
            }

            if let Some(debug) = debug {
                if let Some(label) = debug.label {
                    writeln!(out, "{}:", label)?;
                }
            }

            write!(out, "  {:04} = {}", n, inst)?;

            if let Some(debug) = debug {
                if let Some(comment) = &debug.comment {
                    write!(out, " // {}", comment)?;
                }
            }

            writeln!(out)?;
        }

        println!("# imports:");

        for (hash, f) in unit.iter_imports() {
            println!("{} = {}", hash, f);
        }

        println!("# functions:");

        for (hash, f) in unit.iter_functions() {
            println!("{} = {} (at: {})", hash, f.signature, f.offset);
        }

        println!("# strings:");

        for (hash, string) in unit.iter_static_strings() {
            println!("{} = {:?}", hash, string);
        }

        println!("---");
    }

    let mut vm = st::Vm::new();

    let task: st::Task<st::Value> = vm.call_function(&context, &unit, &["main"], ())?;
    let last = std::time::Instant::now();

    let result = if trace {
        do_trace(task, dump_vm).await
    } else {
        task.run_to_completion().await.map_err(anyhow::Error::from)
    };

    let result = match result {
        Ok(result) => result,
        Err(e) => {
            println!("#0: {}", e);
            let mut e = &*e as &dyn Error;
            let mut i = 1;

            while let Some(err) = e.source() {
                println!("#{}: {}", i, err);
                i += 1;
                e = err;
            }

            return Ok(());
        }
    };

    let duration = std::time::Instant::now().duration_since(last);
    println!("== {:?} ({:?})", result, duration);

    if dump_vm {
        println!("# stack dump after completion");

        for (n, (slot, value)) in vm.iter_stack_debug().enumerate() {
            println!("{} = {:?} ({:?})", n, slot, value);
        }

        println!("---");
    }

    Ok(())
}

/// Perform a detailed trace of the program.
async fn do_trace<T>(mut task: st::Task<'_, T>, dump_vm: bool) -> Result<T>
where
    T: st::FromValue,
{
    loop {
        use std::io::Write as _;

        let out = std::io::stdout();
        let mut out = out.lock();

        let debug = task.unit.debug_info_at(task.vm.ip());

        if let Some((hash, function)) = task.unit.function_at(task.vm.ip()) {
            writeln!(out, "fn {} ({}):", function.signature, hash)?;
        }

        if let Some(debug) = debug {
            if let Some(label) = debug.label {
                writeln!(out, "{}:", label)?;
            }
        }

        if let Some(inst) = task.unit.instruction_at(task.vm.ip()) {
            write!(out, "  {:04} = {}", task.vm.ip(), inst)?;
        } else {
            write!(out, "  {:04} = *out of bounds*", task.vm.ip(),)?;
        }

        if let Some(debug) = debug {
            if let Some(comment) = &debug.comment {
                write!(out, " // {}", comment)?;
            }
        }

        writeln!(out)?;

        let result = task.step().await?;

        if dump_vm {
            println!("# stack dump");

            for (n, (slot, value)) in task.vm.iter_stack_debug().enumerate() {
                println!("{} = {:?} ({:?})", n, slot, value);
            }

            println!("---");
        }

        if let Some(result) = result {
            break Ok(result);
        }
    }
}
