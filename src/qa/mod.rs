pub mod bot;
pub mod fuzzer;
pub mod static_audit;
pub mod visual;

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub const DEFAULT_QA_OUT_DIR: &str = "autotest_report";
pub const DEFAULT_QA_BASELINES_DIR: &str = "qa_baselines";
pub const DEFAULT_QA_FUZZ_SEEDS: usize = 5;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QaCliCommand {
    Suite {
        out_dir: PathBuf,
        baselines_dir: PathBuf,
        fuzz_seeds: usize,
    },
    Fuzz {
        seeds: usize,
        out_dir: PathBuf,
    },
    Playthrough {
        out_dir: PathBuf,
    },
    UpdateBaselines {
        baselines_dir: PathBuf,
        out_dir: PathBuf,
    },
}

/// Parse `--qa-suite`, `--qa-fuzz <N>`, `--qa-playthrough`, and `--qa-update-baselines` flags.
/// Expects full process `argv` (binary name at index 0).
pub fn parse_qa_args(args: &[String]) -> anyhow::Result<Option<QaCliCommand>> {
    let mut mode: Option<&'static str> = None;
    let mut fuzz_seeds = DEFAULT_QA_FUZZ_SEEDS;
    let mut out_dir = PathBuf::from(DEFAULT_QA_OUT_DIR);
    let mut baselines_dir = PathBuf::from(DEFAULT_QA_BASELINES_DIR);

    let mut i = 1usize;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--qa-suite" {
            mode = Some("suite");
            i += 1;
        } else if arg == "--qa-playthrough" {
            mode = Some("playthrough");
            i += 1;
        } else if arg == "--qa-update-baselines" {
            mode = Some("update_baselines");
            i += 1;
        } else if let Some(val) = arg.strip_prefix("--qa-fuzz=") {
            let n = parse_positive_usize(val, "--qa-fuzz")?;
            fuzz_seeds = n;
            if mode.is_none() {
                mode = Some("fuzz");
            }
            i += 1;
        } else if arg == "--qa-fuzz" {
            let val = args
                .get(i + 1)
                .ok_or_else(|| anyhow::anyhow!("Missing integer count after --qa-fuzz"))?;
            let n = parse_positive_usize(val, "--qa-fuzz")?;
            fuzz_seeds = n;
            if mode.is_none() {
                mode = Some("fuzz");
            }
            i += 2;
        } else if let Some(val) = arg.strip_prefix("--qa-out=") {
            if val.trim().is_empty() {
                anyhow::bail!("Empty path passed to --qa-out");
            }
            out_dir = PathBuf::from(val);
            i += 1;
        } else if arg == "--qa-out" {
            let val = args
                .get(i + 1)
                .ok_or_else(|| anyhow::anyhow!("Missing path after --qa-out"))?;
            if val.trim().is_empty() {
                anyhow::bail!("Empty path passed to --qa-out");
            }
            out_dir = PathBuf::from(val);
            i += 2;
        } else if let Some(val) = arg.strip_prefix("--qa-baselines=") {
            if val.trim().is_empty() {
                anyhow::bail!("Empty path passed to --qa-baselines");
            }
            baselines_dir = PathBuf::from(val);
            i += 1;
        } else if arg == "--qa-baselines" {
            let val = args
                .get(i + 1)
                .ok_or_else(|| anyhow::anyhow!("Missing path after --qa-baselines"))?;
            if val.trim().is_empty() {
                anyhow::bail!("Empty path passed to --qa-baselines");
            }
            baselines_dir = PathBuf::from(val);
            i += 2;
        } else {
            i += 1;
        }
    }

    Ok(match mode {
        Some("suite") => Some(QaCliCommand::Suite {
            out_dir,
            baselines_dir,
            fuzz_seeds,
        }),
        Some("fuzz") => Some(QaCliCommand::Fuzz {
            seeds: fuzz_seeds,
            out_dir,
        }),
        Some("playthrough") => Some(QaCliCommand::Playthrough { out_dir }),
        Some("update_baselines") => Some(QaCliCommand::UpdateBaselines {
            baselines_dir,
            out_dir,
        }),
        _ => None,
    })
}

fn parse_positive_usize(raw: &str, flag: &str) -> anyhow::Result<usize> {
    let n: usize = raw
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid integer '{raw}' for {flag}"))?;
    if n == 0 {
        anyhow::bail!("{flag} requires a positive count >= 1");
    }
    Ok(n)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaMasterReport {
    pub passed: bool,
    pub mode: String,
    pub timestamp_unix: u64,
    pub elapsed_secs: f64,
    pub static_audit: Option<static_audit::StaticAuditReport>,
    pub fuzzer: Option<fuzzer::FuzzerReport>,
    pub playthrough: Option<bot::PlaythroughReport>,
    pub visual: Option<visual::VisualRegressionReport>,
    pub failures: Vec<String>,
}

pub fn unix_timestamp_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn write_master_report(out_dir: &Path, report: &QaMasterReport) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(out_dir)?;
    let path = out_dir.join("qa_master_report.json");
    let json = serde_json::to_string_pretty(report)?;
    std::fs::write(&path, json)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_qa_args_handles_all_modes_and_errors() {
        assert!(parse_qa_args(&["cubex-rs".into(), "--smoke-test".into()])
            .unwrap()
            .is_none());

        assert_eq!(
            parse_qa_args(&["cubex-rs".into(), "--qa-suite".into()])
                .unwrap()
                .unwrap(),
            QaCliCommand::Suite {
                out_dir: PathBuf::from(DEFAULT_QA_OUT_DIR),
                baselines_dir: PathBuf::from(DEFAULT_QA_BASELINES_DIR),
                fuzz_seeds: DEFAULT_QA_FUZZ_SEEDS,
            }
        );

        assert_eq!(
            parse_qa_args(&[
                "cubex-rs".into(),
                "--qa-fuzz".into(),
                "12".into(),
                "--qa-out=custom_qa".into()
            ])
            .unwrap()
            .unwrap(),
            QaCliCommand::Fuzz {
                seeds: 12,
                out_dir: PathBuf::from("custom_qa"),
            }
        );

        assert_eq!(
            parse_qa_args(&["cubex-rs".into(), "--qa-playthrough".into()])
                .unwrap()
                .unwrap(),
            QaCliCommand::Playthrough {
                out_dir: PathBuf::from(DEFAULT_QA_OUT_DIR),
            }
        );

        assert_eq!(
            parse_qa_args(&[
                "cubex-rs".into(),
                "--qa-update-baselines".into(),
                "--qa-baselines".into(),
                "goldens".into()
            ])
            .unwrap()
            .unwrap(),
            QaCliCommand::UpdateBaselines {
                baselines_dir: PathBuf::from("goldens"),
                out_dir: PathBuf::from(DEFAULT_QA_OUT_DIR),
            }
        );

        assert!(parse_qa_args(&["cubex-rs".into(), "--qa-fuzz".into(), "0".into()]).is_err());
        assert!(parse_qa_args(&["cubex-rs".into(), "--qa-fuzz".into()]).is_err());
    }
}
