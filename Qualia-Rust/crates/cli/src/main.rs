//! Guardian CLI - Quality optimization for your codebase

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use anyhow::Result;
use tracing::{info, error};
use tracing_subscriber::EnvFilter;

mod commands;
mod output;
mod database;

use commands::{analyze, evolve, gamify, history, self_improve};
pub use output::OutputFormat;

/// Guardian - Autonomous code quality optimization system
#[derive(Parser)]
#[command(author = "DarkLightX/Dana Edwards", version, about, long_about = None)]
struct Cli {
    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
    
    /// Output format
    #[arg(short, long, global = true, default_value = "text")]
    format: OutputFormat,
    
    #[command(subcommand)]
    command: Commands,
}


#[derive(Subcommand)]
enum Commands {
    /// Analyze code quality
    Analyze {
        /// Path to the project to analyze
        path: PathBuf,
        
        /// Run quality analysis
        #[arg(long)]
        run_quality: bool,
        
        /// Quality scoring mode
        #[arg(long, default_value = "betes_v3.1")]
        quality_mode: String,
        
        /// Risk class for threshold evaluation
        #[arg(long)]
        risk_class: Option<String>,
        
        /// Run specific sensors
        #[arg(long, value_delimiter = ',')]
        sensors: Option<Vec<String>>,
    },
    
    /// Evolve tests using evolutionary algorithms
    #[command(name = "ec-evolve")]
    Evolve {
        /// Source directory
        src: PathBuf,
        
        /// Test directory
        tests: PathBuf,
        
        /// Population size
        #[arg(long, default_value = "20")]
        pop_size: usize,
        
        /// Number of generations
        #[arg(long, default_value = "10")]
        generations: usize,
        
        /// Mutation rate
        #[arg(long, default_value = "0.1")]
        mutation_rate: f64,
    },
    
    /// Gamification features
    Gamify {
        #[command(subcommand)]
        command: GamifyCommands,
    },
    
    /// View analysis history
    History {
        /// Number of recent runs to show
        #[arg(short, long, default_value = "10")]
        limit: usize,
        
        /// Filter by project path
        #[arg(long)]
        project: Option<String>,
        
        /// Show detailed metrics
        #[arg(long)]
        detailed: bool,
    },
    
    /// Run self-improvement analysis
    SelfImprove {
        /// Dry run mode (no actual changes)
        #[arg(long)]
        dry_run: bool,
        
        /// Maximum improvements to apply
        #[arg(long, default_value = "5")]
        max_improvements: usize,
    },
}

#[derive(Subcommand)]
enum GamifyCommands {
    /// Show gamification status
    Status,
    
    /// List available badges
    Badges {
        /// Show only earned badges
        #[arg(long)]
        earned: bool,
    },
    
    /// Show current quests
    Quests {
        /// Show completed quests
        #[arg(long)]
        completed: bool,
    },
    
    /// Show top performers (Shapley crown)
    Crown {
        /// Number of top items to show
        #[arg(long, default_value = "10")]
        top: usize,
        
        /// Test directory
        #[arg(long)]
        test_dir: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse CLI arguments
    let cli = Cli::parse();
    
    // Initialize logging
    let filter = if cli.verbose {
        EnvFilter::from_default_env()
            .add_directive("guardian=debug".parse()?)
            .add_directive("qualia=debug".parse()?)
    } else {
        EnvFilter::from_default_env()
            .add_directive("guardian=info".parse()?)
            .add_directive("qualia=info".parse()?)
    };
    
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .init();
    
    // Execute command
    match cli.command {
        Commands::Analyze { 
            path, 
            run_quality, 
            quality_mode, 
            risk_class,
            sensors,
        } => {
            info!("Analyzing project: {}", path.display());
            analyze::run(
                path,
                run_quality,
                &quality_mode,
                risk_class.as_deref(),
                sensors,
                cli.format,
            ).await?;
        }
        
        Commands::Evolve { 
            src, 
            tests, 
            pop_size, 
            generations,
            mutation_rate,
        } => {
            info!("Evolving tests: {} -> {}", src.display(), tests.display());
            evolve::run(
                src,
                tests,
                pop_size,
                generations,
                mutation_rate,
                cli.format,
            ).await?;
        }
        
        Commands::Gamify { command } => {
            match command {
                GamifyCommands::Status => {
                    gamify::show_status(cli.format).await?;
                }
                GamifyCommands::Badges { earned } => {
                    gamify::show_badges(earned, cli.format).await?;
                }
                GamifyCommands::Quests { completed } => {
                    gamify::show_quests(completed, cli.format).await?;
                }
                GamifyCommands::Crown { top, test_dir } => {
                    gamify::show_crown(top, test_dir, cli.format).await?;
                }
            }
        }
        
        Commands::History { limit, project, detailed } => {
            history::show(limit, project.as_deref(), detailed, cli.format).await?;
        }
        
        Commands::SelfImprove { dry_run, max_improvements } => {
            self_improve::run(dry_run, max_improvements, cli.format).await?;
        }
    }
    
    Ok(())
}