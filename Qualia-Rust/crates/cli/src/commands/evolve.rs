//! Evolve command implementation

use std::path::PathBuf;
use std::time::Instant;
use anyhow::Result;
use tracing::{info, debug};
use serde::{Serialize, Deserialize};
use indicatif::{ProgressBar, ProgressStyle};

use qualia_evolution::{
    EvolutionConfig, NSGA2,
    fitness::{TestSuite, TestCase, MutationScoreFitness, SpeedFitness, CompositeFitness},
    operators::{TestMutationOperator, TestCrossoverOperator},
};
use qualia_sensors::{
    SensorContext, SensorPlugin,
    mutation::MutationSensor,
    speed::SpeedSensor,
};
use qualia_core::{DbPool, init_database, database::{Repository, NewEvolutionResult}};

use crate::OutputFormat;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EvolutionProgress {
    generation: usize,
    best_fitness: Vec<f64>,
    pareto_front_size: usize,
    elapsed_secs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EvolutionSummary {
    total_generations: usize,
    final_population_size: usize,
    pareto_front_size: usize,
    best_fitness_values: Vec<f64>,
    improvement_percentage: f64,
    elapsed_time_secs: f64,
    optimized_test_suites: Vec<OptimizedSuite>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OptimizedSuite {
    name: String,
    mutation_score: f64,
    execution_time_ms: f64,
    test_count: usize,
}

/// Run the evolve command
pub async fn run(
    src: PathBuf,
    tests: PathBuf,
    pop_size: usize,
    generations: usize,
    mutation_rate: f64,
    format: OutputFormat,
) -> Result<()> {
    let start_time = Instant::now();
    
    info!("Starting evolutionary test optimization");
    info!("Source: {}", src.display());
    info!("Tests: {}", tests.display());
    info!("Population size: {}", pop_size);
    info!("Generations: {}", generations);
    info!("Mutation rate: {}", mutation_rate);
    
    // Validate paths
    if !src.exists() {
        anyhow::bail!("Source directory does not exist: {}", src.display());
    }
    if !tests.exists() {
        anyhow::bail!("Test directory does not exist: {}", tests.display());
    }
    
    // Create progress bar
    let pb = ProgressBar::new(generations as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
            .progress_chars("=>-")
    );
    
    // Initialize sensors for fitness evaluation
    let context = SensorContext {
        project_root: src.clone(),
        test_dir: Some(tests.clone()),
        cache_dir: None,
        config: Default::default(),
    };
    
    let mutation_sensor = MutationSensor::new();
    let speed_sensor = SpeedSensor::new();
    
    // Get baseline metrics
    pb.set_message("Collecting baseline metrics...");
    let baseline_mutation = mutation_sensor.measure(&context).await?;
    let baseline_speed = speed_sensor.measure(&context).await?;
    
    info!("Baseline mutation score: {:.2}%", baseline_mutation * 100.0);
    info!("Baseline test execution time: {:.2}s", baseline_speed);
    
    // Load initial test suites
    let initial_suites = load_test_suites(&tests)?;
    info!("Loaded {} test suites", initial_suites.len());
    
    // Configure evolution
    let config = EvolutionConfig {
        population_size: pop_size,
        generations,
        mutation_rate,
        crossover_rate: 0.8,
        tournament_size: 3,
        elitism_size: 2,
        parallel: true,
    };
    
    // Create fitness functions
    let mutation_fitness = MutationScoreFitness::new(mutation_sensor);
    let speed_fitness = SpeedFitness::new(speed_sensor);
    let composite_fitness = CompositeFitness::new(vec![
        Box::new(mutation_fitness),
        Box::new(speed_fitness),
    ]);
    
    // Create operators
    let mutation_op = TestMutationOperator::new(mutation_rate);
    let crossover_op = TestCrossoverOperator::new();
    
    // Initialize NSGA-II
    let mut nsga2 = NSGA2::new(
        config.clone(),
        Box::new(composite_fitness),
        Box::new(mutation_op),
        Box::new(crossover_op),
    );
    
    // Track progress
    let mut progress_updates = Vec::new();
    
    // Evolution callback
    let evolution_callback = |gen: usize, pop: &[TestSuite]| {
        pb.set_position(gen as u64);
        pb.set_message(format!("Generation {}/{}", gen, generations));
        
        // Calculate best fitness values
        let best_fitness = pop.iter()
            .take(1)
            .flat_map(|suite| suite.fitness_values.clone())
            .collect::<Vec<_>>();
        
        let pareto_front_size = pop.iter()
            .filter(|s| s.rank == Some(0))
            .count();
        
        let progress = EvolutionProgress {
            generation: gen,
            best_fitness: best_fitness.clone(),
            pareto_front_size,
            elapsed_secs: start_time.elapsed().as_secs_f64(),
        };
        
        progress_updates.push(progress.clone());
        
        debug!("Generation {} - Best fitness: {:?}, Pareto front size: {}", 
            gen, best_fitness, pareto_front_size);
    };
    
    // Run evolution
    pb.set_message("Running evolution...");
    let final_population = nsga2.evolve_with_callback(initial_suites, evolution_callback).await?;
    
    pb.finish_with_message("Evolution complete!");
    
    // Get final metrics
    let final_mutation = mutation_sensor.measure(&context).await?;
    let final_speed = speed_sensor.measure(&context).await?;
    
    // Calculate improvement
    let mutation_improvement = (final_mutation - baseline_mutation) / baseline_mutation * 100.0;
    let speed_improvement = (baseline_speed - final_speed) / baseline_speed * 100.0;
    
    info!("Final mutation score: {:.2}% ({}% improvement)", 
        final_mutation * 100.0, mutation_improvement);
    info!("Final test execution time: {:.2}s ({}% improvement)", 
        final_speed, speed_improvement);
    
    // Get optimized suites from Pareto front
    let pareto_front: Vec<_> = final_population.iter()
        .filter(|s| s.rank == Some(0))
        .collect();
    
    let optimized_suites = pareto_front.iter()
        .take(5) // Top 5 suites
        .map(|suite| OptimizedSuite {
            name: format!("Suite_{}", suite.id),
            mutation_score: suite.fitness_values.get(0).copied().unwrap_or(0.0),
            execution_time_ms: suite.fitness_values.get(1).copied().unwrap_or(0.0) * 1000.0,
            test_count: suite.test_cases.len(),
        })
        .collect();
    
    // Create summary
    let summary = EvolutionSummary {
        total_generations: generations,
        final_population_size: final_population.len(),
        pareto_front_size: pareto_front.len(),
        best_fitness_values: final_population[0].fitness_values.clone(),
        improvement_percentage: mutation_improvement.max(0.0),
        elapsed_time_secs: start_time.elapsed().as_secs_f64(),
        optimized_test_suites: optimized_suites.clone(),
    };
    
    // Save to database if available
    if let Ok(db_path) = std::env::var("GUARDIAN_DB_PATH") {
        if let Ok(pool) = init_database(&db_path).await {
            save_evolution_result(&pool, &src, &summary).await?;
        }
    }
    
    // Output results based on format
    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        OutputFormat::Text => {
            print_evolution_summary(&summary, &progress_updates);
        }
        OutputFormat::Markdown => {
            print_evolution_markdown(&summary, &progress_updates);
        }
    }
    
    // Save optimized test suites to disk if requested
    if std::env::var("GUARDIAN_SAVE_OPTIMIZED_TESTS").is_ok() {
        save_optimized_suites(&tests, &pareto_front).await?;
        info!("Optimized test suites saved to: {}/optimized", tests.display());
    }
    
    Ok(())
}

fn load_test_suites(test_dir: &PathBuf) -> Result<Vec<TestSuite>> {
    use walkdir::WalkDir;
    
    let mut suites = Vec::new();
    let mut suite_id = 0;
    
    for entry in WalkDir::new(test_dir)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        
        // Look for test files
        if path.is_file() && is_test_file(path) {
            let content = std::fs::read_to_string(path)?;
            let test_cases = extract_test_cases(&content, path)?;
            
            if !test_cases.is_empty() {
                suites.push(TestSuite {
                    id: suite_id,
                    test_cases,
                    fitness_values: Vec::new(),
                    rank: None,
                    crowding_distance: 0.0,
                });
                suite_id += 1;
            }
        }
    }
    
    Ok(suites)
}

fn is_test_file(path: &std::path::Path) -> bool {
    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
        // Rust test files
        if name.ends_with(".rs") && (name.starts_with("test_") || name.contains("_test")) {
            return true;
        }
        // Python test files
        if name.ends_with(".py") && (name.starts_with("test_") || name.contains("_test")) {
            return true;
        }
        // JavaScript/TypeScript test files
        if (name.ends_with(".test.js") || name.ends_with(".test.ts") || 
            name.ends_with(".spec.js") || name.ends_with(".spec.ts")) {
            return true;
        }
    }
    false
}

fn extract_test_cases(content: &str, path: &std::path::Path) -> Result<Vec<TestCase>> {
    let mut test_cases = Vec::new();
    
    // Simple test extraction based on file extension
    if path.extension().and_then(|e| e.to_str()) == Some("rs") {
        // Rust tests
        for (idx, line) in content.lines().enumerate() {
            if line.trim().starts_with("#[test]") || line.contains("fn test_") {
                test_cases.push(TestCase {
                    name: format!("{}:{}", path.display(), idx + 1),
                    content: line.to_string(),
                    assertions: count_assertions(line),
                });
            }
        }
    } else if path.extension().and_then(|e| e.to_str()) == Some("py") {
        // Python tests
        for (idx, line) in content.lines().enumerate() {
            if line.trim().starts_with("def test_") || line.trim().starts_with("async def test_") {
                test_cases.push(TestCase {
                    name: format!("{}:{}", path.display(), idx + 1),
                    content: line.to_string(),
                    assertions: count_assertions(line),
                });
            }
        }
    }
    
    Ok(test_cases)
}

fn count_assertions(content: &str) -> usize {
    let assertion_patterns = [
        "assert", "expect", "should", "must", "verify", "check",
        "assert_eq", "assert_ne", "assert!", "assertEquals", "assertTrue"
    ];
    
    assertion_patterns.iter()
        .map(|pattern| content.matches(pattern).count())
        .sum()
}

async fn save_evolution_result(pool: &DbPool, project_path: &PathBuf, summary: &EvolutionSummary) -> Result<()> {
    let repo = Repository::new(pool);
    
    let best_fitness = serde_json::json!({
        "mutation_score": summary.best_fitness_values.get(0),
        "execution_time": summary.best_fitness_values.get(1),
    });
    
    let optimized_suites = serde_json::json!(summary.optimized_test_suites);
    
    let improvement_metrics = serde_json::json!({
        "improvement_percentage": summary.improvement_percentage,
        "elapsed_time_secs": summary.elapsed_time_secs,
    });
    
    let result = NewEvolutionResult {
        project_path: project_path.to_string_lossy().to_string(),
        generations: summary.total_generations as i32,
        population_size: summary.final_population_size as i32,
        pareto_front_size: summary.pareto_front_size as i32,
        best_fitness,
        optimized_suites,
        improvement_metrics,
    };
    
    repo.evolution.create(result).await?;
    Ok(())
}

async fn save_optimized_suites(test_dir: &PathBuf, pareto_front: &[&TestSuite]) -> Result<()> {
    let optimized_dir = test_dir.join("optimized");
    std::fs::create_dir_all(&optimized_dir)?;
    
    for (idx, suite) in pareto_front.iter().enumerate() {
        let suite_file = optimized_dir.join(format!("optimized_suite_{}.txt", idx));
        let mut content = String::new();
        
        content.push_str(&format!("# Optimized Test Suite {}\n", idx));
        content.push_str(&format!("# Fitness values: {:?}\n", suite.fitness_values));
        content.push_str(&format!("# Test count: {}\n\n", suite.test_cases.len()));
        
        for test in &suite.test_cases {
            content.push_str(&format!("## {}\n", test.name));
            content.push_str(&format!("{}\n\n", test.content));
        }
        
        std::fs::write(suite_file, content)?;
    }
    
    Ok(())
}

fn print_evolution_summary(summary: &EvolutionSummary, progress: &[EvolutionProgress]) {
    println!("\n🧬 Evolution Complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Generations:          {}", summary.total_generations);
    println!("Final population:     {}", summary.final_population_size);
    println!("Pareto front size:    {}", summary.pareto_front_size);
    println!("Improvement:          {:.1}%", summary.improvement_percentage);
    println!("Total time:           {:.1}s", summary.elapsed_time_secs);
    
    println!("\n📊 Top Optimized Suites:");
    println!("┌─────────────────┬───────────────┬──────────────┬────────────┐");
    println!("│ Suite           │ Mutation Score│ Exec Time(ms)│ Test Count │");
    println!("├─────────────────┼───────────────┼──────────────┼────────────┤");
    
    for suite in &summary.optimized_test_suites {
        println!("│ {:<15} │ {:>13.1}%│ {:>12.1} │ {:>10} │",
            suite.name,
            suite.mutation_score * 100.0,
            suite.execution_time_ms,
            suite.test_count
        );
    }
    println!("└─────────────────┴───────────────┴──────────────┴────────────┘");
    
    // Show progress sparkline
    if !progress.is_empty() {
        println!("\n📈 Fitness Progress:");
        let max_fitness = progress.iter()
            .flat_map(|p| &p.best_fitness)
            .fold(0.0f64, |a, &b| a.max(b));
        
        let sparkline: String = progress.iter()
            .filter_map(|p| p.best_fitness.first())
            .map(|&f| {
                let normalized = (f / max_fitness * 7.0) as usize;
                ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'][normalized.min(7)]
            })
            .collect();
        
        println!("   {}", sparkline);
    }
}

fn print_evolution_markdown(summary: &EvolutionSummary, progress: &[EvolutionProgress]) {
    println!("# Evolution Results\n");
    println!("## Summary\n");
    println!("| Metric | Value |");
    println!("|--------|-------|");
    println!("| Generations | {} |", summary.total_generations);
    println!("| Final Population | {} |", summary.final_population_size);
    println!("| Pareto Front Size | {} |", summary.pareto_front_size);
    println!("| Improvement | {:.1}% |", summary.improvement_percentage);
    println!("| Total Time | {:.1}s |", summary.elapsed_time_secs);
    
    println!("\n## Optimized Test Suites\n");
    println!("| Suite | Mutation Score | Execution Time | Test Count |");
    println!("|-------|----------------|----------------|------------|");
    
    for suite in &summary.optimized_test_suites {
        println!("| {} | {:.1}% | {:.1}ms | {} |",
            suite.name,
            suite.mutation_score * 100.0,
            suite.execution_time_ms,
            suite.test_count
        );
    }
    
    if !progress.is_empty() {
        println!("\n## Evolution Progress\n");
        println!("```");
        for (i, p) in progress.iter().enumerate() {
            if i % 10 == 0 {  // Show every 10th generation
                println!("Gen {:3}: Fitness {:?}, Pareto size: {}",
                    p.generation, p.best_fitness, p.pareto_front_size);
            }
        }
        println!("```");
    }
}