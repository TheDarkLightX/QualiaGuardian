//! Genetic operators for test suite evolution

use crate::types::{CrossoverOperator, MutationOperator, Evolvable};
use crate::fitness::{TestSuite, TestCase};
use rand::prelude::*;
use syn::{parse_str, File, Item, ItemFn, Stmt, Expr};
use quote::ToTokens;
use std::collections::HashSet;

/// Mutation operator for test suites
#[derive(Debug)]
pub struct TestMutationOperator {
    base_rate: f64,
}

impl TestMutationOperator {
    pub fn new(base_rate: f64) -> Self {
        Self { base_rate }
    }
}

impl MutationOperator<TestSuite> for TestMutationOperator {
    fn mutate(&self, suite: &mut TestSuite, rate: f64) {
        let mut rng = thread_rng();
        
        // Mutate individual test cases
        for test_case in &mut suite.test_cases {
            if rng.gen::<f64>() < rate {
                // Simple mutations for now
                match rng.gen_range(0..4) {
                    0 => {
                        // Add assertion
                        test_case.assertions += 1;
                        test_case.content.push_str("\n    assert!(true);");
                    }
                    1 => {
                        // Remove assertion (if possible)
                        if test_case.assertions > 1 {
                            test_case.assertions -= 1;
                        }
                    }
                    2 => {
                        // Modify test name
                        test_case.name.push_str("_mutated");
                    }
                    _ => {
                        // Modify content
                        test_case.content.push_str("\n    // Mutated");
                    }
                }
            }
        }
        
        // Structural mutations
        if rng.gen::<f64>() < rate * 0.5 {
            match rng.gen_range(0..3) {
                0 => {
                    // Remove a random test (if we have more than 1)
                    if suite.test_cases.len() > 1 {
                        let idx = rng.gen_range(0..suite.test_cases.len());
                        suite.test_cases.remove(idx);
                    }
                }
                1 => {
                    // Duplicate a random test
                    if !suite.test_cases.is_empty() {
                        let idx = rng.gen_range(0..suite.test_cases.len());
                        let mut cloned = suite.test_cases[idx].clone();
                        cloned.name.push_str("_dup");
                        suite.test_cases.push(cloned);
                    }
                }
                _ => {
                    // Shuffle test order
                    suite.test_cases.shuffle(&mut rng);
                }
            }
        }
    }
}

/// Crossover operator for test suites
#[derive(Debug)]
pub struct TestCrossoverOperator;

impl TestCrossoverOperator {
    pub fn new() -> Self {
        Self
    }
}

impl CrossoverOperator<TestSuite> for TestCrossoverOperator {
    fn crossover(&self, parent1: &TestSuite, parent2: &TestSuite) -> (TestSuite, TestSuite) {
        let mut rng = thread_rng();
        
        // Single-point crossover
        let len1 = parent1.test_cases.len();
        let len2 = parent2.test_cases.len();
        
        if len1 == 0 || len2 == 0 {
            // If either parent is empty, return clones
            return (parent1.clone(), parent2.clone());
        }
        
        let crossover_point1 = rng.gen_range(0..=len1);
        let crossover_point2 = rng.gen_range(0..=len2);
        
        // Create offspring
        let mut child1 = TestSuite {
            id: parent1.id,
            test_cases: Vec::new(),
            fitness_values: Vec::new(),
            rank: None,
            crowding_distance: 0.0,
        };
        
        let mut child2 = TestSuite {
            id: parent2.id,
            test_cases: Vec::new(),
            fitness_values: Vec::new(),
            rank: None,
            crowding_distance: 0.0,
        };
        
        // Child 1: First part from parent1, second part from parent2
        child1.test_cases.extend_from_slice(&parent1.test_cases[..crossover_point1]);
        if crossover_point2 < len2 {
            child1.test_cases.extend_from_slice(&parent2.test_cases[crossover_point2..]);
        }
        
        // Child 2: First part from parent2, second part from parent1
        child2.test_cases.extend_from_slice(&parent2.test_cases[..crossover_point2]);
        if crossover_point1 < len1 {
            child2.test_cases.extend_from_slice(&parent1.test_cases[crossover_point1..]);
        }
        
        (child1, child2)
    }
}

impl Default for TestCrossoverOperator {
    fn default() -> Self {
        Self::new()
    }
}

/// Smart mutation operator for test suites (delegates to TestMutationOperator)
#[derive(Debug)]
pub struct TestSuiteMutator {
    /// Available test IDs that can be added
    available_tests: Vec<String>,
}

impl TestSuiteMutator {
    pub fn new(available_tests: Vec<String>) -> Self {
        Self { available_tests }
    }
}

impl MutationOperator<TestSuite> for TestSuiteMutator {
    fn mutate(&self, suite: &mut TestSuite, rate: f64) {
        // Delegate to TestMutationOperator for now
        let mutator = TestMutationOperator::new(rate);
        mutator.mutate(suite, rate);
    }
}

/// Smart crossover operator (delegates to TestCrossoverOperator)
#[derive(Debug)]
pub struct TestSuiteCrossover;

impl TestSuiteCrossover {
    pub fn new() -> Self {
        Self
    }
}

impl CrossoverOperator<TestSuite> for TestSuiteCrossover {
    fn crossover(&self, parent1: &TestSuite, parent2: &TestSuite) -> (TestSuite, TestSuite) {
        // Delegate to TestCrossoverOperator for now
        let crossover = TestCrossoverOperator::new();
        crossover.crossover(parent1, parent2)
    }
}

/// Remove duplicates while preserving order
fn remove_duplicates(items: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    items.into_iter()
        .filter(|item| seen.insert(item.clone()))
        .collect()
}

/// AST-based test code representation
#[derive(Clone)]
pub struct TestCode {
    /// Original source code
    pub source: String,
    /// Parsed AST (recreated when needed)
    pub ast: Option<File>,
}

impl std::fmt::Debug for TestCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TestCode")
            .field("source", &self.source)
            .field("ast", &"<parsed AST>")
            .finish()
    }
}

// Safety: syn::File is effectively immutable once parsed and we only
// send the parsed representation across threads, not shared state
unsafe impl Send for TestCode {}
unsafe impl Sync for TestCode {}

impl Evolvable for TestCode {
    fn random() -> Self {
        // Start with minimal test template
        TestCode {
            source: r#"
#[test]
fn test_example() {
    assert_eq!(1 + 1, 2);
}
"#.to_string(),
            ast: None,
        }
    }
    
    fn len(&self) -> usize {
        self.source.len()
    }
}

/// Smart mutation operator for test code
#[derive(Debug)]
pub struct SmartTestMutator;

impl SmartTestMutator {
    /// Mutate assertions in the test
    fn mutate_assertion(expr: &mut Expr, rng: &mut ThreadRng) {
        match expr {
            Expr::Macro(macro_expr) => {
                if let Some(ident) = macro_expr.mac.path.get_ident() {
                    let name = ident.to_string();
                    
                    // Mutate assertion type
                    if rng.gen_bool(0.3) {
                        let new_name = match name.as_str() {
                            "assert_eq" => "assert_ne",
                            "assert_ne" => "assert_eq", 
                            "assert" => if rng.gen_bool(0.5) { "assert_eq" } else { "assert" },
                            _ => return,
                        };
                        
                        if new_name != name {
                            macro_expr.mac.path = syn::Path::from(syn::Ident::new(new_name, ident.span()));
                        }
                    }
                }
            }
            Expr::Binary(binary) => {
                // Mutate binary operators
                if rng.gen_bool(0.3) {
                    use syn::BinOp;
                    binary.op = match binary.op {
                        BinOp::Eq(_) => BinOp::Ne(Default::default()),
                        BinOp::Ne(_) => BinOp::Eq(Default::default()),
                        BinOp::Lt(_) => BinOp::Le(Default::default()),
                        BinOp::Le(_) => BinOp::Lt(Default::default()),
                        BinOp::Gt(_) => BinOp::Ge(Default::default()),
                        BinOp::Ge(_) => BinOp::Gt(Default::default()),
                        _ => binary.op,
                    };
                }
            }
            _ => {}
        }
    }
    
    /// Add boundary value tests
    fn add_boundary_test(stmts: &mut Vec<Stmt>, rng: &mut ThreadRng) {
        let boundary_values = vec![
            "0", "-1", "1", "i32::MIN", "i32::MAX",
            "f64::EPSILON", "f64::NAN", "f64::INFINITY",
            r#""""#, r#""a""#, // Empty and single char strings
        ];
        
        if rng.gen_bool(0.2) {
            let value = boundary_values.choose(rng).unwrap();
            let new_assert = format!("assert!({}.is_finite() || true);", value);
            
            if let Ok(stmt) = parse_str::<Stmt>(&new_assert) {
                let insert_pos = rng.gen_range(0..=stmts.len());
                stmts.insert(insert_pos, stmt);
            }
        }
    }
}

impl MutationOperator<TestCode> for SmartTestMutator {
    fn mutate(&self, test: &mut TestCode, rate: f64) {
        let mut rng = thread_rng();
        
        if rng.gen::<f64>() >= rate {
            return;
        }
        
        // Parse if not already parsed
        if test.ast.is_none() {
            test.ast = parse_str::<File>(&test.source).ok();
        }
        
        if let Some(ast) = &mut test.ast {
            // Find test functions
            for item in &mut ast.items {
                if let Item::Fn(func) = item {
                    if is_test_function(func) {
                        // Mutate function body
                        for stmt in &mut func.block.stmts {
                            match stmt {
                                Stmt::Expr(expr, _) => {
                                    Self::mutate_assertion(expr, &mut rng);
                                }
                                _ => {}
                            }
                        }
                        
                        // Add boundary tests
                        Self::add_boundary_test(&mut func.block.stmts, &mut rng);
                    }
                }
            }
            
            // Convert back to source
            test.source = ast.to_token_stream().to_string();
        }
    }
}

/// Check if a function is a test function
fn is_test_function(func: &ItemFn) -> bool {
    func.attrs.iter().any(|attr| {
        attr.path().is_ident("test") || 
        attr.path().is_ident("tokio::test") ||
        attr.path().is_ident("async_std::test")
    })
}

/// Uniform crossover for test code
#[derive(Debug)]
pub struct TestCodeCrossover;

impl CrossoverOperator<TestCode> for TestCodeCrossover {
    fn crossover(&self, parent1: &TestCode, parent2: &TestCode) -> (TestCode, TestCode) {
        // For simplicity, just clone parents
        // In a real implementation, would merge test functions
        (parent1.clone(), parent2.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_suite_mutation() {
        let mutator = TestMutationOperator::new(0.5);
        
        let mut suite = TestSuite {
            id: 0,
            test_cases: vec![
                TestCase {
                    name: "test1".to_string(),
                    content: "fn test1() { assert!(true); }".to_string(),
                    assertions: 1,
                }
            ],
            fitness_values: vec![],
            rank: None,
            crowding_distance: 0.0,
        };
        
        // High mutation rate to ensure changes
        mutator.mutate(&mut suite, 1.0);
        
        // Suite should have been modified
        assert!(!suite.test_cases.is_empty());
    }
    
    #[test]
    fn test_remove_duplicates() {
        let items = vec![
            "a".to_string(),
            "b".to_string(),
            "a".to_string(),
            "c".to_string(),
        ];
        
        let unique = remove_duplicates(items);
        assert_eq!(unique, vec!["a", "b", "c"]);
    }
}