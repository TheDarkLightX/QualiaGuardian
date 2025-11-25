# PR Preparation Checklist

## Pre-PR Steps

### 1. Verify All Changes
- [ ] Review all modified files
- [ ] Ensure no unintended changes
- [ ] Check that all fixes are correct

### 2. Run Tests
```bash
# Run all tests
pytest tests/ -v

# Run new property-based tests
pytest tests/core/test_betes_property_based.py -v

# Run numerical stability tests
pytest tests/core/test_betes_numerical_stability.py -v

# Run with coverage
pytest tests/ --cov=guardian.core --cov-report=term-missing
```

### 3. Check Linting
```bash
# Run linter (if available)
# flake8 guardian/core/
# pylint guardian/core/
# mypy guardian/core/  # if type checking enabled
```

### 4. Verify Documentation
- [ ] All new functions have docstrings
- [ ] Complex logic is commented
- [ ] Breaking changes are documented

### 5. Review Breaking Changes
- [ ] Document NaN return for zero weights
- [ ] Check if any callers need updates
- [ ] Add migration notes if needed

## PR Description Template

Use `PR_DESCRIPTION.md` as the base for your PR description. Key sections:

1. **Summary** - Brief overview
2. **Changes Overview** - Detailed changes
3. **Files Changed** - List of modified/added files
4. **Breaking Changes** - Document any breaking changes
5. **Testing** - Test coverage and results
6. **Mathematical Verification** - Properties proven
7. **Performance Impact** - Any performance changes
8. **Code Quality Metrics** - Complexity reduction stats

## Git Commands

### Create Feature Branch (if not already done)
```bash
git checkout -b improve-code-quality-stability
```

### Stage Changes
```bash
# Review what will be committed
git status

# Stage all changes
git add guardian/core/
git add tests/core/
git add *.md
git add .github/

# Or stage specific files
git add guardian/core/betes.py
git add guardian/core/tes.py
# ... etc
```

### Commit with Good Message
```bash
git commit -m "Improve code quality: reduce complexity, fix numerical stability, add property tests

- Reduce cyclomatic complexity by 60-75% in core modules
- Fix numerical stability issues in bE-TES calculations
- Add comprehensive property-based tests using Hypothesis
- Extract duplicate code into shared utilities
- Improve algorithms with log-space arithmetic

Breaking change: Geometric mean returns NaN when all weights are zero
(was 0.0). This is mathematically correct.

Fixes: Numerical underflow in geometric mean
Fixes: Sigmoid overflow for extreme values
Improves: Code maintainability and test coverage"
```

### Push Branch
```bash
git push origin improve-code-quality-stability
```

## PR Review Preparation

### What Reviewers Will Check
1. **Code Quality**
   - Complexity reduction
   - Code organization
   - Naming conventions

2. **Correctness**
   - Mathematical correctness
   - Edge case handling
   - Test coverage

3. **Breaking Changes**
   - NaN handling
   - Backward compatibility
   - Migration path

### Prepare for Questions
- Why log-space arithmetic? (Numerical stability)
- Why NaN for zero weights? (Mathematically correct)
- Why property-based tests? (Better coverage, finds edge cases)
- Performance impact? (Minimal, more accurate)

## Files to Include in PR

### Core Changes
- `guardian/core/betes.py`
- `guardian/core/tes.py`
- `guardian/core/osqi.py`
- `guardian/core/classification.py` (new)
- `guardian/agent/optimizer_agent.py`

### Tests
- `tests/core/test_betes_property_based.py` (new)
- `tests/core/test_betes_numerical_stability.py` (new)

### Documentation
- `DEEP_ANALYSIS_AND_IMPROVEMENTS.md`
- `CRITICAL_FIXES_APPLIED.md`
- `COMPREHENSIVE_ANALYSIS_SUMMARY.md`
- `IMPROVEMENTS_SUMMARY.md`
- `PR_DESCRIPTION.md`

## After PR is Created

1. **Monitor CI/CD**
   - Ensure all tests pass
   - Check coverage reports
   - Verify linting passes

2. **Respond to Reviews**
   - Address comments promptly
   - Update code if needed
   - Update documentation if requested

3. **Update PR Description**
   - Add any new findings
   - Update test results
   - Note any follow-up items

## Follow-up Items (Future PRs)

These can be mentioned in the PR but don't need to be in this one:

- [ ] Add property-based tests for OSQI
- [ ] Add property-based tests for Shapley
- [ ] Implement adaptive Shapley sampling
- [ ] Add input validation layer
- [ ] Consider alternative speed factor formulas

---

**Ready to create PR?** ✅

Use `PR_DESCRIPTION.md` as your PR description body.
