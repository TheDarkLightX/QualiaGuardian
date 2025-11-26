# Simplification Recommendations

## Current State: Over-Engineered?

### Assessment

**Current System:**
- 13 modules
- 7 aggregation methods
- 6 trust models
- Multiple uncertainty quantification methods
- Temporal analysis, sensitivity analysis, etc.

**Verdict:** 
- ✅ **Well-engineered for research** - All features serve distinct purposes
- ⚠️ **Over-engineered for production** - Most users need simplicity
- ✅ **Good modularity** - Can use only what's needed

## Simplification Strategy

### 1. Primary Metric: CIRS

**Recommendation:** Make CIRS the default/primary metric

**Why:**
- Single best predictor of quality issues
- Simple (one number)
- Actionable (tells you what to fix)
- Validated by research

**Implementation:**
```python
from guardian.core.cirs import CIRSCalculator

calc = CIRSCalculator()
result = calc.calculate(
    change_frequency=5.0,
    complexity=15.0,
    mutation_score=0.6,
    coupling=8.0
)

print(f"Risk Score: {result.cirs_score:.2f}")
print("Insights:", result.insights)
```

### 2. Simplified API

**Current:** Multiple calculators, many options
**Proposed:** Single unified interface

```python
# Simple (default)
score = calculate_quality_score(project_path)

# Advanced (optional)
score = calculate_quality_score(
    project_path,
    use_uncertainty=True,
    use_temporal=True,
    aggregation_method='harmonic'
)
```

### 3. Progressive Disclosure

**Principle:** Simple by default, advanced on demand

**Levels:**
1. **Basic**: CIRS score only
2. **Standard**: CIRS + components + insights
3. **Advanced**: CIRS + uncertainty + temporal + sensitivity
4. **Research**: All features

### 4. Default Settings

**Recommendation:** Sensible defaults that work for 80% of users

- Aggregation: Geometric mean (current default)
- Trust model: Exponential (good balance)
- Uncertainty: Enabled (but hidden unless requested)
- Advanced features: Disabled by default

### 5. Module Organization

**Current:** 13 separate modules
**Proposed:** Organized into tiers

```
guardian/core/
  quality.py          # Main entry point (CIRS + bE-TES)
  uncertainty.py      # Uncertainty quantification
  aggregation.py      # Aggregation methods
  advanced/           # Research features
    temporal.py
    sensitivity.py
    information.py
```

## Refined Architecture

### Core (Always Loaded)
1. `cirs.py` - Change Impact Risk Score
2. `betes.py` - Test effectiveness (complementary)
3. `quality.py` - Unified interface

### Standard (Optional)
4. `uncertainty.py` - Uncertainty quantification
5. `aggregation.py` - Alternative aggregation
6. `trust_models.py` - Non-linear trust

### Advanced (Research)
7. `temporal.py` - Temporal analysis
8. `sensitivity.py` - Sensitivity analysis
9. `information.py` - Information theory
10. `monte_carlo.py` - Monte Carlo
11. `robust.py` - Robust statistics
12. `model_selection.py` - Model selection

## Migration Path

### Phase 1: Add CIRS (No Breaking Changes)
- Add CIRS calculator
- Keep all existing code
- Make CIRS available as option

### Phase 2: Simplify API (Backward Compatible)
- Add unified `calculate_quality_score()` function
- Keep existing calculators
- Default to CIRS

### Phase 3: Reorganize (Optional)
- Move advanced features to `advanced/` subdirectory
- Update imports (with deprecation warnings)

## Example: Simplified Usage

### Before (Complex)
```python
from guardian.core.betes_enhanced import EnhancedBETESCalculator
from guardian.core.aggregation_methods import AggregationMethod
from guardian.core.trust_models import TrustModel

calc = EnhancedBETESCalculator(
    aggregation_method=AggregationMethod.HARMONIC,
    trust_model=TrustModel.EXPONENTIAL,
    enable_uncertainty=True
)
result = calc.calculate(...)
```

### After (Simple)
```python
from guardian.core import calculate_quality_score

# Simple
score = calculate_quality_score(project_path)

# With options
score = calculate_quality_score(
    project_path,
    metric='cirs',  # or 'betes'
    include_uncertainty=True
)
```

## Benefits of Simplification

1. **Easier to Use**: 80% of users get what they need with one function call
2. **Faster Adoption**: Lower barrier to entry
3. **Better Documentation**: Can focus on simple use cases first
4. **Maintainability**: Clear separation between core and advanced
5. **Flexibility**: Advanced features still available for power users

## Conclusion

**Recommendation:**
1. ✅ Keep all current features (they're valuable)
2. ✅ Add CIRS as primary metric
3. ✅ Create simplified unified API
4. ✅ Organize into core/standard/advanced tiers
5. ✅ Progressive disclosure (simple by default)

**Result:** Best of both worlds - simple for most users, powerful for researchers.
