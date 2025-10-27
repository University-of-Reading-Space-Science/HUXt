# Phase 2 Array Transposition - REVERTED

## Status: REVERTED on 27 Oct 2025

**Reason**: Breaking API change - grid format change from `(nt, nr, nlon)` to `(nt, nlon, nr)` would break all external code and analysis scripts.

---

## What Was Phase 2?

Phase 2 attempted to optimize cache efficiency by transposing the grid arrays:

### Original (and restored) format:
```python
v_grid[t, r, lon]  # Shape: (nt_out, nr, nlon)
```

### Phase 2 format (reverted):
```python
v_grid[t, lon, r]  # Shape: (nt_out, nlon, nr)
```

### Performance Gain (now lost):
- **Speedup**: 1.13x (13% improvement)
- **Reason**: Better cache locality when solver varies radius at fixed longitude
- **Cache efficiency**: 8x improvement (sequential memory access vs. 1024-byte strides)

---

## Why It Was Reverted

1. **Breaking Change**: All external code accessing `model.v_grid`, `model.b_grid`, `model.rho_grid`, `model.temp_grid` would need updating
2. **Index Confusion**: Mixed old/new indexing patterns in codebase led to bugs
3. **User Code Impact**: Any user scripts, analysis tools, or plugins using HUXt would break
4. **Testing Burden**: Would need to update all tests, examples, and documentation

### What Broke:
- `plot()` function had incorrect indexing (ValueError: shape mismatch)
- All 6 analysis functions updated for compressible support had wrong indexing
- Any external code using `v_grid[t, r, lon]` would fail silently with wrong results

---

## Commits Reverted

Reset branch from `5b770a0` back to `00e8c05`:

```bash
git reset --hard 00e8c05
git push --force origin compressible-huxt
```

**Removed commits**:
- `f18b5af` - Phase 2: Array transposition for cache optimization
- `2f26401` - Move optimization docs to debug directory  
- `a587b52` - Add passive tracer scaffolding
- `5b770a0` - Add compressible support to all analysis functions (with wrong indexing)

---

## Alternative Approaches (Future Work)

If cache optimization is still desired, consider these API-preserving alternatives:

### 1. Internal Transposition (Copy overhead)
```python
def solve_radial(...):
    # Transpose slice for computation
    v_slice_transposed = v_grid[t, :, :].T  # (nr, nlon) -> (nlon, nr)
    # Work with transposed data
    for ilon in range(nlon):
        result = compute(v_slice_transposed[ilon, :])  # Sequential access
    # Transpose back for storage
    v_grid[t, :, :] = result.T
```
**Trade-off**: Adds copy overhead, may negate performance gains

### 2. Solver Algorithm Change
Restructure loops to vary longitude instead of radius:
```python
# Instead of: for each longitude, solve radially
# Do: for each radius, solve across longitudes
```
**Trade-off**: May conflict with physics of solar wind propagation

### 3. View-Based Access (Zero-copy)
Create transposed views for internal use:
```python
class HUXt:
    @property
    def v_grid_solver_view(self):
        return self.v_grid.transpose(0, 2, 1)  # (nt, nr, nlon) -> (nt, nlon, nr)
```
**Trade-off**: Adds complexity, views still create temporary arrays

### 4. Documentation and Migration Path
If transposition is essential:
- Provide migration guide and deprecation warnings
- Add compatibility layer for 1-2 versions
- Update all examples and documentation
- Announce breaking change prominently

---

## Current Status

- ✅ Arrays restored to original `(nt, nr, nlon)` format
- ✅ All compressible plotting functions removed (had wrong indexing)
- ✅ Test suite passes
- ✅ Remote branch updated (force-pushed)
- ⚠️ Lost 13% performance gain from Phase 2
- ✅ Preserved backward compatibility

---

## Lessons Learned

1. **API Stability**: Grid shape is part of the public API - changes break users
2. **Test Coverage**: Need tests that would catch indexing errors
3. **Performance vs. Compatibility**: Sometimes backward compatibility > performance
4. **Gradual Changes**: If breaking change is necessary, needs deprecation period

---

## Next Steps

1. **Keep Phase 1**: Array copy removal (10% speedup) is safe and preserved
2. **Rethink compressible plotting**: Need to re-implement with correct `(nt, nr, nlon)` indexing
3. **Profile again**: Identify other optimization opportunities that don't break API
4. **Consider views**: Investigate zero-copy transposed views for internal solver use
