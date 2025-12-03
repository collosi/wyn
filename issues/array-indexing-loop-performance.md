# Array Indexing in Loops: Redundant OpStore Bug

## Summary

When indexing into an array with a dynamic index inside a loop, the codegen emits an `OpStore` instruction on every iteration, even when the array value is loop-invariant. This causes catastrophic performance degradation - in one test case, a shader that should render at 60fps instead takes 2+ seconds per frame.

## Root Cause

The array indexing intrinsic handler in `wyn-core/src/lowering.rs` (around line 2115) does this:

```rust
"index" => {
    let array_val = lower_expr(constructor, &args[0])?;
    let index_val = lower_expr(constructor, &args[1])?;

    // Store array in a variable to get a pointer
    let array_type = constructor.ast_type_to_spirv(&args[0].ty);
    let array_var = constructor.declare_variable("__index_tmp", array_type)?;
    constructor.builder.store(array_var, array_val, None, [])?;  // BUG: in loop body!

    // Use OpAccessChain to get pointer to element
    let elem_ptr = constructor.builder.access_chain(..., array_var, [index_val])?;
    Ok(constructor.builder.load(..., elem_ptr, ...)?)
}
```

The `declare_variable` correctly hoists the `OpVariable` to the function's variables block, but the `OpStore` is emitted in the *current* block - which is the loop body when indexing happens inside a loop.

## SPIR-V Constraints

1. **OpVariable placement**: All `OpVariable` instructions must be in the first basic block of a function.

2. **SSA form**: Values must be defined before use. We can't put `OpStore` in the variables block because the array value being stored may not be defined there yet.

3. **OpAccessChain requirement**: Dynamic array indexing requires a *pointer* to the array, not the array value itself. Hence we must store the value to a variable to get a pointer.

## Wyn Language Semantics

1. **Immutable bindings**: `let x = ...` creates an immutable binding.

2. **Loop variables**: In `loop (i, acc) = (init) while cond do body`, the variables are rebound each iteration via `OpPhi` - they're fresh bindings, not mutations.

3. **Arrays are values**: Arrays are immutable values, not references.

## Cases to Consider

| Source of Array | Loop-Invariant? | Ideal Behavior |
|-----------------|-----------------|----------------|
| Literal (`let edges = [...]`) | Yes | Store once at function entry |
| Function parameter | Yes | Store once at function entry |
| Computed before loop | Yes | Store once before loop |
| Loop variable (from OpPhi) | No | Must store each iteration |
| Computed inside loop | No | Must store each iteration |

## Proposed Solution

### Phase 1: Handle Constants (Easy)

For `OpConstant` and `OpConstantComposite` values:
- Add a cache: `HashMap<spirv::Word, spirv::Word>` mapping array value ID to temp variable ID
- On first index of a constant array, create temp and store
- On subsequent indexes, reuse the cached temp variable

This fixes literal arrays, which are the most common case.

### Phase 2: Handle Function Parameters (Medium)

Function parameters are also available from function entry:
- Track parameter value IDs
- Apply same caching strategy

### Phase 3: General Loop-Invariant Hoisting (Hard)

For computed arrays that are loop-invariant:
- Requires control flow analysis to determine if the array value dominates all uses
- Could use SPIR-V's dominator tree or track loop nesting during lowering
- May be overkill if most cases are covered by Phase 1-2

## Test Case

`testfiles/da_rasterizer_minus.wyn` with `sum_edges` call demonstrates the issue:
- `edges` array (12 elements) stored on each of 12 loop iterations
- `instA` array (18 elements) stored on each of 18 loop iterations
- Result: 18 + 18*12 = 234 redundant stores per pixel
- At 1920x1080, that's ~486 million redundant stores per frame

## Files Involved

- `wyn-core/src/lowering.rs`: The `"index"` intrinsic handler around line 2115
- `wyn-core/src/lowering.rs`: `declare_variable` and potential new `declare_variable_with_init` helper

## Related

This is similar to loop-invariant code motion (LICM) optimization, but at the codegen level rather than as a separate optimization pass.
