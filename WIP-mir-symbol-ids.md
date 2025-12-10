# MIR Symbol IDs Refactoring - WIP

## Overview

Refactoring MIR to use typed symbol IDs instead of strings for variable and function references.

## Completed

### New ID Types (mir/mod.rs)
- `LocalId(u32)` - identifies local variables within a function
- `DefId(u32)` - identifies top-level definitions (functions, constants, etc.)
- `IntrinsicId` enum - TupleAccess, Index, Length, Assert, RecordAccess, RecordUpdate, DebugI32, DebugF32, DebugStr

### ExprKind Changes
- `Var(String)` → `Var(LocalId)`
- `Let { name, binding_id, value, body }` → `Let { local: LocalId, value, body }`
- `Call { func: String, args }` → `Call { func: DefId, func_name: Option<String>, args }`
- `Intrinsic { name: String, args }` → `Intrinsic { id: IntrinsicId, args }`
- `Closure { lambda_name: String, captures }` → `Closure { lambda: DefId, captures }`
- `LoopKind::For { var: String, ... }` → `LoopKind::For { var: LocalId, ... }`
- `LoopKind::ForRange { var: String, ... }` → `LoopKind::ForRange { var: LocalId, ... }`

### Updated Passes
- flattening.rs - emits LocalId/DefId, pre-registers function names
- lib.rs - pre-registers module function names before flattening
- folder.rs - updated visit_expr_call signature to include func_name
- normalize.rs - uses LocalId for let bindings
- binding_lifter.rs - uses LocalId
- materialize_hoisting.rs - uses LocalId/DefId
- monomorphization.rs - handles func_name for specialized functions
- reachability.rs - uses DefId with func_name fallback
- spirv/lowering.rs - resolves func_name for builtins, DefId for user functions
- glsl/lowering.rs - same pattern as SPIR-V
- diags.rs - Display impl shows func_name when available

### Test Files Updated
- binding_lifter_tests.rs - uses LocalId
- constant_folding_tests.rs - uses LocalId
- normalize_tests.rs - uses LocalId/DefId
- mir/tests.rs - uses LocalId

## Remaining Work

### 1. Update flattening_tests.rs String Patterns (9 tests failing)

The MIR output format changed:
- Variables now display as `_N` instead of `name{N}`
- Closures show `@closure(defN, ...)` instead of `@closure(_w_lam_name, ...)`

Failing tests:
- test_simple_function - expects `(x + y)`
- test_let_binding - expects `let x{`
- test_lambda_defunctionalization - expects `@closure(_w_lam_f_`
- test_nested_let - expects `let x{`
- test_for_range_loop - expects `for i <`
- test_if_expression - expects `if x then`
- test_unary_op - expects `(-x)`
- test_map_with_closure_application - expects `@closure(_w_lam_test_map_`
- test_direct_closure_call - expects `_w_lam_test_apply_0 f 10`

**Fix:** Update the expected string patterns in these tests to match new output format, OR update the Display impl to show more readable names.

### 2. Investigate "Unknown function: None" Error (2 tests failing)

Tests:
- `normalize_tests::test_normalize_loop_with_tuple_state`
- `monomorphization_tests::test_monomorphization_asserts_on_unresolved_size_params`

Error: `SpirvError("Unknown function: None", None)`

This happens in spirv/lowering.rs:2173 when:
1. `func_name` is `None` (not a builtin with explicit name)
2. `constructor.functions.get(func)` returns `None` (DefId not in functions map)

Possible causes:
- DefId assigned during flattening doesn't match position in final defs list (lambdas generated during flattening shift indices)
- Specialized functions from monomorphization have DefIds that don't correspond to actual defs
- Some function call path not setting func_name when it should

**Investigation needed:** Add debug logging to trace which function call is failing and why its DefId isn't found.

## Test Status

```
409 passed, 11 failed, 1 ignored
```

## Build Status

Builds successfully with 3 dead code warnings (can be cleaned up later):
- `local_tables_by_name` field in Flattener
- `reset_local_table` and `save_local_table` methods in Flattener
- `reset_local_table` method in Normalizer
