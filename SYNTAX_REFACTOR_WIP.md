# Wyn Syntax Refactoring - Work In Progress

## Summary

Refactoring Wyn's core syntax to be more Rust-like. This is a **hard cutover** - no backward compatibility.

## New Syntax

| Feature | Old Syntax | New Syntax |
|---------|-----------|------------|
| Function def | `def foo [n] 'a (x: [n]a) (y: a): a = body` | `def foo<[n], A>(x: [n]A, y: A) -> A = body` |
| Lambda | `\x y -> x + y` | `\|x, y\| x + y` |
| Typed lambda | `\(x: i32) -> x + 1` | `\|x: i32\| x + 1` |
| Lambda w/ return type | N/A | `\|x\| -> i32 x + 1` |
| Type signature | `sig foo 'a: a -> a -> a` | `sig foo<A>: A -> A -> A` |
| Type variables | `'a`, `'t` | `A`, `T` (uppercase) |
| No-param function | `def foo = 42` | `def foo() = 42` |
| Unit pattern param | `def foo () = 42` | `def foo(()) = 42` |

## Completed

### Parser Changes (`wyn-core/src/parser.rs`)
- [x] `parse_generic_params()` - parses `<[n], A, B>` after function/sig name
- [x] `parse_comma_separated_params()` - parses `(x: T, y: U)` style params
- [x] `parse_lambda()` - parses `|x, y| body` with optional `-> type`
- [x] Updated `parse_decl()` for new function syntax
- [x] Updated `parse_sig_decl()` for new sig syntax
- [x] Updated entry point parsing for new syntax
- [x] Fixed `|` disambiguation (lambda delimiter vs bitwise OR operator)
- [x] Fixed `let` vs `def` parsing (let doesn't require parens)

### Parser Module (`wyn-core/src/parser/module.rs`)
- [x] Updated `parse_spec()` to use `parse_type_params()`
- [x] Updated `parse_type_bind()` for new generic syntax
- [x] Removed unused `can_start_type_param()` function

### Test File Conversions
- [x] `prelude/math.wyn` - 49 defs, 157 sigs
- [x] `prelude/soacs.wyn` - 5 defs with type params
- [x] `prelude/graphics.wyn` - 4 defs
- [x] `prelude/rand.wyn` - 5 defs
- [x] `prelude/gdp.wyn` - 4 defs
- [x] `wyn-core/src/parser/tests.rs` - 159 tests passing
- [x] `wyn-core/src/flattening_tests.rs` - 44 tests passing
- [x] `wyn-core/src/alias_checker_tests.rs` - 22 tests passing (1 ignored)

## Remaining Work

### Test Files to Convert
- [ ] `wyn-core/src/types/checker_tests.rs` - ~163 defs, 53 lambdas
- [ ] `wyn-core/src/monomorphization_tests.rs` - 4 defs, 2 lambdas
- [ ] `wyn-core/src/normalize_tests.rs` - 2 defs
- [ ] Any other Rust test files with inline Wyn code

### Source Files to Convert
- [ ] `testfiles/primitives.wyn` - 36 defs
- [ ] `testfiles/seascape.wyn` - 33 defs
- [ ] `testfiles/da_rasterizer.wyn` - 26 defs, 2 lambdas
- [ ] `testfiles/holodice.wyn` - 8 defs, 20 lambdas
- [ ] `testfiles/entrylevel.wyn`
- [ ] `testfiles/wall_anneal.wyn`
- [ ] `testfiles/lava.wyn`
- [ ] `testfiles/red_triangle.wyn`
- [ ] `testfiles/compute_test.wyn`
- [ ] `testfiles/debug_test.wyn`
- [ ] `examples/spinning_cube.wyn` - 14 defs, 3 lambdas

## Conversion Patterns

### Function Definitions
```
# Old: space-separated curried params
def add (x: f32) (y: f32): f32 = x + y

# New: comma-separated params in parens, arrow return type
def add(x: f32, y: f32) -> f32 = x + y
```

### Generic Functions
```
# Old: size params in brackets, type params with tick
def map [n] 'a 'b (f: a -> b) (xs: [n]a): [n]b = ...

# New: all generics in angle brackets, uppercase type vars
def map<[n], A, B>(f: A -> B, xs: [n]A) -> [n]B = ...
```

### Lambdas
```
# Old: backslash with arrow
\x -> x + 1
\(x: i32) -> x + 1

# New: pipe delimiters
|x| x + 1
|x: i32| x + 1
|x| -> i32 x + 1  # with return type
```

### Type Signatures
```
# Old
sig sum 'a: [n]a -> a

# New
sig sum<A>: [n]A -> A
```

## Notes

- Uniforms keep old colon syntax since they're variable declarations, not functions:
  `#[uniform(binding=0)] def material_color: vec3`
- Empty parens `()` means 0 params; use `(())` for explicit unit pattern
- The `|` character is disambiguated: at expression start = lambda, after expression = bitwise OR
