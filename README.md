# Wyn

A minimal compiler for a Futhark-like programming language that generates SPIR-V code for GPU shaders.

## Features

- Simple, Futhark-inspired syntax
- Type inference using polytype
- SPIR-V code generation for Vulkan/OpenGL shaders
- Support for vertex and fragment shaders
- Array literals and indexing
- Basic arithmetic operations

## Project Structure

The project is organized as a Rust workspace with two crates:

- `compiler/` - The core compiler library
  - `lexer.rs` - Tokenization using nom
  - `parser.rs` - Parsing to AST
  - `ast.rs` - Abstract syntax tree definitions  
  - `type_checker.rs` - Type checking with polytype
  - `codegen.rs` - SPIR-V code generation
  - `error.rs` - Error handling with thiserror
- `driver/` - The command-line executable using clap

## Example Program

```futhark
-- Full-screen triangle in NDC (like classic shader demos).
let verts: [3][4]f32 =
  [[-1.0f32, -1.0f32, 0.0f32, 1.0f32],
   [ 3.0f32, -1.0f32, 0.0f32, 1.0f32],
   [-1.0f32,  3.0f32, 0.0f32, 1.0f32]]

-- Vertex stage: return clip-space position for a given vertex index.
entry vertex_main (vertex_id: i32) : [4]f32 =
  verts[vertex_id]

-- Fragment stage: constant sky blue (#87CEEB).
let SKY_RGBA : [4]f32 =
  [135f32/255f32, 206f32/255f32, 235f32/255f32, 1.0f32]

entry fragment_main () : [4]f32 =
  SKY_RGBA
```

### Type Inference Example

```futhark
def zip_arrays xs ys = zip xs ys
```

This demonstrates Hindley-Milner type inference. The compiler infers:
```
val zip_arrays [d0] 't1 't2 : [d0]t1 -> [d0]t2 -> [d0](t1, t2)
```

Where `[d0]` represents arrays of dimension `d0`, and `t1`, `t2` are polymorphic type variables.

## Usage

### Compile a source file to SPIR-V:
```bash
wyn compile input.wyn -o output.spv
```

### Check a source file without generating output:
```bash
wyn check input.wyn
```

## Building

```bash
cargo build --release
```

## Testing

```bash
cargo test
```

## Supported Language Features

### Types
- `i32` - 32-bit signed integer
- `f32` - 32-bit floating point
- `[N]T` - Array of N elements of type T
- Multi-dimensional arrays: `[M][N]T`
- `(T1, T2, ...)` - Tuple types
- Function types with inference

### Declarations
- `let` - Constant bindings with explicit types
- `entry` - Shader entry points (must contain "vertex" or "fragment" in name)
- `def` - Function definitions with type inference

### Expressions
- Integer and float literals with type suffixes (e.g., `42`, `3.14f32`)
- Array literals: `[1.0f32, 2.0f32, 3.0f32]`
- Array indexing: `arr[i]`
- Division operator: `/`
- Function calls: `zip xs ys`
- Tuples: `(a, b)`
- Comments: `-- comment`

## Limitations

This is a minimal implementation focused on the specific example. Current limitations include:

- Dynamic array indexing simplified (returns first element as workaround - full SPIR-V implementation would require OpAccessChain)
- Limited operators (only division)
- No control flow structures
- Function calls and tuples supported in type checking but not SPIR-V generation
- No modules or imports
- Type inference implemented using polytype for `def` declarations