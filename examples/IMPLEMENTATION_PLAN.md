# Spinning Cube Implementation Plan

## Overview
This document outlines the features needed to compile and run the spinning cube demo in Wyn.

## Required Language Features

### 1. Vector Types (IMMEDIATE PRIORITY)
- [ ] Add vec2, vec3, vec4 tokens to lexer
- [ ] Parse vector type annotations
- [ ] Parse vector constructors: `vec3(1.0f32, 2.0f32, 3.0f32)`
- [ ] Add vector types to type system
- [ ] Implement vector swizzling: `v.x`, `v.xyz`
- [ ] Generate LLVM vector types in codegen

### 2. Basic Vector Operations
- [ ] Vector component access (`.x`, `.y`, `.z`, `.w`)
- [ ] Vector arithmetic (`+`, `-`, `*`, `/`)
- [ ] Scalar-vector multiplication
- [ ] Vector constructors with different arities

### 3. Matrix Support (Can use arrays initially)
- [ ] Type alias for mat4: `type mat4 = [4][4]f32`
- [ ] Matrix multiplication (already expressible with map/reduce)
- [ ] Matrix-vector multiplication

### 4. Built-in Math Functions
- [ ] `sin`, `cos`, `tan` for trigonometry
- [ ] Basic arithmetic already supported

### 5. Uniform Buffer Support
- [ ] Parse `#[uniform(binding = N)]` attribute
- [ ] Generate proper SPIR-V uniform decorations
- [ ] Handle uniform access in shaders

### 6. Multiple Shader Outputs
- [ ] Parse tuple return types for shaders
- [ ] Map tuple elements to different output locations
- [ ] Handle varying interpolation between stages

## Implementation Phases

### Phase 1: Basic Vector Support (FIRST COMMIT POINT)
1. Add vector tokens to lexer (vec2, vec3, vec4)
2. Parse vector type annotations
3. Parse vector constructor expressions
4. Add vector types to type system
5. Generate LLVM vector types

### Phase 2: Vector Operations (SECOND COMMIT POINT)
1. Implement component access (.x, .y, .z, .w)
2. Add vector arithmetic operations
3. Support vector constructor variations

### Phase 3: Uniforms and Attributes (THIRD COMMIT POINT)
1. Parse uniform attributes
2. Parse vertex input attributes
3. Generate proper SPIR-V decorations

### Phase 4: Math Functions (FOURTH COMMIT POINT)
1. Add sin/cos/tan as built-in functions
2. Link to LLVM math intrinsics

### Phase 5: Shader Interface (FINAL COMMIT POINT)
1. Support multiple outputs from vertex shader
2. Handle varying interpolation
3. Complete the pipeline

## Simplified First Version

For initial testing, we could simplify to:
- Fixed rotation angle (no time uniform)
- Single color per face (no per-vertex interpolation)
- Simpler geometry (triangle instead of cube)

## Testing Strategy
1. Start with simple vector operations
2. Test each phase independently
3. Build up to full cube gradually