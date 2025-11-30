# GIR (GPU IR) Specification

A typed, SSA-based intermediate representation for GPU code targeting SPIR-V and CUDA backends.

## 1. Overview

GIR is a purpose-built GPU assembly/IR featuring:
- Strong typing (integers, unsigned integers, floats of various widths)
- Functions with basic blocks (SSA form)
- Branching and conditional control flow
- Unified atomic model covering all SPIR-V atomics with clean CUDA mapping

## 2. Module Structure

A **module** consists of global declarations and function definitions:

```
module ::= { global_decl | func_decl }*
```

## 3. Type System

### 3.1 Scalar Types

#### Integer Types (signed)
- `i8`   - 8-bit signed integer
- `i16`  - 16-bit signed integer
- `i32`  - 32-bit signed integer
- `i64`  - 64-bit signed integer

#### Integer Types (unsigned)
- `u8`   - 8-bit unsigned integer
- `u16`  - 16-bit unsigned integer
- `u32`  - 32-bit unsigned integer
- `u64`  - 64-bit unsigned integer

#### Floating Point Types
- `f16`  - 16-bit floating point (half precision)
- `f32`  - 32-bit floating point (single precision)
- `f64`  - 64-bit floating point (double precision)

### 3.2 Pointer Types

Pointers are typed with an address space and pointee type:

```
ptr[AS]<T>
```

Where `AS` is one of:
- `generic`  - Generic address space
- `global`   - Global device memory
- `shared`   - Shared/workgroup memory
- `local`    - Thread-local/function memory
- `const`    - Constant memory

**Examples:**
```
ptr[global]<u32>
ptr[shared]<i32>
ptr[const]<f32>
ptr[global]<f64>
```

**Backend Mapping:**
- **SPIR-V**: Maps to storage classes (StorageBuffer, Workgroup, Function, UniformConstant, etc.)
- **CUDA**: Maps to global/shared/local/constant memory spaces

## 4. Literals

### Integer Literals (signed)
```
int_lit ::= [0-9]+ ('i8' | 'i16' | 'i32' | 'i64')
```
Examples: `42i32`, `-10i64`, `127i8`

### Integer Literals (unsigned)
```
uint_lit ::= [0-9]+ ('u8' | 'u16' | 'u32' | 'u64')
```
Examples: `42u32`, `255u8`, `1000u64`

### Floating Point Literals
```
float_lit ::= [0-9]+ ('.' [0-9]+)? ('f16' | 'f32' | 'f64')
```
Examples: `1.0f32`, `3.14159f64`, `0.5f16`

**Shorthand suffixes** (when type is unambiguous):
- `i` → `i32`
- `u` → `u32`
- `f` → `f32`

## 5. Global Declarations

```
global_decl ::= 'global' '@' ident ':' ptr_type ['=' initializer]

initializer ::= 'addr' '(' hex_literal ')'
              | literal
```

**Examples:**
```
global @g_counter : ptr[global]<u32> = addr(0x1000)
global @weights   : ptr[global]<f32>
global @flags     : ptr[shared]<u64>
```

## 6. Function Declarations

```
func_decl ::= 'func' func_attr* '@' ident '(' param_list? ')' '->' ret_ty '{'
              basic_block+
              '}'

func_attr ::= 'kernel' | 'inline' | 'noinline'

param_list ::= param (',' param)*
param      ::= '%' ident ':' type

ret_ty     ::= type | 'void'
```

**Example:**
```
func kernel @increment(%ptr: ptr[global]<u32>) -> void {
entry:
  %old = atomic.rmw add %ptr, 1u ordering=acq_rel scope=workgroup
  ret
}
```

## 7. Basic Blocks and Control Flow

### 7.1 Basic Blocks

```
basic_block ::= label ':' instr* terminator

label ::= ident
```

### 7.2 Terminators

**Unconditional branch:**
```
br label
```

**Conditional branch:**
```
br_if %cond, label_true, label_false
```

**Return:**
```
ret              ; for void functions
ret %val         ; for non-void functions
```

**Example:**
```
func @abs_i32(%x: i32) -> i32 {
entry:
  %is_neg = icmp.lt %x, 0i
  br_if %is_neg, neg, done

neg:
  %neg_x = sub 0i, %x
  br done

done:
  %res = phi i32 [ %x, entry ], [ %neg_x, neg ]
  ret %res
}
```

### 7.3 PHI Nodes

```
%result = phi <type> [ %val1, label1 ], [ %val2, label2 ], ...
```

## 8. Core Instructions

All instructions operate on SSA values (`%ident`). Type checking is strict.

### 8.1 Move / Constants

```
%v = mov %x                    ; copy value
%v = iconst <int_lit>          ; signed integer constant
%v = uconst <uint_lit>         ; unsigned integer constant
%v = fconst <float_lit>        ; floating point constant
```

### 8.2 Arithmetic Operations

**Integer/Float arithmetic:**
```
%z = add %x, %y        ; addition (int/uint/float)
%z = sub %x, %y        ; subtraction
%z = mul %x, %y        ; multiplication
%z = div %x, %y        ; division (signed/unsigned/float)
%z = rem %x, %y        ; remainder (int/uint only)
%z = neg %x            ; arithmetic negation
```

**Backend mapping:**
- **SPIR-V**: OpIAdd, OpFAdd, OpISub, OpFSub, OpIMul, OpFMul, OpSDiv, OpUDiv, OpFDiv, OpSRem, OpUMod
- **CUDA**: Standard C operators (+, -, *, /, %)

### 8.3 Bitwise Operations

**Integer/unsigned types only:**
```
%z = and %x, %y        ; bitwise AND
%z = or  %x, %y        ; bitwise OR
%z = xor %x, %y        ; bitwise XOR
%z = not %x            ; bitwise NOT (one's complement)
%z = shl %x, %y        ; shift left
%z = shr %x, %y        ; shift right (logical for unsigned, arithmetic for signed)
```

### 8.4 Comparison Operations

All comparisons produce `u32` result (0 for false, 1 for true).

**Signed integer comparisons:**
```
%c = icmp.eq  %x, %y   ; equal
%c = icmp.ne  %x, %y   ; not equal
%c = icmp.lt  %x, %y   ; less than (signed)
%c = icmp.le  %x, %y   ; less or equal (signed)
%c = icmp.gt  %x, %y   ; greater than (signed)
%c = icmp.ge  %x, %y   ; greater or equal (signed)
```

**Unsigned integer comparisons:**
```
%c = ucmp.eq  %x, %y   ; equal
%c = ucmp.ne  %x, %y   ; not equal
%c = ucmp.lt  %x, %y   ; less than (unsigned)
%c = ucmp.le  %x, %y   ; less or equal (unsigned)
%c = ucmp.gt  %x, %y   ; greater than (unsigned)
%c = ucmp.ge  %x, %y   ; greater or equal (unsigned)
```

**Floating point comparisons (ordered):**
```
%c = fcmp.oeq %x, %y   ; ordered equal
%c = fcmp.one %x, %y   ; ordered not equal
%c = fcmp.olt %x, %y   ; ordered less than
%c = fcmp.ole %x, %y   ; ordered less or equal
%c = fcmp.ogt %x, %y   ; ordered greater than
%c = fcmp.oge %x, %y   ; ordered greater or equal
```

**Floating point comparisons (unordered):**
```
%c = fcmp.ueq %x, %y   ; unordered equal
%c = fcmp.une %x, %y   ; unordered not equal
%c = fcmp.ult %x, %y   ; unordered less than
%c = fcmp.ule %x, %y   ; unordered less or equal
%c = fcmp.ugt %x, %y   ; unordered greater than
%c = fcmp.uge %x, %y   ; unordered greater or equal
```

### 8.5 Select Operation

**Conditional select (ternary operator):**
```
%result = select %cond, %true_val, %false_val
```

Selects between two values based on a condition without branching.
- `%cond` must be a value that can be interpreted as boolean (typically u32 with 0=false, non-zero=true)
- `%true_val` and `%false_val` must have the same type
- Returns `%true_val` if `%cond` is non-zero, otherwise `%false_val`

**Example:**
```
%cmp = ucmp.lt %x, 10u
%result = select %cmp, %x, 10u   ; clamp to maximum of 10
```

**Backend mapping:**
- **SPIR-V**: OpSelect
- **CUDA**: Ternary operator `?:` or branchless computation

### 8.6 Memory Operations

**Load:**
```
%v = load %ptr
```
Type of `%v` is `T` if `%ptr : ptr[AS]<T>`

**Store:**
```
store %ptr, %v
```

Memory space is encoded in the pointer type.

### 8.7 Address Arithmetic

**Get element pointer:**
```
%p2 = gep %base, %index, stride=<uint_lit>
```
Where:
- `%base : ptr[AS]<T>`
- `%index : u32` (or other unsigned integer type)
- `stride` is in bytes

**Bitcast (type punning):**
```
%p2 = bitcast %p1
```
Changes pointer type while preserving address.

### 8.8 Type Conversions

**Bitcast (reinterpret bits):**
```
%y = bitcast %x
```
Reinterprets the bit pattern of `%x` as a different type without changing the bits.
Common uses: `f32 → u32`, `u32 → f32`, `i32 → u32`

**Integer truncation:**
```
%y = trunc %x
```
Truncates an integer to a smaller width. Example: `i64 → i32`, `u32 → u16`

**Integer zero extension:**
```
%y = zext %x
```
Zero-extends an unsigned integer to a larger width. Example: `u8 → u32`, `u16 → u64`

**Integer sign extension:**
```
%y = sext %x
```
Sign-extends a signed integer to a larger width. Example: `i8 → i32`, `i16 → i64`

**Float to signed int:**
```
%y = fptosi %x
```
Converts floating point to signed integer (truncates toward zero).

**Float to unsigned int:**
```
%y = fptoui %x
```
Converts floating point to unsigned integer (truncates toward zero).

**Signed int to float:**
```
%y = sitofp %x
```
Converts signed integer to floating point.

**Unsigned int to float:**
```
%y = uitofp %x
```
Converts unsigned integer to floating point.

**Float extension:**
```
%y = fpext %x
```
Extends a float to higher precision. Example: `f16 → f32`, `f32 → f64`

**Float truncation:**
```
%y = fptrunc %x
```
Truncates a float to lower precision. Example: `f64 → f32`, `f32 → f16`

**Backend mapping:**
- **SPIR-V**: OpBitcast, OpSConvert, OpUConvert, OpFConvert, OpConvertFToS, OpConvertFToU, OpConvertSToF, OpConvertUToF
- **CUDA**: C-style casts, `__float_as_uint`, `__uint_as_float`, etc.

## 9. Atomic Operations

GIR provides a unified atomic model with explicit memory ordering and scope.

### 9.1 Memory Ordering

```
ordering ::= relaxed | acquire | release | acq_rel | seq_cst
```

Semantics align with C11/C++11 memory model and SPIR-V.

### 9.2 Memory Scope

```
scope ::= invocation    ; single invocation/thread
        | subgroup      ; warp/wavefront
        | workgroup     ; CUDA block / OpenCL workgroup
        | device        ; entire device
        | system        ; system-wide (multi-device)
```

**Backend mapping:**
- **SPIR-V**: Maps directly to Scope enumeration
- **CUDA**: Maps to scope suffixes on atomic operations

### 9.3 Atomic Load

```
%val = atomic.load %ptr ordering=<ordering> scope=<scope>
```

**Backend mapping:**
- **SPIR-V**: OpAtomicLoad
- **CUDA**: `__atomic_load` or emulated with CAS loop

### 9.4 Atomic Store

```
atomic.store %ptr, %val ordering=<ordering> scope=<scope>
```

**Backend mapping:**
- **SPIR-V**: OpAtomicStore
- **CUDA**: `__atomic_store` or `atomicExch`/CAS loop

### 9.5 Atomic Read-Modify-Write

```
%old = atomic.rmw <op> %ptr, %val ordering=<ordering> scope=<scope>
```

Returns the old value at `%ptr`.

#### Integer/Unsigned Operations

```
<op> ::= add          ; atomic addition
       | sub          ; atomic subtraction
       | min_s        ; signed minimum
       | min_u        ; unsigned minimum
       | max_s        ; signed maximum
       | max_u        ; unsigned maximum
       | and          ; bitwise AND
       | or           ; bitwise OR
       | xor          ; bitwise XOR
       | exchange     ; atomic exchange (swap)
       | inc_wrap     ; wrapping increment
       | dec_wrap     ; wrapping decrement
```

#### Floating Point Operations

```
<op> ::= fadd         ; floating point addition
       | fmin         ; floating point minimum
       | fmax         ; floating point maximum
```

**Note**: Floating point atomics may require hardware support or be lowered to CAS loops.

#### Bit Flag Operations

```
<op> ::= flag_test_and_set    ; test and set flag
       | flag_clear           ; clear flag
```

**Backend mapping:**
- **SPIR-V**: OpAtomic* family (OpAtomicIAdd, OpAtomicSMin, OpAtomicUMax, OpAtomicAnd, OpAtomicOr, OpAtomicXor, OpAtomicExchange, OpAtomicFAddEXT, etc.)
- **CUDA**: `atomicAdd`, `atomicMin`, `atomicMax`, `atomicAnd`, `atomicOr`, `atomicXor`, `atomicExch`, `atomicAdd` (float), etc.

### 9.6 Atomic Compare-Exchange

```
%old = atomic.cmpxchg %ptr, %expected, %desired
                      ordering_succ=<ordering>
                      ordering_fail=<ordering>
                      scope=<scope>
```

Atomically:
1. Loads value from `%ptr`
2. If equal to `%expected`, stores `%desired`
3. Returns the loaded value

Success check:
```
%old     = atomic.cmpxchg %ptr, %expected, %desired ordering_succ=acq_rel ordering_fail=acquire scope=device
%success = icmp.eq %old, %expected
```

**Backend mapping:**
- **SPIR-V**: OpAtomicCompareExchange / OpAtomicCompareExchangeWeak
- **CUDA**: `atomicCAS`

## 10. Complete Examples

### 10.1 Conditional Branch

```
func @max_u32(%a: u32, %b: u32) -> u32 {
entry:
  %cmp = ucmp.ge %a, %b
  br_if %cmp, a_ge_b, otherwise

a_ge_b:
  ret %a

otherwise:
  ret %b
}
```

### 10.2 Atomic Counter Increment

```
global @counter : ptr[global]<u32> = addr(0x0)

func kernel @bump() -> void {
entry:
  %p   = mov @counter
  %old = atomic.rmw add %p, 1u ordering=acq_rel scope=device
  ret
}
```

### 10.3 Spinlock Implementation

```
global @lock : ptr[global]<u32>

func @lock_acquire(%lock: ptr[global]<u32>) -> void {
entry:
  br try_lock

try_lock:
  %old = atomic.cmpxchg %lock, 0u, 1u
                     ordering_succ=acq_rel
                     ordering_fail=acquire
                     scope=device
  %ok  = ucmp.eq %old, 0u
  br_if %ok, done, try_lock

done:
  ret
}

func @lock_release(%lock: ptr[global]<u32>) -> void {
entry:
  atomic.store %lock, 0u ordering=release scope=device
  ret
}
```

### 10.4 64-bit Floating Point Accumulation

```
global @sum : ptr[global]<f64>

func kernel @accumulate(%value: f64) -> void {
entry:
  %p = mov @sum
  %old = atomic.rmw fadd %p, %value ordering=acq_rel scope=device
  ret
}
```

### 10.5 16-bit Integer Min Reduction

```
func @reduce_min_i16(%ptr: ptr[shared]<i16>, %val: i16) -> void {
entry:
  %old = atomic.rmw min_s %ptr, %val ordering=acq_rel scope=workgroup
  ret
}
```

### 10.6 Byte Reversal (Endian Swap)

```
func @swap_bytes_u32(%x: u32) -> u32 {
entry:
  ; Extract individual bytes
  %const_8  = uconst 8u
  %const_16 = uconst 16u
  %const_24 = uconst 24u
  %const_ff = uconst 0xFFu

  %byte0    = and %x, %const_ff
  %shr_8    = shr %x, %const_8
  %byte1    = and %shr_8, %const_ff
  %shr_16   = shr %x, %const_16
  %byte2    = and %shr_16, %const_ff
  %byte3    = shr %x, %const_24

  ; Rebuild in reversed order
  %b1_shift = shl %byte2, %const_8
  %b2_shift = shl %byte1, %const_16
  %b3_shift = shl %byte0, %const_24

  %temp1 = or %byte3, %b1_shift
  %temp2 = or %temp1, %b2_shift
  %result = or %temp2, %b3_shift

  ret %result
}
```

### 10.7 Branchless Conditional Select

```
func @clamp_u32(%x: u32, %min: u32, %max: u32) -> u32 {
entry:
  ; Clamp to minimum
  %lt_min = ucmp.lt %x, %min
  %clamped_min = select %lt_min, %min, %x

  ; Clamp to maximum
  %gt_max = ucmp.gt %clamped_min, %max
  %clamped = select %gt_max, %max, %clamped_min

  ret %clamped
}
```

### 10.8 Float to Bytes (Bitcast and Manipulation)

```
func @float32_to_reversed_bytes(%f: f32) -> u32 {
entry:
  ; Convert float to u32 bit representation
  %bits = bitcast %f

  ; Byte-reverse for endian conversion
  %const_8  = uconst 8u
  %const_16 = uconst 16u
  %const_24 = uconst 24u
  %const_ff = uconst 0xFFu

  %byte0 = and %bits, %const_ff
  %shr_8 = shr %bits, %const_8
  %byte1 = and %shr_8, %const_ff
  %shr_16 = shr %bits, %const_16
  %byte2 = and %shr_16, %const_ff
  %byte3 = shr %bits, %const_24

  %b1_shift = shl %byte2, %const_8
  %b2_shift = shl %byte1, %const_16
  %b3_shift = shl %byte0, %const_24

  %temp1 = or %byte3, %b1_shift
  %temp2 = or %temp1, %b2_shift
  %reversed = or %temp2, %b3_shift

  ret %reversed
}
```

## 11. Backend Mapping Summary

### 11.1 SPIR-V

| GIR Construct | SPIR-V Mapping |
|---------------|----------------|
| Scalar types | OpTypeInt, OpTypeFloat with appropriate widths |
| Pointer types | OpTypePointer with storage class from address space |
| Arithmetic | OpIAdd, OpFAdd, OpISub, OpFSub, OpIMul, OpFMul, OpSDiv, OpUDiv, OpFDiv |
| Bitwise | OpBitwiseAnd, OpBitwiseOr, OpBitwiseXor, OpNot, OpShiftLeftLogical, OpShiftRightLogical/Arithmetic |
| Comparisons | OpIEqual, OpINotEqual, OpSLessThan, OpULessThan, OpFOrdLessThan, etc. |
| Select | OpSelect |
| Type conversions | OpBitcast, OpSConvert, OpUConvert, OpFConvert, OpConvertFToS, OpConvertFToU, OpConvertSToF, OpConvertUToF |
| Memory | OpLoad, OpStore with appropriate memory semantics |
| Atomic load | OpAtomicLoad with memory semantics and scope |
| Atomic store | OpAtomicStore with memory semantics and scope |
| Atomic RMW | OpAtomicIAdd, OpAtomicSMin, OpAtomicUMax, OpAtomicAnd, OpAtomicOr, OpAtomicXor, OpAtomicExchange, OpAtomicFAddEXT, etc. |
| Atomic CAS | OpAtomicCompareExchange with success/failure semantics |
| Control flow | OpBranch, OpBranchConditional, OpReturn, OpReturnValue, OpPhi |

### 11.2 CUDA

| GIR Construct | CUDA Mapping |
|---------------|--------------|
| Scalar types | int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, half, float, double |
| ptr[global] | `T*` in global memory |
| ptr[shared] | `__shared__ T*` |
| ptr[local] | Local/register variables |
| ptr[const] | `__constant__ T*` |
| Arithmetic | Standard C operators |
| Bitwise | Standard C bitwise operators (&, \|, ^, ~, <<, >>) |
| Select | Ternary operator ?: or branchless computation |
| Type conversions | C-style casts, `__float_as_uint`, `__uint_as_float`, `__int_as_float`, `__float_as_int`, etc. |
| Atomic add | atomicAdd, atomicAdd_block, atomicAdd_system |
| Atomic min/max | atomicMin, atomicMax (with signed/unsigned variants) |
| Atomic bitwise | atomicAnd, atomicOr, atomicXor |
| Atomic exchange | atomicExch |
| Atomic CAS | atomicCAS |
| Atomic float ops | atomicAdd (float/double), or CAS loop for fmin/fmax |

**Scope mapping in CUDA:**
- `workgroup` → `_block` suffix (e.g., `atomicAdd_block`)
- `device` → `_system` suffix (e.g., `atomicAdd_system`)
- `system` → `_system` suffix with system fence

## 12. Design Rationale

### 12.1 Why This Design is Simple Yet Complete

1. **Compact instruction set**: ~25 core operations (arithmetic, bitwise, comparisons, conversions, select) + ~10 atomic operations + control flow primitives
2. **Unified atomics**: Single `atomic.rmw` + `atomic.cmpxchg` covers all SPIR-V atomic operations
3. **Classic CFG**: Standard labels, branches, and phi nodes (proven SSA design)
4. **Natural backend mapping**: Address spaces, scopes, and orderings map directly to both SPIR-V and CUDA
5. **Type safety**: Strong typing prevents errors and simplifies code generation
6. **Complete type conversions**: Full coverage of numeric conversions for GPU interop
7. **Branchless operations**: Select instruction enables efficient GPU code patterns
8. **Extensible**: Easy to add new types (vectors, matrices) or operations without changing core design

### 12.2 Future Extensions

Possible additions without breaking core design:
- Vector types: `vec2<f32>`, `vec4<u32>`, etc.
- Matrix types for GPU compute
- Texture/sampler operations
- Subgroup/warp intrinsics
- Additional memory orderings (if needed)
- Structured control flow hints (for optimization)
