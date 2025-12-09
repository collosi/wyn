# Plan: Replace MIR Strings with Symbol IDs

## Problem

MIR is "stringly typed" - variables, functions, intrinsics, and lambdas are all identified by `String`. This undermines correctness invariants built by earlier passes:

```rust
pub enum ExprKind {
    Var(String),                    // Which binding does this refer to?
    Let { name: String, binding_id: u64, ... },  // Two identifiers for one thing
    Call { func: String, ... },     // Is this a valid function name?
    Intrinsic { name: String, ... }, // Is this a real intrinsic?
    Closure { lambda_name: String, ... }, // Maps to lambda_registry how?
}
```

Issues:
- Alpha-conversion/renaming is error-prone
- Monomorphization must coordinate string generation everywhere
- Typos in func/lambda_name fail late (lowering) not early
- Can't distinguish local vs def vs intrinsic at the type level

## Solution: Introduce Typed IDs

### Phase 1: Define ID Types

Add to `mir/mod.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DefId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntrinsicId {
    Index,
    Slice,
    Length,
    Assert,
    TupleAccess,
    RecordAccess,
    RecordUpdate,
    // ... enumerate all intrinsics
}
```

### Phase 2: Add Symbol Tables to Program

```rust
pub struct Program {
    pub defs: Vec<Def>,
    pub lambda_registry: Vec<(String, usize)>,  // or migrate to DefId

    // New: symbol tables
    pub def_names: HashMap<DefId, String>,      // DefId -> debug name
    pub local_tables: HashMap<DefId, LocalTable>, // per-function local info
}

pub struct LocalTable {
    pub locals: Vec<LocalInfo>,  // LocalId.0 indexes into this
}

pub struct LocalInfo {
    pub debug_name: String,
    pub ty: PolyType<TypeName>,
    pub span: Option<Span>,
}
```

### Phase 3: Update ExprKind

```rust
pub enum ExprKind {
    Literal(Literal),
    Unit,
    Var(LocalId),  // was String

    BinOp { op: BinaryOp, lhs: Box<Expr>, rhs: Box<Expr> },  // op could also be enum
    UnaryOp { op: UnaryOp, operand: Box<Expr> },

    If { cond: Box<Expr>, then_branch: Box<Expr>, else_branch: Box<Expr> },

    Let {
        local: LocalId,           // replaces name + binding_id
        value: Box<Expr>,
        body: Box<Expr>,
    },

    Loop {
        loop_var: LocalId,        // was String
        init: Box<Expr>,
        init_bindings: Vec<(LocalId, Expr)>,  // was (String, Expr)
        kind: LoopKind,
        body: Box<Expr>,
    },

    Call {
        func: DefId,              // was String
        args: Vec<Expr>,
    },

    Intrinsic {
        id: IntrinsicId,          // was String
        args: Vec<Expr>,
    },

    Closure {
        lambda: DefId,            // was String lambda_name
        captures: Vec<Expr>,
    },

    Attributed { attributes: Vec<Attribute>, expr: Box<Expr> },
    Materialize(Box<Expr>),
}
```

### Phase 4: Update Flattening

Flattening creates MIR from AST. Changes needed:

1. Maintain a `def_table: HashMap<String, DefId>` built from Program.defs
2. Maintain a per-function `local_env: HashMap<String, LocalId>`
3. When creating `Let`, allocate new `LocalId` and record in local table
4. When encountering `Var`, look up `LocalId` from `local_env`
5. When encountering function call, look up `DefId` from `def_table`
6. Parse intrinsic names into `IntrinsicId` enum (with exhaustive match)

### Phase 5: Update Downstream Passes

Each pass that reads/writes MIR needs updates:

- **normalize.rs**: Use LocalId instead of generating temp name strings
- **hoisting.rs**: Track LocalIds for hoisted constants
- **monomorphize.rs**: Generate new DefIds for specialized functions
- **spirv/lowering.rs**: Look up debug names from tables for SPIR-V debug info
- **glsl/lowering.rs**: Same for GLSL output

### Phase 6: Migrate binding_id

Currently `Let` has both `name: String` and `binding_id: u64`. After migration:
- `LocalId` replaces both
- `binding_id` use sites (backing stores, liveness) use `LocalId` directly
- Remove redundant `binding_id` field

## Implementation Order

1. Add ID types (LocalId, DefId, IntrinsicId) - no breakage
2. Add IntrinsicId enum with `from_str` / `to_str` methods
3. Add symbol tables to Program (empty initially)
4. Update flattening to populate tables AND use IDs (parallel strings for now)
5. Update each downstream pass one at a time
6. Remove string fields once all passes migrated
7. Remove binding_id once LocalId is used everywhere

## Benefits

- **Stronger invariants**: Var(LocalId) provably refers to a known local
- **Safer transforms**: No string coordination for renaming/inlining
- **Exhaustiveness**: IntrinsicId enum catches missing cases
- **Def-use chains**: Easy to track by ID
- **SSA-ready**: LocalId is natural SSA variable numbering
