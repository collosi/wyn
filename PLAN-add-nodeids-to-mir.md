# Plan: Add NodeIds to MIR and Context to Visitor

## Goal
Enable optimization passes by:
1. Adding unique `NodeId`s to MIR `Expr` and `Def` types
2. Adding a generic "context" parameter to visitor functions for per-branch state

## Part 1: Add NodeIds to MIR

### 1.1 Reuse ast::NodeId
Use existing `ast::NodeId` and `ast::NodeCounter` types. Import in `mir/mod.rs`:

```rust
use crate::ast::{NodeId, NodeCounter, Span, TypeName};
```

### 1.2 Add NodeId to Expr
Modify `mir/mod.rs`:
```rust
pub struct Expr {
    pub id: NodeId,      // NEW
    pub ty: Type<TypeName>,
    pub kind: ExprKind,
    pub span: Span,
}

impl Expr {
    pub fn new(id: NodeId, ty: Type<TypeName>, kind: ExprKind, span: Span) -> Self {
        Expr { id, ty, kind, span }
    }
}
```

### 1.3 Add NodeId to Def
Add NodeId to all Def variants:
```rust
pub enum Def {
    Function {
        id: NodeId,  // NEW
        name: String,
        // ... rest unchanged
    },
    Constant {
        id: NodeId,  // NEW
        name: String,
        // ...
    },
    Uniform {
        id: NodeId,  // NEW
        name: String,
        ty: Type<TypeName>,
    },
}
```

### 1.4 Use ast::NodeCounter
Reuse `ast::NodeCounter` which already has:
- `next(&mut self) -> NodeId`
- `mk_node(&mut self, kind, span) -> Node<T>`

Thread `NodeCounter` through passes that create MIR nodes.

### 1.5 Update Passes to Use NodeCounter

Files to update:
- `flattening.rs` - Add `NodeCounter` field to `Flattener`, use for all `Expr::new` calls
- `normalize.rs` - Add `NodeCounter` field to `Normalizer`, use when creating temp bindings
- `monomorphization.rs` - Add `NodeCounter` to `Monomorphizer`
- `constant_folding.rs` - Thread through
- `binding_lifter.rs` - Thread through
- `mir/visitor.rs` - Walk functions need to preserve IDs when reconstructing

### 1.6 Update Pipeline Stages
In `lib.rs`, the pipeline stages need to thread `NodeCounter`:
```rust
pub struct Flattened {
    pub mir: mir::Program,
    pub node_counter: NodeCounter,  // NEW
}

impl Flattened {
    pub fn normalize(self) -> Normalized {
        let (mir, node_counter) = normalize::normalize_program(self.mir, self.node_counter);
        Normalized { mir, node_counter }
    }
}
```

---

## Part 2: Add Context to Visitor

### 2.1 Add Generic Context Parameter
Modify `mir/visitor.rs` to add a `Ctx` associated type:

```rust
pub trait MirVisitor: Sized {
    type Error;
    type Ctx;  // NEW: per-branch context

    fn visit_expr(&mut self, e: Expr, ctx: &mut Self::Ctx) -> Result<Expr, Self::Error> {
        walk_expr(self, e, ctx)
    }

    // All visit methods gain ctx parameter...
}
```

### 2.2 Update Walk Functions
All `walk_*` functions need the context parameter:

```rust
pub fn walk_expr_if<V: MirVisitor>(
    v: &mut V,
    cond: Expr,
    then_branch: Expr,
    else_branch: Expr,
    expr: Expr,
    ctx: &mut V::Ctx,  // NEW
) -> Result<Expr, V::Error> {
    let cond = v.visit_expr(cond, ctx)?;
    // For branches, passes may want to clone/fork context
    let then_branch = v.visit_expr(then_branch, ctx)?;
    let else_branch = v.visit_expr(else_branch, ctx)?;
    // ...
}
```

### 2.3 Context Forking for Branches
For control flow (if/loop), passes may want independent contexts per branch:

```rust
pub trait MirVisitor: Sized {
    // ...

    /// Clone context for entering a new branch. Default: just use same context.
    fn fork_ctx(&self, ctx: &Self::Ctx) -> Self::Ctx
    where Self::Ctx: Clone
    {
        ctx.clone()
    }
}
```

Then in `walk_expr_if`:
```rust
let mut then_ctx = v.fork_ctx(ctx);
let mut else_ctx = v.fork_ctx(ctx);
let then_branch = v.visit_expr(then_branch, &mut then_ctx)?;
let else_branch = v.visit_expr(else_branch, &mut else_ctx)?;
```

### 2.4 Simple Context for Passes That Don't Need It
Passes that don't need per-branch context can use `()`:

```rust
impl MirVisitor for SimplePass {
    type Error = MyError;
    type Ctx = ();  // No context needed

    // ...
}
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `mir/mod.rs` | Import `NodeId`, add `id` field to `Expr` and all `Def` variants |
| `mir/visitor.rs` | Add `Ctx` type param, update all methods and walk functions |
| `flattening.rs` | Add `NodeCounter`, use for all Expr creation |
| `normalize.rs` | Accept and use `NodeCounter` |
| `monomorphization.rs` | Accept and use `NodeCounter` |
| `constant_folding.rs` | Thread `NodeCounter`, add `Ctx` impl |
| `binding_lifter.rs` | Thread `NodeCounter`, add `Ctx` impl |
| `lib.rs` | Update pipeline stages to thread `NodeCounter` |
| `diags.rs` | Update MIR display if needed |
| Various test files | Update `Expr::new` calls to include ID |

---

## Implementation Order

1. Import `NodeId` and update `Expr`/`Def` in `mir/mod.rs`
2. Update `mir/visitor.rs` with `Ctx` type parameter
3. Update `flattening.rs` to use `NodeCounter`
4. Update `normalize.rs` to accept `NodeCounter`
5. Update `monomorphization.rs`
6. Update remaining passes (`constant_folding.rs`, `binding_lifter.rs`)
7. Update `lib.rs` pipeline stages
8. Fix all tests

---

## Decisions

1. **Reuse `ast::NodeId`** - Single NodeId type shared between AST and MIR
2. **Uniform defs get NodeIds** - All Def variants have IDs for consistency
3. **Loop context forking** - TBD during implementation, start simple (no special forking)
