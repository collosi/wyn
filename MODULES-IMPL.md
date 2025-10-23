# Implementation Notes for Wyn Module System

This document outlines approaches for implementing the ML-style module system in Wyn.

## Code Organization

All module system implementation should go in a new `src/module/` subdirectory:

```
src/
  module/
    mod.rs           // Public interface, ModuleElaborator
    env.rs           // ModuleEnv, ModuleSignature, name resolution
    signature.rs     // Signature matching and checking
    functor.rs       // Parametric module expansion
    flatten.rs       // Flattening modules to top-level declarations
  ast.rs             // AST definitions (already has module nodes)
  parser.rs          // Parser (already handles module syntax)
  type_checker.rs    // Type checker (receives flattened output)
  // ...
```

## Overview

The AST already has module system nodes defined (lines 368-442 in ast.rs):
- `ModuleBind`: Module declarations with optional parameters (functors)
- `ModuleTypeBind`: Module type declarations
- `ModuleExpression`: Module-level expressions (struct, name, ascription, lambda, application)
- `ModuleTypeExpression`: Module type expressions (signature, name, with, arrow)
- `Spec`: Specifications in module types (val, type, module, include)

## Required AST Extension

We need to add a new `TypeName` variant to represent module type parameters:

```rust
pub enum TypeName {
    Str(&'static str),
    Array,
    Vec,
    Size(usize),
    SizeVar(String),
    Unique,
    Record(Vec<(String, Type)>),
    Sum(Vec<(String, Vec<Type>)>),
    Existential(Vec<String>, Box<Type>),
    NamedParam(String, Box<Type>),
    ModuleTypeParam(String),  // NEW: Type parameters from module signatures
}
```

**Usage**:
- When parsing module signatures like `{ type t; val add: t -> t -> t }`, the type `t` is represented as `TypeName::ModuleTypeParam("t")`
- During signature matching, `ModuleTypeParam("t")` is unified/matched against the concrete type from the implementation
- If the type is abstract (no definition given), a unique name is generated and all `ModuleTypeParam("t")` occurrences are replaced with `TypeName::Str("M_t$abs_1")`
- This provides clear separation between module-level type parameters and concrete types

**Benefits**:
- Distinguishes module type parameters from concrete type names
- Enables proper tracking of abstract vs concrete types during elaboration
- Better error messages ("module type parameter 't'" vs "type 't'")
- Similar to how `SizeVar(String)` represents array size variables

## Current AST Structure

```rust
pub enum Declaration {
    ModuleBind(ModuleBind),         // module name params = body
    ModuleTypeBind(ModuleTypeBind), // module type name = sig
    Open(ModuleExpression),         // open mod_exp
    // ... other declarations
}

pub struct ModuleBind {
    pub name: String,
    pub params: Vec<ModuleParam>,              // For parametric modules
    pub signature: Option<ModuleTypeExpression>, // Optional ascription
    pub body: ModuleExpression,
}

pub enum ModuleExpression {
    Name(String),                               // Reference to module
    Ascription(Box<ModuleExpression>, ModuleTypeExpression),
    Lambda(Vec<ModuleParam>, Option<ModuleTypeExpression>, Box<ModuleExpression>),
    Application(Box<ModuleExpression>, Box<ModuleExpression>),
    Struct(Vec<Declaration>),                   // { declarations }
    Import(String),                             // import "path"
}

pub enum ModuleTypeExpression {
    Name(String),                               // Reference to module type
    Signature(Vec<Spec>),                       // { specs }
    With(Box<ModuleTypeExpression>, String, Vec<TypeParam>, Type), // with refinement
    Arrow(String, Box<ModuleTypeExpression>, Box<ModuleTypeExpression>),
    FunctorType(Box<ModuleTypeExpression>, Box<ModuleTypeExpression>),
}

pub enum Spec {
    Val(String, Vec<TypeParam>, Type),          // val name : type
    ValOp(String, Type),                        // val (op) : type
    Type(TypeBindKind, String, Vec<TypeParam>, Option<Type>), // type specs
    Module(String, ModuleTypeExpression),       // module name : sig
    Include(ModuleTypeExpression),              // include sig
}
```

## Implementation Strategy

### Phase 1: Module Environment & Name Resolution

Create a module elaboration phase that runs after parsing, before type checking.

**Module Environment**:
```rust
struct ModuleEnv {
    // Map module names to their elaborated signatures
    modules: HashMap<QualifiedName, ModuleSignature>,

    // Map module type names to their definitions
    module_types: HashMap<QualifiedName, ModuleTypeExpression>,

    // Current module path (for generating qualified names)
    path: Vec<String>,

    // Opened modules (for unqualified name lookup)
    opened: Vec<QualifiedName>,
}

struct ModuleSignature {
    // Types defined in this module
    types: HashMap<String, TypeInfo>,

    // Values defined in this module
    values: HashMap<String, Type>,

    // Nested modules
    modules: HashMap<String, ModuleSignature>,
}

enum TypeInfo {
    Abstract(String),      // Abstract type with unique name
    Alias(Type),           // Type alias
    Datatype(/* ... */),   // Datatype definition
}

type QualifiedName = Vec<String>;
```

**Name Resolution and Scoping**:

The module system creates hierarchical namespaces. Names are resolved in the following order:

1. **Local bindings** in the current scope
2. **Opened modules** (via `open M`)
3. **Parent module** (if nested)
4. **Top-level** bindings

**Qualified Names**:
- `M.x` is parsed as `QualifiedName(vec!["M"], "x")`
- `M.N.x` is parsed as `QualifiedName(vec!["M", "N"], "x")`
- During elaboration, these are resolved to fully qualified paths
- After flattening, they become simple identifiers with mangled names: `M_$_N_$_x`

**Example**:
```wyn
module M = {
  type t = i32
  let x: t = 42

  module N = {
    let y = x + 1  // Resolves to M.x (parent scope)
  }
}

let z = M.N.y  // Resolves to M.N.y (qualified name)
```

After elaboration:
```wyn
type M_$_t = i32
def M_$_x: i32 = 42
def M_$_N_$_y: i32 = M_$_x + 1
def z: i32 = M_$_N_$_y
```

**Scope Rules**:

1. **Module-level bindings** create a new namespace:
   ```wyn
   module M = { type t = i32 }
   module N = { type t = f32 }  // Different 't', no conflict
   ```

2. **Nested modules** can reference parent bindings:
   ```wyn
   module M = {
     type t = i32
     module N = {
       let x: t = 42  // 't' resolves to M.t
     }
   }
   ```

3. **Open brings names into scope**:
   ```wyn
   module M = { let x = 42 }
   module N = {
     open M
     let y = x  // 'x' resolves to M.x
   }
   ```

4. **Module parameters create local bindings**:
   ```wyn
   module F(A: {type t}) = {
     let x: A.t  // 'A' is in scope within F
   }
   ```

5. **Abstract types are opaque outside their module**:
   ```wyn
   module M: { type t; val x: t } = { type t = i32; let x = 42 }
   // Inside M: t = i32
   // Outside M: t = M.t$abs_1 (opaque)
   let y: i32 = M.x  // ERROR: M.t ≠ i32 externally
   ```

**Name Shadowing**:

Inner scopes shadow outer scopes:
```wyn
type t = i32
module M = {
  type t = f32  // Shadows outer 't'
  let x: t = 3.14  // Uses M.t (f32), not outer t (i32)
}
let y: t = 42  // Uses outer t (i32)
```

**Two Namespaces**:

Wyn has two separate namespaces:
1. **Value namespace**: modules, values, functions (all share the same namespace)
2. **Module type namespace**: module types (signatures)

This means:
```wyn
module M = { let x = 42 }
module type M = { val x: i32 }  // OK: different namespace
```

**Qualified Type References**:

Types can also be qualified:
```wyn
module M = { type t = i32 }
let x: M.t = 42  // Type reference to M.t
```

During elaboration, `M.t` is resolved to the type definition and may be:
- Expanded inline: `let x: i32 = 42`
- Kept as qualified: `let x: M_$_t = 42` (if t is abstract)

### Phase 2: Module Type Checking (Signature Matching)

When checking `module M: MT = Body`:

1. **Elaborate Body** to get inferred signature `MT'`
2. **Match `MT'` against `MT`**:
   - For each `val name : type` in MT:
     - Check that MT' has a value `name` with compatible type
   - For each `type name [= type]` in MT:
     - If `= type` given, check MT' has exactly this type alias
     - If abstract, generate unique name and record mapping
   - For each `module name : sig` in MT:
     - Recursively match nested module signatures
3. **Generate Filtered Signature** that only includes specs from MT
4. **Track Abstract Type Mappings** for scoping

**Abstract Types**:
```rust
// In signature: module M : { type t; val x : t }
// Implementation: module M = { type t = i32; val x = 42 }

// After elaboration:
// - Within M: t = i32 (concrete)
// - Outside M: t = M.t$abs_1 (opaque, unique name)
// - Type checker enforces that M.t$abs_1 ≠ i32 externally
```

### Phase 3: Parametric Module (Functor) Expansion

When applying a functor `F(M)`:

**Option A: Substitution-Based (Recommended for Wyn)**
```rust
// Given: module F(A: SIG) = { ... A.t ... A.x ... }
// Applied: module F_inst = F(Concrete)

fn apply_functor(
    functor: &ModuleBind,
    argument: &ModuleExpression,
    env: &ModuleEnv,
) -> ModuleExpression {
    // 1. Check argument matches parameter signature
    let arg_sig = elaborate_module(argument, env)?;
    check_signature_match(&arg_sig, &functor.params[0].signature)?;

    // 2. Build substitution map: param_name -> argument
    let subst = HashMap::from([(functor.params[0].name.clone(), argument.clone())]);

    // 3. Copy functor body and substitute all references
    substitute_module(&functor.body, &subst)
}

fn substitute_module(
    body: &ModuleExpression,
    subst: &HashMap<String, ModuleExpression>,
) -> ModuleExpression {
    match body {
        ModuleExpression::Name(n) if subst.contains_key(n) => {
            subst[n].clone()
        }
        ModuleExpression::Struct(decls) => {
            ModuleExpression::Struct(
                decls.iter().map(|d| substitute_decl(d, subst)).collect()
            )
        }
        // ... handle other cases
    }
}
```

Each functor application creates a fresh copy. This is simple but can lead to code duplication.

**Option B: Applicative Functors (More Complex)**
Share structure between multiple applications of the same functor. Requires tracking which applications have compatible arguments. More complex to implement but reduces duplication.

For Wyn, **Option A is recommended** initially. GPU shaders are small, so duplication is acceptable.

### Phase 4: Flattening & Hoisting

After elaboration, flatten the module hierarchy into top-level declarations:

```wyn
module M = {
  type t = i32
  let x: t = 42
}
module N = {
  let y = M.x + 1
}
```

Becomes:
```wyn
type M_t = i32
def M_x: i32 = 42
def N_y: i32 = M_x + 1
```

**Qualified Name Generation**:
- Simple: concatenate module path with `_`: `M.N.x` → `M_N_x`
- Better: mangle to avoid collisions: `M.N.x` → `M_$_N_$_x`
- Best: use unique IDs: `M.N.x` → `x_$42` with metadata tracking original path

### Phase 5: Integration with Type Checker

The type checker receives the flattened AST and:
- Resolves qualified names in the flattened scope
- Type checks as normal (modules are already gone)
- Abstract types are represented as unique `TypeName` variants

**Handling `open M`**:
After elaboration, `open M` is just adding `M.*` to the local scope. In the flattened output:
```wyn
module M = { let x = 42 }
def test = {
  open M;  // brings M_x into scope as 'x'
  x + 1
}
```

Becomes:
```wyn
def M_x = 42
def test = M_x + 1  // 'x' resolved to M_x
```

## Detailed Implementation Steps

### Step 1: Parser (Already Done)
The parser already handles module syntax. No changes needed.

### Step 2: Module Elaborator

Create `src/module_elaborator.rs`:

```rust
pub struct ModuleElaborator {
    env: ModuleEnv,
    abstract_type_counter: u32,
}

impl ModuleElaborator {
    pub fn elaborate_program(&mut self, program: Program) -> Result<Program> {
        // Two-pass approach:
        // Pass 1: Collect all module and module type bindings
        for decl in &program.declarations {
            match decl {
                Declaration::ModuleBind(mb) => {
                    self.collect_module_binding(mb)?;
                }
                Declaration::ModuleTypeBind(mtb) => {
                    self.collect_module_type_binding(mtb)?;
                }
                _ => {}
            }
        }

        // Pass 2: Elaborate and flatten
        let mut flat_decls = Vec::new();
        for decl in program.declarations {
            flat_decls.extend(self.elaborate_declaration(decl)?);
        }

        Ok(Program { declarations: flat_decls })
    }

    fn elaborate_declaration(&mut self, decl: Declaration) -> Result<Vec<Declaration>> {
        match decl {
            Declaration::ModuleBind(mb) => self.elaborate_module_bind(mb),
            Declaration::ModuleTypeBind(_) => Ok(vec![]), // Erased
            Declaration::Open(me) => self.elaborate_open(me),
            _ => Ok(vec![decl]), // Pass through
        }
    }

    fn elaborate_module_bind(&mut self, mb: ModuleBind) -> Result<Vec<Declaration>> {
        // If parametric module (has params), defer until applied
        if !mb.params.is_empty() {
            self.env.modules.insert(
                self.qualify(mb.name.clone()),
                self.infer_functor_signature(&mb)?,
            );
            return Ok(vec![]);
        }

        // Non-parametric module: elaborate body
        let body_decls = self.elaborate_module_expr(&mb.body)?;

        // If has signature, check and filter
        if let Some(sig) = mb.signature {
            self.check_and_filter(body_decls, &sig)
        } else {
            Ok(body_decls)
        }
    }

    fn elaborate_module_expr(&mut self, expr: &ModuleExpression) -> Result<Vec<Declaration>> {
        match expr {
            ModuleExpression::Struct(decls) => {
                let mut result = Vec::new();
                for decl in decls {
                    result.extend(self.elaborate_declaration(decl.clone())?);
                }
                Ok(result)
            }
            ModuleExpression::Name(n) => {
                // Look up module and return its declarations
                self.lookup_module_decls(n)
            }
            ModuleExpression::Application(func, arg) => {
                self.apply_functor(func, arg)
            }
            ModuleExpression::Ascription(body, sig) => {
                let decls = self.elaborate_module_expr(body)?;
                self.check_and_filter(decls, sig)
            }
            // ... other cases
        }
    }

    fn apply_functor(
        &mut self,
        func: &ModuleExpression,
        arg: &ModuleExpression,
    ) -> Result<Vec<Declaration>> {
        // Get the functor definition
        let functor = self.lookup_functor(func)?;

        // Elaborate argument
        let arg_sig = self.infer_module_signature(arg)?;

        // Check argument matches parameter signature
        self.check_signature_match(&arg_sig, &functor.params[0].signature)?;

        // Substitute and elaborate body
        let subst_body = self.substitute_functor_body(&functor, arg);
        self.elaborate_module_expr(&subst_body)
    }

    fn check_signature_match(
        &mut self,
        impl_sig: &ModuleSignature,
        req_sig: &ModuleTypeExpression,
    ) -> Result<()> {
        match req_sig {
            ModuleTypeExpression::Signature(specs) => {
                for spec in specs {
                    match spec {
                        Spec::Val(name, _type_params, ty) => {
                            // Check impl has value with compatible type
                            let impl_ty = impl_sig.values.get(name).ok_or_else(|| {
                                CompilerError::ModuleError(format!("Missing value: {}", name))
                            })?;
                            // Type compatibility check (may involve subtyping)
                            self.check_type_compat(impl_ty, ty)?;
                        }
                        Spec::Type(kind, name, _params, defn) => {
                            match defn {
                                Some(ty) => {
                                    // Manifest type: must match exactly
                                    let impl_ty_info = impl_sig.types.get(name).ok_or_else(|| {
                                        CompilerError::ModuleError(format!("Missing type: {}", name))
                                    })?;
                                    // Check exact match
                                }
                                None => {
                                    // Abstract type: generate unique name
                                    let abstract_name = self.fresh_abstract_type();
                                    // Record abstraction
                                }
                            }
                        }
                        // ... handle other specs
                    }
                }
                Ok(())
            }
            // ... handle other module type expressions
        }
    }

    fn fresh_abstract_type(&mut self) -> String {
        let name = format!("$abs_{}", self.abstract_type_counter);
        self.abstract_type_counter += 1;
        name
    }
}
```

### Step 3: Type Checker Integration

Minimal changes needed:
- Accept `QualifiedName` expressions
- Resolve qualified names in the environment
- Abstract types are just unique `TypeName::Str` variants

### Step 4: Testing

```rust
#[test]
fn test_simple_module() {
    let input = r#"
        module M = {
            type t = i32
            let x: t = 42
        }
        let y = M.x
    "#;

    let elaborated = elaborate(input)?;
    // Should produce:
    // type M_t = i32
    // def M_x: i32 = 42
    // def y: i32 = M_x
}

#[test]
fn test_functor() {
    let input = r#"
        module type Addable = {
            type t
            val add: t -> t -> t
        }

        module Sum(M: Addable) = {
            let sum (xs: []M.t): M.t = reduce M.add xs
        }

        module IntAdd = {
            type t = i32
            let add (x: i32) (y: i32): i32 = x + y
        }

        module SumInts = Sum(IntAdd)
    "#;

    let elaborated = elaborate(input)?;
    // Should produce Sum instantiated with IntAdd
}

#[test]
fn test_abstract_type() {
    let input = r#"
        module M: { type t; val x: t } = {
            type t = i32
            let x: t = 42
        }

        let y = M.x  // OK: y has type M.t
        let z: i32 = M.x  // ERROR: M.t ≠ i32 externally
    "#;

    // Should error on z
}
```

## Error Messages

Provide helpful errors for:

1. **Signature Mismatch**:
   ```
   Module M does not match signature:
     Missing value: add : i32 -> i32 -> i32
   ```

2. **Abstract Type Escape**:
   ```
   Type M.t escapes its scope
   ```

3. **Cyclic Modules**:
   ```
   Cyclic module dependency: M -> N -> M
   ```

4. **Functor Arity**:
   ```
   Functor Sum expects 1 argument, got 2
   ```

## Future Enhancements

1. **Include**: `include M` brings M's contents into current module
2. **Recursive Modules**: `module rec M = { ... }` for mutual recursion
3. **First-class Modules**: Allow modules as values (probably too complex)
4. **Separate Compilation**: Compile modules independently
5. **Module Type Inference**: Infer module types instead of requiring ascription

## References

- Futhark module system: [github.com/diku-dk/futhark](https://github.com/diku-dk/futhark/tree/master/src/Language/Futhark/Semantic.hs)
- "A Modular Module System" by Xavier Leroy
- SML Definition (Revised): Chapter on modules
- OCaml manual: Module system chapter
