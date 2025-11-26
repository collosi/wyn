use crate::Compiler;
use crate::alias_checker::{AliasCheckResult, AliasChecker};

fn check_alias(source: &str) -> AliasCheckResult {
    let parsed = Compiler::parse(source).expect("parse failed");
    let elaborated = parsed.elaborate().expect("elaborate failed");
    let resolved = elaborated.resolve().expect("resolve failed");
    let type_checked = resolved.type_check().expect("type_check failed");

    let checker = AliasChecker::new(&type_checked.type_table);
    checker.check_program(&type_checked.ast).expect("alias check failed")
}

#[test]
fn test_no_error_simple() {
    let result = check_alias(r#"def main(x: i32): i32 = x + 1"#);
    assert!(!result.has_errors());
}

#[test]
fn test_use_after_move() {
    let source = r#"
def consume(arr: *[4]i32): i32 = arr[0]

def main(arr: [4]i32): i32 =
    let _ = consume(arr) in
    arr[0]
"#;
    let result = check_alias(source);
    assert!(result.has_errors(), "Expected use-after-move error");
}

#[test]
fn test_alias_through_let() {
    let source = r#"
def consume(arr: *[4]i32): i32 = arr[0]

def main(arr: [4]i32): i32 =
    let alias = arr in
    let _ = consume(arr) in
    alias[0]
"#;
    let result = check_alias(source);
    assert!(result.has_errors(), "Expected use-after-move error for alias");
}

#[test]
fn test_copy_type_no_tracking() {
    // i32 is copy, should work fine
    let source = r#"
def main(x: i32): i32 =
    let y = x in
    let z = x in
    y + z
"#;
    let result = check_alias(source);
    assert!(!result.has_errors());
}

// === Tricky edge cases ===

#[test]
fn test_transitive_aliasing() {
    // a -> b -> c, consume c, use a should error
    let source = r#"
def consume(arr: *[4]i32): i32 = arr[0]

def main(arr: [4]i32): i32 =
    let a = arr in
    let b = a in
    let c = b in
    let _ = consume(c) in
    a[0]
"#;
    let result = check_alias(source);
    assert!(
        result.has_errors(),
        "Expected error: a transitively aliases c which was consumed"
    );
}

#[test]
fn test_consume_alias_use_original() {
    // Consume the alias, then try to use the original - should error
    let source = r#"
def consume(arr: *[4]i32): i32 = arr[0]

def main(arr: [4]i32): i32 =
    let alias = arr in
    let _ = consume(alias) in
    arr[0]
"#;
    let result = check_alias(source);
    assert!(
        result.has_errors(),
        "Expected error: arr's backing store was consumed via alias"
    );
}

#[test]
fn test_shadowing_does_not_affect_outer() {
    // Inner variable shadows outer with same name, consume inner, use outer
    // This should be OK because they're different backing stores
    let source = r#"
def consume(arr: *[4]i32): i32 = arr[0]

def main(arr: [4]i32): i32 =
    let x = arr in
    let result =
        let x = [1, 2, 3, 4] in
        consume(x)
    in
    x[0]
"#;
    let result = check_alias(source);
    assert!(
        !result.has_errors(),
        "Should be OK: inner x is different from outer x"
    );
}

#[test]
#[ignore = "Conservative: doesn't track branch-specific state yet"]
fn test_if_branches_independent() {
    // Consume in one branch shouldn't affect use in other branch
    // (they're mutually exclusive execution paths)
    let source = r#"
def consume(arr: *[4]i32): i32 = arr[0]

def main(t: (bool, [4]i32)): i32 =
    let (cond, arr) = t in
    if cond then consume(arr) else arr[0]
"#;
    let result = check_alias(source);
    // This SHOULD be OK since the branches are mutually exclusive
    // But a conservative implementation might reject it
    assert!(
        !result.has_errors(),
        "Should be OK: branches are mutually exclusive"
    );
}

#[test]
fn test_use_after_if_that_consumes() {
    // Use after an if expression where one branch consumed - should error
    let source = r#"
def consume(arr: *[4]i32): i32 = arr[0]

def main(t: (bool, [4]i32)): i32 =
    let (cond, arr) = t in
    let _ = if cond then consume(arr) else 0 in
    arr[0]
"#;
    let result = check_alias(source);
    assert!(
        result.has_errors(),
        "Expected error: arr might have been consumed in if branch"
    );
}
