//! Pattern matching utilities shared across compiler passes.
//!
//! This module provides a generic API for extracting bindings from patterns,
//! which can be used in type checking, code generation, and other passes.

use crate::ast::{Pattern, PatternKind};

/// A binding extracted from a pattern: (name, value).
pub type Binding<T> = (String, T);

/// Trait for values that can be decomposed according to patterns.
///
/// Different passes implement this trait for their value types:
/// - Type checker: implements for `Type`
/// - Code generator: implements for registers/values
pub trait PatternValue: Clone {
    /// Extract the i-th element from a tuple value.
    /// Returns None if the value is not a tuple or index is out of bounds.
    fn tuple_element(&self, index: usize) -> Option<Self>;

    /// Get the number of elements if this is a tuple.
    /// Returns None if not a tuple.
    fn tuple_len(&self) -> Option<usize>;
}

/// Error that can occur during pattern matching.
#[derive(Debug, Clone)]
pub enum PatternError {
    /// Pattern expects a tuple but value is not a tuple
    NotATuple,
    /// Tuple pattern has wrong number of elements
    TupleLengthMismatch { expected: usize, actual: usize },
    /// Pattern kind not supported
    UnsupportedPattern(String),
}

/// Extract all bindings from a pattern matched against a value.
///
/// Returns a list of (name, value) pairs for all Name patterns in the tree.
/// Wildcards and Units produce no bindings.
///
/// # Example
/// ```ignore
/// // Pattern: (x, (y, z))
/// // Value: (1, (2, 3))
/// // Result: [("x", 1), ("y", 2), ("z", 3)]
/// ```
pub fn extract_bindings<T: PatternValue>(
    pattern: &Pattern,
    value: T,
) -> Result<Vec<Binding<T>>, PatternError> {
    let mut bindings = Vec::new();
    extract_bindings_inner(pattern, value, &mut bindings)?;
    Ok(bindings)
}

fn extract_bindings_inner<T: PatternValue>(
    pattern: &Pattern,
    value: T,
    bindings: &mut Vec<Binding<T>>,
) -> Result<(), PatternError> {
    match &pattern.kind {
        PatternKind::Name(name) => {
            bindings.push((name.clone(), value));
            Ok(())
        }

        PatternKind::Wildcard => {
            // Wildcard binds nothing
            Ok(())
        }

        PatternKind::Unit => {
            // Unit pattern binds nothing
            Ok(())
        }

        PatternKind::Tuple(patterns) => {
            let len = value.tuple_len().ok_or(PatternError::NotATuple)?;
            if len != patterns.len() {
                return Err(PatternError::TupleLengthMismatch {
                    expected: patterns.len(),
                    actual: len,
                });
            }

            for (i, sub_pattern) in patterns.iter().enumerate() {
                let elem = value.tuple_element(i).ok_or(PatternError::NotATuple)?;
                extract_bindings_inner(sub_pattern, elem, bindings)?;
            }
            Ok(())
        }

        PatternKind::Typed(inner, _ty) => {
            // Type annotation doesn't affect binding extraction
            // (type checking happens separately)
            extract_bindings_inner(inner, value, bindings)
        }

        PatternKind::Attributed(_, inner) => {
            // Attributes don't affect binding extraction
            extract_bindings_inner(inner, value, bindings)
        }

        PatternKind::Literal(_) => {
            // Literal patterns don't bind anything (used for matching)
            Ok(())
        }

        PatternKind::Record(_) => {
            Err(PatternError::UnsupportedPattern("Record patterns".to_string()))
        }

        PatternKind::Constructor(_, _) => {
            Err(PatternError::UnsupportedPattern("Constructor patterns".to_string()))
        }
    }
}

/// Get all names bound by a pattern (without values).
///
/// Useful for checking what variables a pattern introduces.
pub fn bound_names(pattern: &Pattern) -> Vec<String> {
    let mut names = Vec::new();
    collect_names(pattern, &mut names);
    names
}

fn collect_names(pattern: &Pattern, names: &mut Vec<String>) {
    match &pattern.kind {
        PatternKind::Name(name) => {
            names.push(name.clone());
        }
        PatternKind::Wildcard | PatternKind::Unit | PatternKind::Literal(_) => {}
        PatternKind::Tuple(patterns) => {
            for p in patterns {
                collect_names(p, names);
            }
        }
        PatternKind::Typed(inner, _) | PatternKind::Attributed(_, inner) => {
            collect_names(inner, names);
        }
        PatternKind::Record(fields) => {
            for field in fields {
                if let Some(p) = &field.pattern {
                    collect_names(p, names);
                } else {
                    // Shorthand: field name is the binding
                    names.push(field.field.clone());
                }
            }
        }
        PatternKind::Constructor(_, patterns) => {
            for p in patterns {
                collect_names(p, names);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Header, NodeId, Span};

    // Simple test value that can be a scalar or tuple
    #[derive(Clone, Debug, PartialEq)]
    enum TestValue {
        Scalar(i32),
        Tuple(Vec<TestValue>),
    }

    impl PatternValue for TestValue {
        fn tuple_element(&self, index: usize) -> Option<Self> {
            match self {
                TestValue::Tuple(elems) => elems.get(index).cloned(),
                _ => None,
            }
        }

        fn tuple_len(&self) -> Option<usize> {
            match self {
                TestValue::Tuple(elems) => Some(elems.len()),
                _ => None,
            }
        }
    }

    fn mk_pattern(kind: PatternKind) -> Pattern {
        Pattern {
            h: Header {
                id: NodeId(0),
                span: Span { start_line: 0, start_col: 0, end_line: 0, end_col: 0 },
            },
            kind,
        }
    }

    #[test]
    fn test_simple_name() {
        let pattern = mk_pattern(PatternKind::Name("x".to_string()));
        let value = TestValue::Scalar(42);

        let bindings = extract_bindings(&pattern, value).unwrap();
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].0, "x");
        assert_eq!(bindings[0].1, TestValue::Scalar(42));
    }

    #[test]
    fn test_wildcard() {
        let pattern = mk_pattern(PatternKind::Wildcard);
        let value = TestValue::Scalar(42);

        let bindings = extract_bindings(&pattern, value).unwrap();
        assert_eq!(bindings.len(), 0);
    }

    #[test]
    fn test_tuple_pattern() {
        let pattern = mk_pattern(PatternKind::Tuple(vec![
            mk_pattern(PatternKind::Name("x".to_string())),
            mk_pattern(PatternKind::Name("y".to_string())),
        ]));
        let value = TestValue::Tuple(vec![
            TestValue::Scalar(1),
            TestValue::Scalar(2),
        ]);

        let bindings = extract_bindings(&pattern, value).unwrap();
        assert_eq!(bindings.len(), 2);
        assert_eq!(bindings[0], ("x".to_string(), TestValue::Scalar(1)));
        assert_eq!(bindings[1], ("y".to_string(), TestValue::Scalar(2)));
    }

    #[test]
    fn test_nested_tuple() {
        // (x, (y, z))
        let pattern = mk_pattern(PatternKind::Tuple(vec![
            mk_pattern(PatternKind::Name("x".to_string())),
            mk_pattern(PatternKind::Tuple(vec![
                mk_pattern(PatternKind::Name("y".to_string())),
                mk_pattern(PatternKind::Name("z".to_string())),
            ])),
        ]));
        let value = TestValue::Tuple(vec![
            TestValue::Scalar(1),
            TestValue::Tuple(vec![
                TestValue::Scalar(2),
                TestValue::Scalar(3),
            ]),
        ]);

        let bindings = extract_bindings(&pattern, value).unwrap();
        assert_eq!(bindings.len(), 3);
        assert_eq!(bindings[0], ("x".to_string(), TestValue::Scalar(1)));
        assert_eq!(bindings[1], ("y".to_string(), TestValue::Scalar(2)));
        assert_eq!(bindings[2], ("z".to_string(), TestValue::Scalar(3)));
    }

    #[test]
    fn test_tuple_with_wildcard() {
        // (x, _, z)
        let pattern = mk_pattern(PatternKind::Tuple(vec![
            mk_pattern(PatternKind::Name("x".to_string())),
            mk_pattern(PatternKind::Wildcard),
            mk_pattern(PatternKind::Name("z".to_string())),
        ]));
        let value = TestValue::Tuple(vec![
            TestValue::Scalar(1),
            TestValue::Scalar(2),
            TestValue::Scalar(3),
        ]);

        let bindings = extract_bindings(&pattern, value).unwrap();
        assert_eq!(bindings.len(), 2);
        assert_eq!(bindings[0], ("x".to_string(), TestValue::Scalar(1)));
        assert_eq!(bindings[1], ("z".to_string(), TestValue::Scalar(3)));
    }

    #[test]
    fn test_bound_names() {
        // (x, (y, _))
        let pattern = mk_pattern(PatternKind::Tuple(vec![
            mk_pattern(PatternKind::Name("x".to_string())),
            mk_pattern(PatternKind::Tuple(vec![
                mk_pattern(PatternKind::Name("y".to_string())),
                mk_pattern(PatternKind::Wildcard),
            ])),
        ]));

        let names = bound_names(&pattern);
        assert_eq!(names, vec!["x", "y"]);
    }

    #[test]
    fn test_tuple_length_mismatch() {
        let pattern = mk_pattern(PatternKind::Tuple(vec![
            mk_pattern(PatternKind::Name("x".to_string())),
            mk_pattern(PatternKind::Name("y".to_string())),
        ]));
        let value = TestValue::Tuple(vec![TestValue::Scalar(1)]);

        let result = extract_bindings(&pattern, value);
        assert!(matches!(result, Err(PatternError::TupleLengthMismatch { .. })));
    }
}
