use nom::{
    IResult,
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{char, digit1, hex_digit1, multispace0, multispace1},
    combinator::{map, map_res, opt, recognize, value as nom_value},
    multi::{many0, separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, separated_pair, terminated, tuple},
};

use crate::error::{ParseError, Result};
use crate::ir::*;

type ParseResult<'a, T> = IResult<&'a str, T>;

// Whitespace and comments
fn ws<'a, F, O>(f: F) -> impl FnMut(&'a str) -> ParseResult<'a, O>
where
    F: FnMut(&'a str) -> ParseResult<'a, O>,
{
    delimited(multispace0, f, multispace0)
}

fn comment(input: &str) -> ParseResult<'_, &str> {
    preceded(char(';'), take_while(|c| c != '\n'))(input)
}

fn ws_with_comments(input: &str) -> ParseResult<'_, &str> {
    recognize(many0(alt((multispace1, comment))))(input)
}

// Identifiers
fn identifier(input: &str) -> ParseResult<'_, String> {
    map(
        recognize(pair(
            alt((tag("_"), take_while1(|c: char| c.is_alphabetic()))),
            take_while(|c: char| c.is_alphanumeric() || c == '_'),
        )),
        |s: &str| s.to_string(),
    )(input)
}

fn register_name(input: &str) -> ParseResult<'_, String> {
    preceded(char('%'), identifier)(input)
}

fn global_name(input: &str) -> ParseResult<'_, String> {
    preceded(char('@'), identifier)(input)
}

fn label_name(input: &str) -> ParseResult<'_, String> {
    identifier(input)
}

// Types
fn scalar_type(input: &str) -> ParseResult<'_, Type> {
    alt((
        nom_value(Type::I8, tag("i8")),
        nom_value(Type::I16, tag("i16")),
        nom_value(Type::I32, tag("i32")),
        nom_value(Type::I64, tag("i64")),
        nom_value(Type::U8, tag("u8")),
        nom_value(Type::U16, tag("u16")),
        nom_value(Type::U32, tag("u32")),
        nom_value(Type::U64, tag("u64")),
        nom_value(Type::F16, tag("f16")),
        nom_value(Type::F32, tag("f32")),
        nom_value(Type::F64, tag("f64")),
    ))(input)
}

fn address_space(input: &str) -> ParseResult<'_, AddressSpace> {
    alt((
        nom_value(AddressSpace::Generic, tag("generic")),
        nom_value(AddressSpace::Global, tag("global")),
        nom_value(AddressSpace::Shared, tag("shared")),
        nom_value(AddressSpace::Local, tag("local")),
        nom_value(AddressSpace::Const, tag("const")),
    ))(input)
}

fn pointer_type(input: &str) -> ParseResult<'_, PointerType> {
    map(
        tuple((
            tag("ptr["),
            ws(address_space),
            tag("]<"),
            ws(type_parser),
            char('>'),
        )),
        |(_, space, _, pointee, _)| PointerType {
            address_space: space,
            pointee: Box::new(pointee),
        },
    )(input)
}

fn type_parser(input: &str) -> ParseResult<'_, Type> {
    alt((map(pointer_type, |pt| Type::Pointer(Box::new(pt))), scalar_type))(input)
}

// Literals and Constants
fn parse_i32_literal(input: &str) -> ParseResult<'_, i32> {
    map_res(recognize(pair(opt(char('-')), digit1)), |s: &str| {
        s.parse::<i32>()
    })(input)
}

fn parse_u32_literal(input: &str) -> ParseResult<'_, u32> {
    alt((
        map_res(preceded(tag("0x"), hex_digit1), |s: &str| {
            u32::from_str_radix(s, 16)
        }),
        map_res(digit1, |s: &str| s.parse::<u32>()),
    ))(input)
}

fn parse_u64_literal(input: &str) -> ParseResult<'_, u64> {
    alt((
        map_res(preceded(tag("0x"), hex_digit1), |s: &str| {
            u64::from_str_radix(s, 16)
        }),
        map_res(digit1, |s: &str| s.parse::<u64>()),
    ))(input)
}

fn parse_f32_literal(input: &str) -> ParseResult<'_, f32> {
    map_res(
        recognize(tuple((opt(char('-')), digit1, opt(pair(char('.'), digit1))))),
        |s: &str| s.parse::<f32>(),
    )(input)
}

fn hex_literal(input: &str) -> ParseResult<'_, u64> {
    preceded(
        tag("0x"),
        map_res(hex_digit1, |s: &str| u64::from_str_radix(s, 16)),
    )(input)
}

// Literals with type suffix
fn int_constant(input: &str) -> ParseResult<'_, Constant> {
    alt((
        map(
            terminated(parse_i32_literal, alt((tag("i32"), tag("i")))),
            Constant::I32,
        ),
        map(terminated(parse_i32_literal, tag("i64")), |v| {
            Constant::I64(v as i64)
        }),
        map(terminated(parse_i32_literal, tag("i16")), |v| {
            Constant::I16(v as i16)
        }),
        map(terminated(parse_i32_literal, tag("i8")), |v| {
            Constant::I8(v as i8)
        }),
    ))(input)
}

fn uint_constant(input: &str) -> ParseResult<'_, Constant> {
    alt((
        map(
            terminated(parse_u32_literal, alt((tag("u32"), tag("u")))),
            Constant::U32,
        ),
        map(terminated(parse_u64_literal, tag("u64")), Constant::U64),
        map(terminated(parse_u32_literal, tag("u16")), |v| {
            Constant::U16(v as u16)
        }),
        map(terminated(parse_u32_literal, tag("u8")), |v| {
            Constant::U8(v as u8)
        }),
    ))(input)
}

fn float_constant(input: &str) -> ParseResult<'_, Constant> {
    alt((
        map(
            terminated(parse_f32_literal, alt((tag("f32"), tag("f")))),
            Constant::F32,
        ),
        map(terminated(parse_f32_literal, tag("f64")), |v| {
            Constant::F64(v as f64)
        }),
    ))(input)
}

fn constant(input: &str) -> ParseResult<'_, Constant> {
    alt((int_constant, uint_constant, float_constant))(input)
}

// Values
fn parse_value(input: &str) -> ParseResult<'_, Value> {
    alt((
        map(constant, Value::Constant),
        map(register_name, Value::Register),
        map(global_name, Value::Global),
    ))(input)
}

// Memory ordering and scope
fn memory_ordering(input: &str) -> ParseResult<'_, MemoryOrdering> {
    alt((
        nom_value(MemoryOrdering::Relaxed, tag("relaxed")),
        nom_value(MemoryOrdering::Acquire, tag("acquire")),
        nom_value(MemoryOrdering::Release, tag("release")),
        nom_value(MemoryOrdering::AcqRel, tag("acq_rel")),
        nom_value(MemoryOrdering::SeqCst, tag("seq_cst")),
    ))(input)
}

fn memory_scope(input: &str) -> ParseResult<'_, MemoryScope> {
    alt((
        nom_value(MemoryScope::Invocation, tag("invocation")),
        nom_value(MemoryScope::Subgroup, tag("subgroup")),
        nom_value(MemoryScope::Workgroup, tag("workgroup")),
        nom_value(MemoryScope::Device, tag("device")),
        nom_value(MemoryScope::System, tag("system")),
    ))(input)
}

fn atomic_op(input: &str) -> ParseResult<'_, AtomicOp> {
    alt((
        nom_value(AtomicOp::Add, tag("add")),
        nom_value(AtomicOp::Sub, tag("sub")),
        nom_value(AtomicOp::MinS, tag("min_s")),
        nom_value(AtomicOp::MinU, tag("min_u")),
        nom_value(AtomicOp::MaxS, tag("max_s")),
        nom_value(AtomicOp::MaxU, tag("max_u")),
        nom_value(AtomicOp::And, tag("and")),
        nom_value(AtomicOp::Or, tag("or")),
        nom_value(AtomicOp::Xor, tag("xor")),
        nom_value(AtomicOp::Exchange, tag("exchange")),
        nom_value(AtomicOp::IncWrap, tag("inc_wrap")),
        nom_value(AtomicOp::DecWrap, tag("dec_wrap")),
        nom_value(AtomicOp::FAdd, tag("fadd")),
        nom_value(AtomicOp::FMin, tag("fmin")),
        nom_value(AtomicOp::FMax, tag("fmax")),
        nom_value(AtomicOp::FlagTestAndSet, tag("flag_test_and_set")),
        nom_value(AtomicOp::FlagClear, tag("flag_clear")),
    ))(input)
}

// Operations
fn binary_op(
    name: &'static str,
    f: impl Fn(Value, Value) -> Operation,
) -> impl FnMut(&str) -> ParseResult<'_, Operation> {
    move |input| {
        map(
            preceded(
                tag(name),
                tuple((ws(parse_value), preceded(ws(char(',')), ws(parse_value)))),
            ),
            |(a, b)| f(a, b),
        )(input)
    }
}

fn unary_op(
    name: &'static str,
    f: impl Fn(Value) -> Operation,
) -> impl FnMut(&str) -> ParseResult<'_, Operation> {
    move |input| map(preceded(tag(name), ws(parse_value)), |v| f(v))(input)
}

fn basic_ops(input: &str) -> ParseResult<'_, Operation> {
    alt((
        // Moves and constants
        unary_op("mov", Operation::Mov),
        map(preceded(tag("iconst"), ws(constant)), Operation::IConst),
        map(preceded(tag("uconst"), ws(constant)), Operation::UConst),
        map(preceded(tag("fconst"), ws(constant)), Operation::FConst),
        // Arithmetic
        binary_op("add", Operation::Add),
        binary_op("sub", Operation::Sub),
        binary_op("mul", Operation::Mul),
        // Division and remainder - support both signed/unsigned and generic forms
        binary_op("udiv", Operation::Div), // Unsigned division
        binary_op("sdiv", Operation::Div), // Signed division
        binary_op("urem", Operation::Rem), // Unsigned remainder
        binary_op("srem", Operation::Rem), // Signed remainder
        binary_op("div", Operation::Div),  // Generic division
        binary_op("rem", Operation::Rem),  // Generic remainder
        unary_op("neg", Operation::Neg),
        // Bitwise
        binary_op("and", Operation::And),
        binary_op("or", Operation::Or),
        binary_op("xor", Operation::Xor),
        unary_op("not", Operation::Not),
        binary_op("shl", Operation::Shl),
        binary_op("shr", Operation::Shr),
    ))(input)
}

fn comparison_ops(input: &str) -> ParseResult<'_, Operation> {
    alt((
        // Comparisons - signed
        map(
            preceded(
                tag("icmp.eq"),
                tuple((ws(parse_value), preceded(ws(char(',')), ws(parse_value)))),
            ),
            |(a, b)| Operation::ICmpEq(a, b),
        ),
        map(
            preceded(
                tag("icmp.ne"),
                tuple((ws(parse_value), preceded(ws(char(',')), ws(parse_value)))),
            ),
            |(a, b)| Operation::ICmpNe(a, b),
        ),
        map(
            preceded(
                tag("icmp.lt"),
                tuple((ws(parse_value), preceded(ws(char(',')), ws(parse_value)))),
            ),
            |(a, b)| Operation::ICmpLt(a, b),
        ),
        map(
            preceded(
                tag("icmp.le"),
                tuple((ws(parse_value), preceded(ws(char(',')), ws(parse_value)))),
            ),
            |(a, b)| Operation::ICmpLe(a, b),
        ),
        map(
            preceded(
                tag("icmp.gt"),
                tuple((ws(parse_value), preceded(ws(char(',')), ws(parse_value)))),
            ),
            |(a, b)| Operation::ICmpGt(a, b),
        ),
        map(
            preceded(
                tag("icmp.ge"),
                tuple((ws(parse_value), preceded(ws(char(',')), ws(parse_value)))),
            ),
            |(a, b)| Operation::ICmpGe(a, b),
        ),
        // Comparisons - unsigned
        map(
            preceded(
                tag("ucmp.eq"),
                tuple((ws(parse_value), preceded(ws(char(',')), ws(parse_value)))),
            ),
            |(a, b)| Operation::UCmpEq(a, b),
        ),
        map(
            preceded(
                tag("ucmp.ne"),
                tuple((ws(parse_value), preceded(ws(char(',')), ws(parse_value)))),
            ),
            |(a, b)| Operation::UCmpNe(a, b),
        ),
        map(
            preceded(
                tag("ucmp.lt"),
                tuple((ws(parse_value), preceded(ws(char(',')), ws(parse_value)))),
            ),
            |(a, b)| Operation::UCmpLt(a, b),
        ),
        map(
            preceded(
                tag("ucmp.le"),
                tuple((ws(parse_value), preceded(ws(char(',')), ws(parse_value)))),
            ),
            |(a, b)| Operation::UCmpLe(a, b),
        ),
        map(
            preceded(
                tag("ucmp.gt"),
                tuple((ws(parse_value), preceded(ws(char(',')), ws(parse_value)))),
            ),
            |(a, b)| Operation::UCmpGt(a, b),
        ),
        map(
            preceded(
                tag("ucmp.ge"),
                tuple((ws(parse_value), preceded(ws(char(',')), ws(parse_value)))),
            ),
            |(a, b)| Operation::UCmpGe(a, b),
        ),
        // Float comparisons - ordered
        map(
            preceded(
                tag("fcmp.oeq"),
                tuple((ws(parse_value), preceded(ws(char(',')), ws(parse_value)))),
            ),
            |(a, b)| Operation::FCmpOEq(a, b),
        ),
        map(
            preceded(
                tag("fcmp.olt"),
                tuple((ws(parse_value), preceded(ws(char(',')), ws(parse_value)))),
            ),
            |(a, b)| Operation::FCmpOLt(a, b),
        ),
        map(
            preceded(
                tag("fcmp.ole"),
                tuple((ws(parse_value), preceded(ws(char(',')), ws(parse_value)))),
            ),
            |(a, b)| Operation::FCmpOLe(a, b),
        ),
        map(
            preceded(
                tag("fcmp.ogt"),
                tuple((ws(parse_value), preceded(ws(char(',')), ws(parse_value)))),
            ),
            |(a, b)| Operation::FCmpOGt(a, b),
        ),
        map(
            preceded(
                tag("fcmp.oge"),
                tuple((ws(parse_value), preceded(ws(char(',')), ws(parse_value)))),
            ),
            |(a, b)| Operation::FCmpOGe(a, b),
        ),
    ))(input)
}

fn memory_and_convert_ops(input: &str) -> ParseResult<'_, Operation> {
    alt((
        // Select
        map(
            preceded(
                tag("select"),
                tuple((
                    ws(parse_value),
                    preceded(ws(char(',')), ws(parse_value)),
                    preceded(ws(char(',')), ws(parse_value)),
                )),
            ),
            |(cond, t, f)| Operation::Select(cond, t, f),
        ),
        // Memory
        unary_op("load", Operation::Load),
        map(
            preceded(
                tag("store"),
                tuple((ws(parse_value), preceded(ws(char(',')), ws(parse_value)))),
            ),
            |(ptr, val)| Operation::Store(ptr, val),
        ),
        // GEP
        map(
            tuple((
                tag("gep"),
                ws(parse_value),
                preceded(ws(char(',')), ws(parse_value)),
                preceded(
                    tuple((ws(char(',')), ws(tag("stride")), ws(char('=')))),
                    parse_u32_literal,
                ),
            )),
            |(_, base, index, stride)| Operation::Gep { base, index, stride },
        ),
        // Type conversions
        unary_op("bitcast", Operation::Bitcast),
        unary_op("trunc", Operation::Trunc),
        unary_op("zext", Operation::Zext),
        unary_op("sext", Operation::Sext),
        unary_op("fptosi", Operation::FpToSi),
        unary_op("fptoui", Operation::FpToUi),
        unary_op("sitofp", Operation::SiToFp),
        unary_op("uitofp", Operation::UiToFp),
        unary_op("fpext", Operation::FpExt),
        unary_op("fptrunc", Operation::FpTrunc),
        // Function call
        map(
            tuple((
                tag("call"),
                ws(global_name),
                delimited(
                    ws(char('(')),
                    separated_list0(ws(char(',')), ws(parse_value)),
                    ws(char(')')),
                ),
            )),
            |(_, func, args)| Operation::Call { func, args },
        ),
    ))(input)
}

fn atomic_ops(input: &str) -> ParseResult<'_, Operation> {
    alt((
        parse_atomic_load,
        parse_atomic_store,
        parse_atomic_rmw,
        parse_atomic_cmpxchg,
    ))(input)
}

fn operation(input: &str) -> ParseResult<'_, Operation> {
    alt((
        basic_ops,
        comparison_ops,
        memory_and_convert_ops,
        atomic_ops,
        parse_phi,
    ))(input)
}

fn parse_atomic_load(input: &str) -> ParseResult<'_, Operation> {
    map(
        tuple((
            tag("atomic.load"),
            ws(parse_value),
            preceded(tuple((ws(tag("ordering")), ws(char('=')))), ws(memory_ordering)),
            preceded(tuple((ws(tag("scope")), ws(char('=')))), ws(memory_scope)),
        )),
        |(_, ptr, ordering, scope)| Operation::AtomicLoad { ptr, ordering, scope },
    )(input)
}

fn parse_atomic_store(input: &str) -> ParseResult<'_, Operation> {
    map(
        tuple((
            tag("atomic.store"),
            ws(parse_value),
            preceded(ws(char(',')), ws(parse_value)),
            preceded(tuple((ws(tag("ordering")), ws(char('=')))), ws(memory_ordering)),
            preceded(tuple((ws(tag("scope")), ws(char('=')))), ws(memory_scope)),
        )),
        |(_, ptr, value, ordering, scope)| Operation::AtomicStore {
            ptr,
            value,
            ordering,
            scope,
        },
    )(input)
}

fn parse_atomic_rmw(input: &str) -> ParseResult<'_, Operation> {
    map(
        tuple((
            tag("atomic.rmw"),
            ws(atomic_op),
            ws(parse_value),
            preceded(ws(char(',')), ws(parse_value)),
            preceded(tuple((ws(tag("ordering")), ws(char('=')))), ws(memory_ordering)),
            preceded(tuple((ws(tag("scope")), ws(char('=')))), ws(memory_scope)),
        )),
        |(_, op, ptr, value, ordering, scope)| Operation::AtomicRmw {
            op,
            ptr,
            value,
            ordering,
            scope,
        },
    )(input)
}

fn parse_atomic_cmpxchg(input: &str) -> ParseResult<'_, Operation> {
    map(
        tuple((
            tag("atomic.cmpxchg"),
            ws(parse_value),
            preceded(ws(char(',')), ws(parse_value)),
            preceded(ws(char(',')), ws(parse_value)),
            preceded(
                tuple((ws(tag("ordering_succ")), ws(char('=')))),
                ws(memory_ordering),
            ),
            preceded(
                tuple((ws(tag("ordering_fail")), ws(char('=')))),
                ws(memory_ordering),
            ),
            preceded(tuple((ws(tag("scope")), ws(char('=')))), ws(memory_scope)),
        )),
        |(_, ptr, expected, desired, ordering_succ, ordering_fail, scope)| Operation::AtomicCmpXchg {
            ptr,
            expected,
            desired,
            ordering_succ,
            ordering_fail,
            scope,
        },
    )(input)
}

fn parse_phi(input: &str) -> ParseResult<'_, Operation> {
    map(
        tuple((
            tag("phi"),
            ws(type_parser),
            separated_list1(
                ws(char(',')),
                delimited(
                    ws(char('[')),
                    separated_pair(ws(parse_value), ws(char(',')), ws(label_name)),
                    ws(char(']')),
                ),
            ),
        )),
        |(_, ty, incoming)| Operation::Phi { ty, incoming },
    )(input)
}

// Instructions
fn instruction(input: &str) -> ParseResult<'_, Instruction> {
    map(
        tuple((opt(terminated(register_name, ws(char('=')))), ws(operation))),
        |(result, op)| Instruction { result, op },
    )(input)
}

// Terminators
fn terminator(input: &str) -> ParseResult<'_, Terminator> {
    alt((
        map(
            tuple((
                tag("br_if"),
                ws(parse_value),
                preceded(ws(char(',')), ws(label_name)),
                preceded(ws(char(',')), ws(label_name)),
            )),
            |(_, cond, true_label, false_label)| Terminator::BrIf {
                cond,
                true_label,
                false_label,
            },
        ),
        map(preceded(tag("br"), ws(label_name)), Terminator::Br),
        map(preceded(tag("ret"), opt(ws(parse_value))), Terminator::Ret),
    ))(input)
}

// Basic blocks
fn basic_block(input: &str) -> ParseResult<'_, BasicBlock> {
    map(
        tuple((
            terminated(ws(label_name), ws(char(':'))),
            many0(preceded(
                opt(ws_with_comments),
                terminated(instruction, opt(ws_with_comments)),
            )),
            preceded(opt(ws_with_comments), ws(terminator)),
        )),
        |(label, instructions, terminator)| BasicBlock {
            label,
            instructions,
            terminator,
        },
    )(input)
}

// Function attributes
fn function_attr(input: &str) -> ParseResult<'_, FunctionAttr> {
    alt((
        nom_value(FunctionAttr::Kernel, tag("kernel")),
        nom_value(FunctionAttr::Inline, tag("inline")),
        nom_value(FunctionAttr::NoInline, tag("noinline")),
    ))(input)
}

// Parameters
fn param(input: &str) -> ParseResult<'_, Param> {
    map(
        separated_pair(register_name, ws(char(':')), ws(type_parser)),
        |(name, ty)| Param { name, ty },
    )(input)
}

fn param_list(input: &str) -> ParseResult<'_, Vec<Param>> {
    separated_list0(ws(char(',')), ws(param))(input)
}

// Return type
fn return_type(input: &str) -> ParseResult<'_, ReturnType> {
    alt((
        nom_value(ReturnType::Void, tag("void")),
        map(type_parser, ReturnType::Type),
    ))(input)
}

// Function
fn function(input: &str) -> ParseResult<'_, Function> {
    map(
        tuple((
            tag("func"),
            many0(ws(function_attr)),
            ws(global_name),
            delimited(ws(char('(')), param_list, ws(char(')'))),
            preceded(tuple((ws(tag("->")), ws(multispace0))), ws(return_type)),
            delimited(
                ws(char('{')),
                many0(preceded(opt(ws_with_comments), basic_block)),
                ws(char('}')),
            ),
        )),
        |(_, attributes, name, params, return_type, blocks)| Function {
            name,
            attributes,
            params,
            return_type,
            blocks,
        },
    )(input)
}

// Globals
fn initializer(input: &str) -> ParseResult<'_, Initializer> {
    map(
        preceded(
            tuple((tag("addr"), ws(char('(')))),
            terminated(hex_literal, ws(char(')'))),
        ),
        Initializer::Addr,
    )(input)
}

fn global(input: &str) -> ParseResult<'_, Global> {
    map(
        tuple((
            tag("global"),
            ws(global_name),
            preceded(ws(char(':')), ws(pointer_type)),
            opt(preceded(ws(char('=')), ws(initializer))),
        )),
        |(_, name, ty, initializer)| Global {
            name,
            ty,
            initializer,
        },
    )(input)
}

// Module
pub fn module(input: &str) -> ParseResult<'_, Module> {
    map(
        preceded(
            opt(ws_with_comments),
            many0(preceded(
                opt(ws_with_comments),
                alt((
                    map(global, |g| (Some(g), None)),
                    map(function, |f| (None, Some(f))),
                )),
            )),
        ),
        |items| {
            let mut globals = Vec::new();
            let mut functions = Vec::new();
            for (g, f) in items {
                if let Some(g) = g {
                    globals.push(g);
                }
                if let Some(f) = f {
                    functions.push(f);
                }
            }
            Module { globals, functions }
        },
    )(input)
}

// Public API
pub fn parse_module(input: &str) -> Result<Module> {
    match module(input) {
        Ok((remaining, module)) => {
            // Check that all input was consumed (only whitespace/comments allowed)
            let (leftover, _) = opt(ws_with_comments)(remaining).unwrap_or((remaining, None));
            if !leftover.is_empty() {
                return Err(ParseError::NomError(format!(
                    "Unexpected content after parsing module. Remaining input ({} bytes): {:?}...",
                    leftover.len(),
                    &leftover[..100.min(leftover.len())]
                )));
            }
            Ok(module)
        }
        Err(e) => Err(ParseError::NomError(e.to_string())),
    }
}

pub fn parse_function(input: &str) -> Result<Function> {
    match preceded(opt(ws_with_comments), function)(input) {
        Ok((remaining, func)) => {
            // Check that all input was consumed (only whitespace/comments allowed)
            let (leftover, _) = opt(ws_with_comments)(remaining).unwrap_or((remaining, None));
            if !leftover.is_empty() {
                return Err(ParseError::NomError(format!(
                    "Unexpected content after parsing function. Remaining input ({} bytes): {:?}...",
                    leftover.len(),
                    &leftover[..100.min(leftover.len())]
                )));
            }
            Ok(func)
        }
        Err(e) => Err(ParseError::NomError(e.to_string())),
    }
}
