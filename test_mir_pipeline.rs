use wyn_core::{lexer, parser, type_checker, mirize, lowering};

fn main() {
    let source = r#"
#[vertex] def vertex_main(#[builtin(vertex_index)] vertex_index: i32): (#[builtin(position)] vec4, #[location(1)] vec3) =
  let position: vec4 = if vertex_index == 0 then
      vec4 -0.8f32 -0.8f32 0.0f32 1.0f32
    else if vertex_index == 1 then
      vec4 0.8f32 -0.8f32 0.0f32 1.0f32
    else
      vec4 0.0f32 0.8f32 0.0f32 1.0f32 in
  let color: vec3 = vec3 1.0f32 0.0f32 0.0f32 in
  (position, color)

#[fragment] def fragment_main(): #[location(0)] vec4 =
  vec4 1.0f32 0.0f32 0.0f32 1.0f32
"#;

    println!("=== Lexing ===");
    let tokens = lexer::tokenize(source).unwrap();
    println!("Tokens: {} total", tokens.len());

    println!("\n=== Parsing ===");
    let mut parser = parser::Parser::new(tokens);
    let program = parser.parse().unwrap();
    println!("Parsed {} declarations", program.declarations.len());

    println!("\n=== Type Checking ===");
    let mut type_checker = type_checker::TypeChecker::new();
    type_checker.load_builtins().unwrap();
    let type_table = type_checker.check_program(&program).unwrap();
    println!("Type table has {} entries", type_table.len());

    println!("\n=== MIR Generation (Mirize) ===");
    let mirize = mirize::Mirize::new(type_table);
    let mir_module = mirize.mirize_program(&program).unwrap();
    println!("Generated MIR with {} functions", mir_module.functions.len());
    for func in &mir_module.functions {
        println!("  - {} ({} params, {} blocks)", func.name, func.params.len(), func.blocks.len());
    }

    println!("\n=== SPIR-V Lowering ===");
    let lowering = lowering::Lowering::new();
    let spirv_bytes = lowering.lower_module(&mir_module).unwrap();
    println!("Generated SPIR-V: {} words ({} bytes)", spirv_bytes.len(), spirv_bytes.len() * 4);

    // Write SPIR-V to file
    let spirv_path = "test_output.spv";
    use std::io::Write;
    let bytes: Vec<u8> = spirv_bytes.iter().flat_map(|word| word.to_le_bytes()).collect();
    std::fs::File::create(spirv_path)
        .unwrap()
        .write_all(&bytes)
        .unwrap();
    println!("Wrote SPIR-V to {}", spirv_path);

    println!("\n=== Validation ===");
    println!("Run: spirv-val {}", spirv_path);
    println!("Run: spirv-dis {} -o test_output.spvasm", spirv_path);
}
