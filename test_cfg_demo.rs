use wyn_core::cfg::CfgExtractor;
use wyn_core::lexer::tokenize;
use wyn_core::parser::Parser;

fn main() {
    let source = r#"
        entry main(x: i32): i32 = x + 1
    "#;
    
    // Parse the source
    let tokens = tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();
    
    // Extract CFG facts
    let mut buffer = Vec::new();
    let extractor = CfgExtractor::new(&mut buffer, true);
    extractor.extract_cfg(&program).unwrap();
    
    let output = String::from_utf8(buffer).unwrap();
    println!("=== CFG FACTS ===");
    println!("{}", output);
}