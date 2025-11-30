use gasm::parse_function;

fn main() {
    let input = r#"
func @max_u32(%a: u32, %b: u32) -> u32 {
entry:
  %cmp = ucmp.ge %a, %b
  br_if %cmp, a_ge_b, otherwise

a_ge_b:
  ret %a

otherwise:
  ret %b
}
"#;
    match parse_function(input) {
        Ok(func) => {
            println!("Parsed successfully!");
            println!("{:#?}", func);
        }
        Err(e) => {
            println!("Parse error: {:?}", e);
        }
    }
}
