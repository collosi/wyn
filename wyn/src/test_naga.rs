fn main() {
    let wgsl = r#"
@vertex
fn main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    var result: f32;
    
    if (vertex_index == 0u) {
        result = 10.0;
    } else if (vertex_index == 1u) {
        result = 20.0;
    } else {
        result = 30.0;
    }
    
    return vec4<f32>(result, 0.0, 0.0, 1.0);
}
"#;

    println!("WGSL source:");
    println!("{}", wgsl);
    
    // For now, just print the WGSL
    // We'd need naga dependency to actually compile it
}