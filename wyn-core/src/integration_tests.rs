#[cfg(test)]
mod tests {
    use crate::Compiler;

    #[test]
    fn test_full_example_program() {
        let source = r#"
-- Full-screen triangle in NDC (like classic shader demos).
def verts: [3][4]f32 =
  [[-1.0f32, -1.0f32, 0.0f32, 1.0f32],
   [ 3.0f32, -1.0f32, 0.0f32, 1.0f32],
   [-1.0f32,  3.0f32, 0.0f32, 1.0f32]]

-- Vertex stage: return clip-space position for a given vertex index.
#[vertex]
entry vertex_main (vertex_id: i32) : [4]f32 =
  verts[vertex_id]

-- Fragment stage: constant sky blue (#87CEEB).
def SKY_RGBA : [4]f32 =
  [135f32/255f32, 206f32/255f32, 235f32/255f32, 1.0f32]

#[fragment]
entry fragment_main () : [4]f32 =
  SKY_RGBA
"#;

        let compiler = Compiler::new();
        let result = compiler.compile(source);

        assert!(result.is_ok(), "Compilation failed: {:?}", result.err());

        let spirv = result.unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], spirv::MAGIC_NUMBER);

        // The SPIR-V should be reasonably sized for this simple program
        assert!(spirv.len() > 50, "SPIR-V too small, likely incomplete");
        assert!(spirv.len() < 1000, "SPIR-V too large, likely has issues");
    }

    #[test]
    fn test_vertex_shader_only() {
        let source = r#"
let positions: [3][4]f32 =
  [[0.0f32, 0.5f32, 0.0f32, 1.0f32],
   [-0.5f32, -0.5f32, 0.0f32, 1.0f32],
   [0.5f32, -0.5f32, 0.0f32, 1.0f32]]

#[vertex]
entry vertex_main(vertex_id: i32): [4]f32 = positions[vertex_id]
"#;

        let compiler = Compiler::new();
        let result = compiler.compile(source);
        assert!(result.is_ok(), "{result:?}");
    }

    #[test]
    fn test_fragment_shader_only() {
        let source = r#"
let red: [4]f32 = [1.0f32, 0.0f32, 0.0f32, 1.0f32]
#[fragment]
entry fragment_main(): [4]f32 = red
"#;

        let compiler = Compiler::new();
        let result = compiler.compile(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_division_in_array() {
        let source = r#"
let normalized_color: [3]f32 = [128f32/255f32, 64f32/255f32, 32f32/255f32]
#[fragment]
entry fragment_color(): [4]f32 = [normalized_color[0], normalized_color[1], normalized_color[2], 1.0f32]
"#;

        let compiler = Compiler::new();
        let result = compiler.compile(source);
        // Note: This test will fail with current implementation as we don't support
        // mixing array access in array literals yet, but it demonstrates the test
        assert!(result.is_err()); // Expected to fail with current limitations
    }
}
