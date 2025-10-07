@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    var result: i32;
    
    if (x == 0u) {
        result = 10;
    } else if (x == 1u) {
        result = 20;
    } else {
        result = 30;
    }
}