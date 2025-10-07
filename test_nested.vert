#version 450

void main() {
    int x = gl_VertexIndex;
    int result;
    
    if (x == 0) {
        result = 10;
    } else if (x == 1) {
        result = 20;
    } else {
        result = 30;
    }
    
    gl_Position = vec4(float(result), 0.0, 0.0, 1.0);
}