struct ShadowUniform {
    light_view_proj: mat4x4<f32>,
    params: vec4<f32>, // x=enabled, y=bias, z=strength, w=texel_size
}

@group(0) @binding(0)
var<uniform> shadow_u: ShadowUniform;

struct ShadowVertexIn {
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) packed: u32,
    @location(3) tint: vec4<f32>,
}

@vertex
fn vs_shadow(in: ShadowVertexIn) -> @builtin(position) vec4<f32> {
    return shadow_u.light_view_proj * vec4<f32>(in.pos, 1.0);
}
