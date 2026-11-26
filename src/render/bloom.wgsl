struct BloomParams {
    texel_size: vec2<f32>,
    threshold: f32,
    strength: f32,
}

@group(0) @binding(0)
var<uniform> params: BloomParams;

@group(1) @binding(0)
var tex0: texture_2d<f32>;
@group(1) @binding(1)
var samp: sampler;
@group(1) @binding(2)
var tex1: texture_2d<f32>;

struct FsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> FsOut {
    var out: FsOut;
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_bright_downsample(in: FsOut) -> @location(0) vec4<f32> {
    let o = params.texel_size * 1.25;
    let c0 = textureSample(tex0, samp, in.uv + vec2<f32>(-o.x, -o.y)).rgb;
    let c1 = textureSample(tex0, samp, in.uv + vec2<f32>( o.x, -o.y)).rgb;
    let c2 = textureSample(tex0, samp, in.uv + vec2<f32>(-o.x,  o.y)).rgb;
    let c3 = textureSample(tex0, samp, in.uv + vec2<f32>( o.x,  o.y)).rgb;
    let avg = max((c0 + c1 + c2 + c3) * 0.25, vec3<f32>(0.0));
    let luma = dot(avg, vec3<f32>(0.2126, 0.7152, 0.0722));
    let knee = 0.20;
    let soft = clamp(luma - params.threshold + knee, 0.0, 2.0 * knee);
    let soft_contrib = (soft * soft) / max(4.0 * knee, 1e-4);
    let contrib = max(max(luma - params.threshold, soft_contrib), 0.0) / max(luma, 1e-4);
    return vec4<f32>(min(avg * contrib, vec3<f32>(6.0)), 1.0);
}

@fragment
fn fs_kawase_blur(in: FsOut) -> @location(0) vec4<f32> {
    let o = params.texel_size * params.strength;
    let c0 = textureSample(tex0, samp, in.uv + vec2<f32>(-o.x, -o.y)).rgb;
    let c1 = textureSample(tex0, samp, in.uv + vec2<f32>( o.x, -o.y)).rgb;
    let c2 = textureSample(tex0, samp, in.uv + vec2<f32>(-o.x,  o.y)).rgb;
    let c3 = textureSample(tex0, samp, in.uv + vec2<f32>( o.x,  o.y)).rgb;
    let center = textureSample(tex0, samp, in.uv).rgb;
    return vec4<f32>(center * 0.36 + (c0 + c1 + c2 + c3) * 0.16, 1.0);
}

fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_composite(in: FsOut) -> @location(0) vec4<f32> {
    let scene = textureSample(tex0, samp, in.uv).rgb;
    let bloom = textureSample(tex1, samp, in.uv).rgb;
    let hdr = max(scene + bloom * params.strength, vec3<f32>(0.0));
    return vec4<f32>(aces_filmic(hdr), 1.0);
}
