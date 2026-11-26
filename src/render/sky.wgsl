struct SkyUniform {
    inv_view_proj: mat4x4<f32>,
    sun_dir_time: vec4<f32>,
    zenith_color: vec4<f32>,
    horizon_color: vec4<f32>,
    ground_color: vec4<f32>,
    sun_color: vec4<f32>,
    cloud_color: vec4<f32>,
    params: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> sky: SkyUniform;

struct SkyOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn vs_sky(@builtin(vertex_index) vid: u32) -> SkyOut {
    var out: SkyOut;
    let x = f32((vid << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vid & 2u) * 2.0 - 1.0;
    out.pos = vec4<f32>(x, y, 1.0, 1.0);
    out.ndc = vec2<f32>(x, y);
    return out;
}

fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 = p3 + dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm_clouds(uv: vec2<f32>, octaves: i32) -> f32 {
    var p = uv;
    var sum = 0.0;
    var amp = 0.52;
    var norm = 0.0;
    let rot = mat2x2<f32>(0.80, -0.60, 0.60, 0.80);
    for (var i = 0; i < octaves; i = i + 1) {
        sum = sum + amp * value_noise(p);
        norm = norm + amp;
        p = (rot * p) * 2.04 + vec2<f32>(13.7, 7.1);
        amp = amp * 0.50;
    }
    return select(0.0, sum / norm, norm > 0.0);
}

fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    return clamp(
        (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
}

@fragment
fn fs_sky(in: SkyOut) -> @location(0) vec4<f32> {
    let p_near_h = sky.inv_view_proj * vec4<f32>(in.ndc, 0.0, 1.0);
    let p_far_h = sky.inv_view_proj * vec4<f32>(in.ndc, 0.95, 1.0);
    let p_near = p_near_h.xyz / p_near_h.w;
    let p_far = p_far_h.xyz / p_far_h.w;
    let dir = normalize(p_far - p_near);

    let sun_dir = normalize(sky.sun_dir_time.xyz);
    let time = sky.sun_dir_time.w;
    let zenith = sky.zenith_color.rgb;
    let horizon = sky.horizon_color.rgb;
    let ground = sky.ground_color.rgb;
    let sunset_factor = sky.horizon_color.w;
    let night_factor = sky.ground_color.w;
    let octaves = i32(sky.sun_color.w + 0.5);

    // 3-tier Zenith -> Horizon -> Ground atmospheric gradient
    let up_t = pow(clamp(dir.y, 0.0, 1.0), 0.58);
    let down_t = pow(clamp(-dir.y, 0.0, 1.0), 0.52);
    var col = mix(horizon, zenith, up_t);
    col = mix(col, ground, down_t);

    // Subtle twilight horizon warmth towards the sun azimuth
    let dir_xz = vec2<f32>(dir.x, dir.z);
    let sun_xz = vec2<f32>(sun_dir.x, sun_dir.z);
    let dir_xz_dir = select(vec2<f32>(1.0, 0.0), normalize(dir_xz), dot(dir_xz, dir_xz) > 1e-6);
    let sun_xz_dir = select(vec2<f32>(1.0, 0.0), normalize(sun_xz), dot(sun_xz, sun_xz) > 1e-6);
    let horiz_az = pow(max(dot(dir_xz_dir, sun_xz_dir), 1e-4), 2.5);
    let horiz_band = exp(-abs(dir.y) * 9.0) * sunset_factor * horiz_az;
    col = col + sky.sun_color.rgb * (0.20 * horiz_band);

    // Sun disc & soft atmospheric Mie scattering halo
    let cos_sun = dot(dir, sun_dir);
    let above_hor = smoothstep(-0.08, 0.04, dir.y);
    let sun_vis = smoothstep(-0.18, 0.12, sun_dir.y);
    let sun_disc = smoothstep(0.9986, 0.9993, cos_sun) * above_hor * sun_vis;
    let mie_halo = (pow(max(cos_sun, 1e-4), 42.0) * 0.45 + pow(max(cos_sun, 1e-4), 6.0) * 0.18) * above_hor * sun_vis;
    col = col + sky.sun_color.rgb * (mie_halo + sun_disc * 2.6);

    // Moon disc & cool silver night halo opposite the sun
    let cos_moon = dot(dir, -sun_dir);
    let moon_vis = smoothstep(-0.05, 0.10, dir.y) * night_factor;
    let moon_disc = smoothstep(0.9989, 0.9994, cos_moon) * moon_vis;
    let moon_halo = pow(max(cos_moon, 1e-4), 26.0) * 0.16 * moon_vis;
    col = col + vec3<f32>(0.76, 0.84, 1.0) * (moon_halo + moon_disc * 0.85);

    // Twinkling stars in upper night hemisphere
    if (night_factor > 0.02 && dir.y > 0.02) {
        let star_pos = dir * 185.0;
        let star_cell = floor(star_pos);
        let star_f = fract(star_pos) - 0.5;
        let star_seed = hash31(star_cell);
        if (star_seed > 0.985) {
            let radial = clamp(1.0 - length(star_f) * 2.2, 0.0, 1.0);
            let twinkle = 0.65 + 0.35 * sin(time * 2.6 + star_seed * 61.7);
            let star_fade = smoothstep(0.02, 0.28, dir.y) * night_factor;
            col = col + vec3<f32>(0.86, 0.91, 1.0) * (radial * radial * twinkle * star_fade * 1.15);
        }
    }

    // Animated drifting procedural FBM clouds (2-octaves Performance, 4-octaves Balanced/Cinematic)
    if (octaves > 0 && dir.y > 0.006) {
        let dome_y = dir.y + 0.14;
        let plane_uv = dir.xz / dome_y;
        let drift = vec2<f32>(time * 0.020, time * 0.007);
        let cloud_n = fbm_clouds(plane_uv * 0.72 + drift, octaves);
        let coverage = sky.cloud_color.w;
        let density = smoothstep(coverage, coverage + 0.30, cloud_n);
        let hor_fade = smoothstep(0.015, 0.24, dir.y);
        let cloud_alpha = density * hor_fade * 0.88;

        if (cloud_alpha > 0.002) {
            let forward_lit = pow(max(cos_sun, 1e-4), 8.0);
            let silver_lining = clamp((1.0 - density) * 1.5, 0.0, 1.0) * forward_lit;
            let shading = 0.82 + 0.18 * smoothstep(coverage, coverage + 0.35, cloud_n);
            let lit_cloud = sky.cloud_color.rgb * shading
                + sky.sun_color.rgb * (silver_lining * 0.65 + forward_lit * 0.18);
            col = mix(col, lit_cloud, cloud_alpha);
        }
    }

    // Match Shader's ACES curve when Warm Sun & Atmosphere is on without HDR Bloom
    if (sky.params.x > 0.5 && sky.params.y < 0.5) {
        col = aces_filmic(col);
    }

    return vec4<f32>(col, 1.0);
}
