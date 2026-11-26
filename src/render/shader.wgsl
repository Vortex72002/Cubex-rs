struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    sun_dir_time: vec4<f32>,
    fog_color_dist: vec4<f32>,
    flags: vec4<f32>, // x=smooth_lighting, y=dynamic_shadows, z=waving_water, w=brightness
    hand_light_pos_intensity: vec4<f32>, // xyz=world pos, w=intensity (0..1)
    enhance_flags: vec4<f32>, // x=warm_sun_atmosphere, y=wind_sway_foliage, z=bloom_hdr_active, w=time_of_day
    sky_zenith_clouds: vec4<f32>, // rgb=sky zenith, w=cloud_octaves
    sky_sun_color_atmo: vec4<f32>, // rgb=sky sun color, w=procedural_sky_clouds
}

@group(0) @binding(0)
var<uniform> cam: CameraUniform;

@group(1) @binding(0)
var tex_array: texture_2d_array<f32>;
@group(1) @binding(1)
var tex_sampler: sampler;

struct ShadowUniform {
    light_view_proj: mat4x4<f32>,
    params: vec4<f32>, // x=enabled, y=bias, z=strength, w=texel_size
}

@group(2) @binding(0)
var<uniform> shadow_u: ShadowUniform;
@group(2) @binding(1)
var shadow_map: texture_depth_2d;
@group(2) @binding(2)
var shadow_sampler: sampler_comparison;

struct VertexIn {
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) packed: u32,
    @location(3) tint: vec4<f32>,
}

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) @interpolate(flat) layer: u32,
    @location(3) @interpolate(flat) cutout: u32,
    @location(4) light_ao: vec4<f32>, // x=dir_shade, y=sky, z=block, w=ao
    @location(5) tint: vec4<f32>,
    @location(6) @interpolate(flat) waving: u32,
    @location(7) @interpolate(flat) hurt_flash: u32,
    @location(8) @interpolate(flat) no_alpha_discard: u32,
    @location(9) @interpolate(flat) mat_flags: u32, // bit 0=specular, bit 1=emissive
    @location(10) @interpolate(flat) face_dir: u32,
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    let layer = in.packed & 0x7FFu;
    let face_dir = (in.packed >> 11u) & 0x7u;
    let sky = f32((in.packed >> 14u) & 0xFu) / 15.0;
    let block = f32((in.packed >> 18u) & 0xFu) / 15.0;
    let ao_q = f32((in.packed >> 22u) & 0x3u);
    let waving = (in.packed >> 24u) & 0x1u;
    let cutout = (in.packed >> 25u) & 0x1u;
    let hurt_flash = (in.packed >> 26u) & 0x1u;
    let no_alpha_discard = (in.packed >> 27u) & 0x1u;
    let is_specular = (in.packed >> 28u) & 0x1u;
    let is_emissive = (in.packed >> 29u) & 0x1u;

    var wpos = in.pos;
    if (waving == 1u) {
        let t = cam.sun_dir_time.w;
        if (cutout == 1u && cam.enhance_flags.y > 0.5) {
            // Wind-swaying grass, flowers, and leaves
            let primary = sin(t * 2.2 + wpos.x * 0.7 + wpos.z * 0.7);
            wpos.x += primary * 0.058;
            wpos.z += cos(t * 1.9 + wpos.z * 0.7 + wpos.x * 0.7) * 0.038;
            wpos.y += sin(t * 2.5 + wpos.x * 0.7 + wpos.z * 0.7) * 0.012;
        } else if (cutout == 0u && cam.flags.z > 0.5) {
            // Waving water surface
            wpos.y += sin(t * 2.4 + wpos.x * 0.95 + wpos.z * 0.75) * 0.036;
        }
    }

    var dir_shade = 1.0;
    if (face_dir == 2u) { dir_shade = 1.0; }       // +Y top
    else if (face_dir == 3u) { dir_shade = 0.56; } // -Y bottom
    else if (face_dir == 0u || face_dir == 1u) { dir_shade = 0.80; } // +/-X
    else { dir_shade = 0.88; }                     // +/-Z

    let ao = mix(1.0, 0.42 + (ao_q / 3.0) * 0.58, cam.flags.x);

    var out: VertexOut;
    out.clip_pos = cam.view_proj * vec4<f32>(wpos, 1.0);
    out.uv = in.uv;
    out.world_pos = wpos;
    out.layer = layer;
    out.cutout = cutout;
    out.light_ao = vec4<f32>(dir_shade, sky, block, ao);
    out.tint = in.tint;
    out.waving = waving;
    out.hurt_flash = hurt_flash;
    out.no_alpha_discard = no_alpha_discard;
    out.mat_flags = is_specular | (is_emissive << 1u);
    out.face_dir = face_dir;
    return out;
}

fn face_normal_from_dir(face_dir: u32) -> vec3<f32> {
    if (face_dir == 0u) { return vec3<f32>(1.0, 0.0, 0.0); }
    if (face_dir == 1u) { return vec3<f32>(-1.0, 0.0, 0.0); }
    if (face_dir == 2u) { return vec3<f32>(0.0, 1.0, 0.0); }
    if (face_dir == 3u) { return vec3<f32>(0.0, -1.0, 0.0); }
    if (face_dir == 4u) { return vec3<f32>(0.0, 0.0, 1.0); }
    return vec3<f32>(0.0, 0.0, -1.0);
}

fn sample_sun_shadow(world_pos: vec3<f32>, normal: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
    if (cam.flags.y < 0.5 || shadow_u.params.x < 0.5 || shadow_u.params.z <= 0.001) {
        return 1.0;
    }
    let ndotl = max(dot(normal, sun_dir), 0.0);
    let slope_bias = clamp(0.0012 * (1.0 - ndotl), 0.0, 0.0035);
    let offset_pos = world_pos + normal * 0.065;
    let lclip = shadow_u.light_view_proj * vec4<f32>(offset_pos, 1.0);
    if (lclip.w <= 0.0001) {
        return 1.0;
    }
    let ndc = lclip.xyz / lclip.w;
    let suv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let edge = max(abs(ndc.x), abs(ndc.y));
    let ref_z = clamp(ndc.z - (shadow_u.params.y + slope_bias), 0.0, 1.0);
    let ts = shadow_u.params.w;

    var sum = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let sample_uv = clamp(suv + vec2<f32>(f32(dx), f32(dy)) * ts, vec2<f32>(0.0), vec2<f32>(1.0));
            sum += textureSampleCompareLevel(shadow_map, shadow_sampler, sample_uv, ref_z);
        }
    }
    let pcf = sum / 9.0;
    let in_bounds = ndc.z >= 0.0 && ndc.z <= 1.0 && suv.x >= 0.0 && suv.x <= 1.0 && suv.y >= 0.0 && suv.y <= 1.0;
    let valid = select(0.0, 1.0, in_bounds);
    let fade = (1.0 - smoothstep(0.80, 0.98, edge)) * shadow_u.params.z * valid;
    return mix(1.0, pcf, fade);
}

fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn sample_voxel_texel(in: VertexOut) -> vec4<f32> {
    var sample_uv = in.uv;
    if (in.waving == 1u && in.cutout == 0u && cam.flags.z > 0.5) {
        let t = cam.sun_dir_time.w;
        sample_uv = sample_uv + vec2<f32>(sin(t * 1.1 + in.world_pos.z * 0.6) * 0.02, t * 0.07);
    }
    let uv_dx = dpdx(sample_uv);
    let uv_dy = dpdy(sample_uv);
    if (in.no_alpha_discard == 0u) {
        let tex_dim = textureDimensions(tex_array);
        if (tex_dim.x == 32u && tex_dim.y == 32u) {
            let half_tex = 0.5 / 32.0;
            sample_uv = clamp(fract(sample_uv), vec2<f32>(half_tex), vec2<f32>(1.0 - half_tex));
        }
    }
    return textureSampleGrad(tex_array, tex_sampler, sample_uv, i32(in.layer), uv_dx, uv_dy);
}

fn water_sky_reflection(
    in: VertexOut,
    base_rgb: vec3<f32>,
    base_alpha: f32,
    sky: f32,
    sun_strength: f32,
) -> vec4<f32> {
    let face_n = face_normal_from_dir(in.face_dir);
    var water_n = face_n;
    let t = cam.sun_dir_time.w;
    if (cam.flags.z > 0.5) {
        let wp = in.world_pos.xz;
        let wx = sin(wp.x * 2.1 + t * 2.3 + cos(wp.y * 1.4 - t * 1.7)) * 0.085;
        let wz = cos(wp.y * 2.3 - t * 2.0 + sin(wp.x * 1.6 + t * 1.5)) * 0.085;
        water_n = normalize(face_n + vec3<f32>(wx, 0.0, wz));
    }
    let to_surf = in.world_pos - cam.camera_pos.xyz;
    let inc = select(vec3<f32>(0.0, -1.0, 0.0), normalize(to_surf), dot(to_surf, to_surf) > 1e-5);
    let refl = reflect(inc, water_n);
    let refl_up = pow(clamp(abs(refl.y), 1e-4, 1.0), 0.58);
    var refl_col = mix(cam.fog_color_dist.rgb, cam.sky_zenith_clouds.rgb, refl_up);

    if (cam.sky_zenith_clouds.w > 0.5 && refl.y > 0.03) {
        let c_uv = refl.xz / (max(refl.y, 0.02) + 0.16) * 0.72 + vec2<f32>(t * 0.020, t * 0.007);
        let c_wave = sin(c_uv.x * 4.3 + cos(c_uv.y * 3.1)) * cos(c_uv.y * 4.7 - sin(c_uv.x * 2.9));
        let c_mask = smoothstep(0.28, 0.72, c_wave * 0.5 + 0.5) * smoothstep(0.03, 0.25, refl.y);
        refl_col = mix(refl_col, vec3<f32>(0.96, 0.98, 1.0) * max(0.25, sun_strength), c_mask * 0.34);
    }

    let sun_dir = normalize(cam.sun_dir_time.xyz);
    let sun_vis = smoothstep(-0.08, 0.15, cam.sun_dir_time.y);
    let glint = pow(max(dot(refl, sun_dir), 1e-4), 72.0) * 0.95 * sky * sun_vis;
    refl_col += cam.sky_sun_color_atmo.rgb * glint;

    let ndotv = clamp(dot(-inc, water_n), 0.0, 1.0);
    let depth_tint = mix(
        vec3<f32>(0.12, 0.56, 0.72),
        vec3<f32>(0.03, 0.16, 0.34),
        clamp(1.0 - ndotv * 0.78, 0.0, 1.0),
    );
    let tinted_base = mix(base_rgb, depth_tint * max(0.24, sun_strength), 0.46);
    let fresnel = 0.07 + 0.78 * pow(max(1.0 - ndotv, 1e-4), 4.0);
    let weight = fresnel * smoothstep(0.08, 0.85, sky);
    let out_rgb = mix(tinted_base, refl_col, weight);
    let out_alpha = clamp(base_alpha + fresnel * 0.24, 0.54, 0.94);
    return vec4<f32>(out_rgb, out_alpha);
}

fn shade_voxel(in: VertexOut, texel: vec4<f32>) -> vec4<f32> {
    let dir_shade = in.light_ao.x;
    let sky = in.light_ao.y;
    let block = in.light_ao.z;
    let ao = in.light_ao.w;
    let t = cam.sun_dir_time.w;

    // Luanti dual-channel skylight + blocklight perceptual curve
    let sun_strength = max(0.14, cam.sun_dir_time.y * 0.82 + 0.34);
    let sky_lit = pow(max(sky, 1e-4), 1.35) * sun_strength;
    let block_lit = pow(max(block, 1e-4), 1.25);

    // Optional real-time handheld torch / lantern / lava-bucket point light
    var hand_light = 0.0;
    if (cam.hand_light_pos_intensity.w > 0.001) {
        let d_hand = distance(in.world_pos, cam.hand_light_pos_intensity.xyz);
        let radius = 15.0;
        if (d_hand < radius) {
            let falloff = clamp(1.0 - (d_hand / radius), 0.0, 1.0);
            let flicker = 0.95 + 0.05 * sin(t * 7.5 + sin(t * 3.1));
            hand_light = falloff * falloff * cam.hand_light_pos_intensity.w * flicker;
        }
    }

    var shadow_factor = 1.0;
    if (cam.flags.y > 0.5 && sky < 0.72 && block_lit < 0.35 && hand_light < 0.35) {
        shadow_factor = 0.78;
    }

    var base_albedo = texel.rgb * in.tint.rgb;
    let tint_cmax = max(in.tint.r, max(in.tint.g, in.tint.b));
    let tint_cmin = min(in.tint.r, min(in.tint.g, in.tint.b));
    if (cam.enhance_flags.x > 0.5) {
        let luma = dot(base_albedo, vec3<f32>(0.2126, 0.7152, 0.0722));
        let sat = select(1.08, 1.34, tint_cmax - tint_cmin > 0.05 || in.cutout == 1u);
        base_albedo = max(mix(vec3<f32>(luma), base_albedo, sat), vec3<f32>(0.0));
    }
    let hand_warm_color = vec3<f32>(1.0, 0.62, 0.267); // #FF9E44 warm orange-gold torch light
    var rgb: vec3<f32>;
    var out_alpha = texel.a * in.tint.a;

    if (cam.enhance_flags.x > 0.5) {
        // Enhanced shading: directional sun shadow map, warm key light, cool shadow bounce, ACES
        let sun_dir = normalize(cam.sun_dir_time.xyz);
        let sun_elev = clamp(cam.sun_dir_time.y, -0.35, 1.0);
        let n = face_normal_from_dir(in.face_dir);
        let ndotl = max(dot(n, sun_dir), 0.0);
        let sun_shadow = sample_sun_shadow(in.world_pos, n, sun_dir);
        let sunset = clamp(1.0 - abs(sun_elev - 0.12) * 3.0, 0.0, 1.0);
        let sun_color = mix(vec3<f32>(1.18, 1.06, 0.90), vec3<f32>(1.34, 0.78, 0.42), sunset);
        let cool_shadow = vec3<f32>(0.32, 0.44, 0.66);

        let direct_mask = ndotl * smoothstep(0.20, 0.90, sky) * smoothstep(-0.04, 0.22, sun_elev) * sun_shadow;
        let sky_tint = mix(cool_shadow, sun_color, direct_mask);
        let shadow_att = mix(0.56, 1.0, sun_shadow) * shadow_factor;
        let sky_contrib = base_albedo * sky_tint * (sky_lit * (0.86 + 0.50 * ndotl * sun_shadow) * dir_shade * ao * shadow_att);
        let block_warm = mix(vec3<f32>(1.0, 0.80, 0.56), hand_warm_color, clamp(block_lit * 1.15, 0.0, 1.0));
        let block_contrib = base_albedo * block_warm * (block_lit * dir_shade * ao * 1.05);
        let hand_contrib = base_albedo * hand_warm_color * (hand_light * 1.28 * (0.65 + 0.35 * dir_shade));
        let ambient_floor = base_albedo * cool_shadow * (0.062 * dir_shade * ao);

        rgb = (max(ambient_floor + hand_contrib, max(sky_contrib, block_contrib) + hand_contrib * 0.65)) * cam.flags.w;

        // Specular highlights on water, ice, ores, and polished metal blocks when sunlit
        let to_cam = cam.camera_pos.xyz - in.world_pos;
        let view_dir = select(vec3<f32>(0.0, 1.0, 0.0), normalize(to_cam), dot(to_cam, to_cam) > 1e-5);
        if ((in.mat_flags & 1u) != 0u && sky > 0.22 && sun_elev > -0.05) {
            let half_v = normalize(sun_dir + view_dir);
            let spec = pow(max(dot(n, half_v), 1e-4), 36.0) * 0.45 * sky_lit * sun_shadow;
            rgb += sun_color * spec * cam.flags.w;
        }
        if (in.waving == 1u && in.cutout == 0u) {
            let water_res = water_sky_reflection(in, rgb, out_alpha, sky, sun_strength);
            rgb = water_res.rgb;
            out_alpha = water_res.a;
        }
        // Emissive boost on torches, lava, glowstone, lanterns (subtle glow in Balanced, full HDR in Cinematic)
        if ((in.mat_flags & 2u) != 0u) {
            let bloom_boost = select(0.48, 0.94, cam.enhance_flags.z > 0.5);
            rgb += texel.rgb * vec3<f32>(1.15, 0.86, 0.52) * bloom_boost;
        }

        if (in.hurt_flash == 1u) {
            let flash_red = vec3<f32>(0.96, 0.14, 0.14) * max(0.55, dir_shade * max(sky_lit, block_lit));
            rgb = mix(rgb, flash_red, 0.62);
        }

        let fog_end = cam.fog_color_dist.w;
        if (fog_end < 4000.0) {
            let dist = length(to_cam);
            let eff_end = max(32.0, fog_end);
            let fog_start = eff_end * 0.65;
            let fog_t = smoothstep(fog_start, eff_end, dist);
            let sun_haze = pow(max(dot(-view_dir, sun_dir), 1e-4), 6.0) * 0.16 * smoothstep(-0.1, 0.25, sun_elev);
            let haze_col = mix(cam.fog_color_dist.rgb, sun_color * 0.95, sun_haze);
            rgb = mix(rgb, haze_col, fog_t);
        }

        // Apply ACES tone mapping inline when Bloom post-pass is disabled;
        // when Bloom is active, composite pass performs ACES after adding bloom.
        if (cam.enhance_flags.z < 0.5) {
            rgb = aces_filmic(rgb);
        }
    } else {
        // Fast classic Performance shading path
        let combined_light = max(sky_lit, max(block_lit, hand_light));
        let total_light = max(0.06, combined_light) * dir_shade * ao * cam.flags.w;
        let hand_glow = base_albedo * hand_warm_color * (hand_light * 0.72);
        let blk_glow = vec3<f32>(1.0, 0.82, 0.56) * (block_lit * 0.22);
        rgb = base_albedo * (total_light * shadow_factor) + blk_glow * texel.rgb + hand_glow;

        if (in.waving == 1u && in.cutout == 0u) {
            let water_res = water_sky_reflection(in, rgb, out_alpha, sky, sun_strength);
            rgb = water_res.rgb;
            out_alpha = water_res.a;
        }
        if ((in.mat_flags & 2u) != 0u && cam.enhance_flags.z > 0.5) {
            rgb += texel.rgb * vec3<f32>(1.10, 0.82, 0.50) * 0.78;
        }
        if (in.hurt_flash == 1u) {
            let flash_red = vec3<f32>(0.96, 0.14, 0.14) * max(0.55, dir_shade * total_light);
            rgb = mix(rgb, flash_red, 0.62);
        }

        let fog_end = cam.fog_color_dist.w;
        if (fog_end < 4000.0) {
            let dist = distance(in.world_pos, cam.camera_pos.xyz);
            let eff_end = max(32.0, fog_end);
            let fog_start = eff_end * 0.65;
            let fog_t = smoothstep(fog_start, eff_end, dist);
            rgb = mix(rgb, cam.fog_color_dist.rgb, fog_t);
        }
    }

    return vec4<f32>(rgb, out_alpha);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let texel = sample_voxel_texel(in);
    if (in.no_alpha_discard == 0u && texel.a < 0.5) {
        discard;
    }
    return shade_voxel(in, texel);
}

@fragment
fn fs_trans(in: VertexOut) -> @location(0) vec4<f32> {
    let texel = sample_voxel_texel(in);
    if (texel.a < 0.01) {
        discard;
    }
    return shade_voxel(in, texel);
}

struct LineVertexIn {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct LineVertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_line(in: LineVertexIn) -> LineVertexOut {
    var out: LineVertexOut;
    out.clip_pos = cam.view_proj * vec4<f32>(in.pos, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_line(in: LineVertexOut) -> @location(0) vec4<f32> {
    return in.color;
}
