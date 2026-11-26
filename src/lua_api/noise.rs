use mlua::{Lua, Result as LuaResult, Table, UserData, UserDataMethods, Value};
use std::sync::{Arc, Mutex};

/// Parse any Lua seed argument (integer, float, numeric/text string, nil) cleanly into `u64`
/// without failing when mapgen seeds exceed `i64::MAX`.
pub fn lua_value_to_u64_seed(val: Value) -> u64 {
    let float_to_u64 = |f: f64| -> u64 {
        if !f.is_finite() {
            return 0;
        }
        if f < 0.0 {
            (f as i64) as u64
        } else if f >= 18446744073709551616.0 {
            f.rem_euclid(18446744073709551616.0) as u64
        } else {
            f as u64
        }
    };
    match val {
        Value::Integer(n) => n as u64,
        Value::Number(f) => float_to_u64(f),
        Value::String(s) => {
            let trimmed = s
                .to_str()
                .map(|st| st.trim().to_string())
                .unwrap_or_default();
            if trimmed.is_empty() {
                return 0;
            }
            if let Ok(u) = trimmed.parse::<u64>() {
                return u;
            }
            if let Ok(i) = trimmed.parse::<i64>() {
                return i as u64;
            }
            if let Ok(f) = trimmed.parse::<f64>() {
                if f.is_finite() {
                    return float_to_u64(f);
                }
            }
            let mut h = 14695981039346656037u64;
            for b in trimmed.bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(1099511628211);
            }
            h
        }
        _ => 0,
    }
}

/// Parse a Luanti NoiseParams table or defaults.
#[derive(Debug, Clone)]
pub struct NoiseParams {
    pub offset: f64,
    pub scale: f64,
    pub spread: [f64; 3],
    pub seed: i64,
    pub octaves: usize,
    pub persist: f64,
    pub lacunarity: f64,
}

impl Default for NoiseParams {
    fn default() -> Self {
        Self {
            offset: 0.0,
            scale: 1.0,
            spread: [250.0, 250.0, 250.0],
            seed: 0,
            octaves: 3,
            persist: 0.6,
            lacunarity: 2.0,
        }
    }
}

impl NoiseParams {
    pub fn from_lua(val: Value) -> Self {
        let mut np = Self::default();
        if let Value::Table(t) = val {
            if let Ok(v) = t.get::<f64>("offset") { np.offset = v; }
            if let Ok(v) = t.get::<f64>("scale") { np.scale = v; }
            if let Ok(v) = t.get::<Value>("seed") {
                if !matches!(v, Value::Nil) {
                    np.seed = lua_value_to_u64_seed(v) as i64;
                }
            }
            if let Ok(v) = t.get::<usize>("octaves") { np.octaves = v.clamp(1, 12); }
            if let Ok(v) = t.get::<f64>("persist") { np.persist = v; }
            if let Ok(v) = t.get::<f64>("persistence") { np.persist = v; }
            if let Ok(v) = t.get::<f64>("lacunarity") { np.lacunarity = v; }
            if let Ok(sp) = t.get::<Table>("spread") {
                if let Ok(x) = sp.get::<f64>("x") { np.spread[0] = x.max(1e-4); }
                if let Ok(y) = sp.get::<f64>("y") { np.spread[1] = y.max(1e-4); }
                if let Ok(z) = sp.get::<f64>("z") { np.spread[2] = z.max(1e-4); }
            }
        }
        np
    }

    #[inline]
    fn hash3(x: i64, y: i64, z: i64, seed: i64) -> f64 {
        let mut n = x.wrapping_mul(1619)
            ^ y.wrapping_mul(31337)
            ^ z.wrapping_mul(6971)
            ^ seed.wrapping_mul(1013);
        n = (n ^ (n >> 13)).wrapping_mul(1274126177);
        let v = (n ^ (n >> 16)) & 0x7FFF_FFFF;
        (v as f64 / 1073741823.5) - 1.0
    }

    #[inline]
    fn fade(t: f64) -> f64 {
        t * t * (3.0 - 2.0 * t)
    }

    pub fn eval_2d(&self, x: f64, y: f64) -> f64 {
        let mut total = 0.0;
        let mut amp = 1.0;
        let mut freq = 1.0;
        for o in 0..self.octaves {
            let sx = x / self.spread[0] * freq;
            let sy = y / self.spread[1] * freq;
            let xi = sx.floor() as i64;
            let yi = sy.floor() as i64;
            let xf = Self::fade(sx - xi as f64);
            let yf = Self::fade(sy - yi as f64);
            let s = self.seed.wrapping_add(o as i64 * 1337);
            let n00 = Self::hash3(xi, yi, 0, s);
            let n10 = Self::hash3(xi + 1, yi, 0, s);
            let n01 = Self::hash3(xi, yi + 1, 0, s);
            let n11 = Self::hash3(xi + 1, yi + 1, 0, s);
            let ix0 = n00 + (n10 - n00) * xf;
            let ix1 = n01 + (n11 - n01) * xf;
            total += (ix0 + (ix1 - ix0) * yf) * amp;
            amp *= self.persist;
            freq *= self.lacunarity;
        }
        self.offset + total * self.scale
    }

    pub fn eval_3d(&self, x: f64, y: f64, z: f64) -> f64 {
        let mut total = 0.0;
        let mut amp = 1.0;
        let mut freq = 1.0;
        for o in 0..self.octaves {
            let sx = x / self.spread[0] * freq;
            let sy = y / self.spread[1] * freq;
            let sz = z / self.spread[2] * freq;
            let xi = sx.floor() as i64;
            let yi = sy.floor() as i64;
            let zi = sz.floor() as i64;
            let xf = Self::fade(sx - xi as f64);
            let yf = Self::fade(sy - yi as f64);
            let zf = Self::fade(sz - zi as f64);
            let s = self.seed.wrapping_add(o as i64 * 1337);
            let c000 = Self::hash3(xi, yi, zi, s);
            let c100 = Self::hash3(xi + 1, yi, zi, s);
            let c010 = Self::hash3(xi, yi + 1, zi, s);
            let c110 = Self::hash3(xi + 1, yi + 1, zi, s);
            let c001 = Self::hash3(xi, yi, zi + 1, s);
            let c101 = Self::hash3(xi + 1, yi, zi + 1, s);
            let c011 = Self::hash3(xi, yi + 1, zi + 1, s);
            let c111 = Self::hash3(xi + 1, yi + 1, zi + 1, s);
            let x00 = c000 + (c100 - c000) * xf;
            let x10 = c010 + (c110 - c010) * xf;
            let x01 = c001 + (c101 - c001) * xf;
            let x11 = c011 + (c111 - c011) * xf;
            let y0 = x00 + (x10 - x00) * yf;
            let y1 = x01 + (x11 - x01) * yf;
            total += (y0 + (y1 - y0) * zf) * amp;
            amp *= self.persist;
            freq *= self.lacunarity;
        }
        self.offset + total * self.scale
    }
}

#[derive(Clone)]
pub struct PerlinNoiseUD(pub NoiseParams);

impl UserData for PerlinNoiseUD {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("get_2d", |_, this, pos: Table| {
            let x: f64 = pos.get("x").unwrap_or(0.0);
            let y: f64 = pos.get("y").unwrap_or(0.0);
            Ok(this.0.eval_2d(x, y))
        });
        methods.add_method("get_3d", |_, this, pos: Table| {
            let x: f64 = pos.get("x").unwrap_or(0.0);
            let y: f64 = pos.get("y").unwrap_or(0.0);
            let z: f64 = pos.get("z").unwrap_or(0.0);
            Ok(this.0.eval_3d(x, y, z))
        });
        methods.add_method("get2d", |_, this, pos: Table| {
            let x: f64 = pos.get("x").unwrap_or(0.0);
            let y: f64 = pos.get("y").unwrap_or(0.0);
            Ok(this.0.eval_2d(x, y))
        });
        methods.add_method("get3d", |_, this, pos: Table| {
            let x: f64 = pos.get("x").unwrap_or(0.0);
            let y: f64 = pos.get("y").unwrap_or(0.0);
            let z: f64 = pos.get("z").unwrap_or(0.0);
            Ok(this.0.eval_3d(x, y, z))
        });
    }
}

#[derive(Clone)]
pub struct PerlinNoiseMapUD {
    pub np: NoiseParams,
    pub size: [usize; 3],
    pub buf: Arc<Mutex<Vec<f64>>>,
}

impl PerlinNoiseMapUD {
    pub fn new(np: NoiseParams, size: [usize; 3]) -> Self {
        Self {
            np,
            size: [size[0].max(1), size[1].max(1), size[2].max(1)],
            buf: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl UserData for PerlinNoiseMapUD {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("get_2d_map", |lua, this, pos: Table| {
            let ox: f64 = pos.get("x").unwrap_or(0.0);
            let oy: f64 = pos.get("y").unwrap_or(0.0);
            let sx = this.size[0];
            let sy = this.size[1];
            let outer = lua.create_table_with_capacity(sy, 0)?;
            for y in 0..sy {
                let row = lua.create_table_with_capacity(sx, 0)?;
                for x in 0..sx {
                    row.set(x + 1, this.np.eval_2d(ox + x as f64, oy + y as f64))?;
                }
                outer.set(y + 1, row)?;
            }
            Ok(outer)
        });
        methods.add_method("get_3d_map", |lua, this, pos: Table| {
            let ox: f64 = pos.get("x").unwrap_or(0.0);
            let oy: f64 = pos.get("y").unwrap_or(0.0);
            let oz: f64 = pos.get("z").unwrap_or(0.0);
            let [sx, sy, sz] = this.size;
            let outer = lua.create_table_with_capacity(sx, 0)?;
            for x in 0..sx {
                let yt = lua.create_table_with_capacity(sy, 0)?;
                for y in 0..sy {
                    let zt = lua.create_table_with_capacity(sz, 0)?;
                    for z in 0..sz {
                        zt.set(z + 1, this.np.eval_3d(ox + x as f64, oy + y as f64, oz + z as f64))?;
                    }
                    yt.set(y + 1, zt)?;
                }
                outer.set(x + 1, yt)?;
            }
            Ok(outer)
        });
        methods.add_method("get_2d_map_flat", |lua, this, (pos, buf_opt): (Table, Option<Table>)| {
            let ox: f64 = pos.get("x").unwrap_or(0.0);
            let oy: f64 = pos.get("y").unwrap_or(0.0);
            let sx = this.size[0];
            let sy = this.size[1];
            let out = buf_opt.unwrap_or_else(|| lua.create_table_with_capacity(sx * sy, 0).unwrap());
            let mut idx = 1;
            for y in 0..sy {
                for x in 0..sx {
                    out.set(idx, this.np.eval_2d(ox + x as f64, oy + y as f64))?;
                    idx += 1;
                }
            }
            Ok(out)
        });
        methods.add_method("get_3d_map_flat", |lua, this, (pos, buf_opt): (Table, Option<Table>)| {
            let ox: f64 = pos.get("x").unwrap_or(0.0);
            let oy: f64 = pos.get("y").unwrap_or(0.0);
            let oz: f64 = pos.get("z").unwrap_or(0.0);
            let [sx, sy, sz] = this.size;
            let out = buf_opt.unwrap_or_else(|| lua.create_table_with_capacity(sx * sy * sz, 0).unwrap());
            let mut idx = 1;
            for z in 0..sz {
                for y in 0..sy {
                    for x in 0..sx {
                        out.set(idx, this.np.eval_3d(ox + x as f64, oy + y as f64, oz + z as f64))?;
                        idx += 1;
                    }
                }
            }
            Ok(out)
        });
        methods.add_method("calc_2d_map", |_, this, pos: Table| {
            let ox: f64 = pos.get("x").unwrap_or(0.0);
            let oy: f64 = pos.get("y").unwrap_or(0.0);
            let sx = this.size[0];
            let sy = this.size[1];
            let mut buf = this.buf.lock().unwrap();
            buf.clear();
            buf.reserve(sx * sy);
            for y in 0..sy {
                for x in 0..sx {
                    buf.push(this.np.eval_2d(ox + x as f64, oy + y as f64));
                }
            }
            Ok(())
        });
        methods.add_method("calc_3d_map", |_, this, pos: Table| {
            let ox: f64 = pos.get("x").unwrap_or(0.0);
            let oy: f64 = pos.get("y").unwrap_or(0.0);
            let oz: f64 = pos.get("z").unwrap_or(0.0);
            let [sx, sy, sz] = this.size;
            let mut buf = this.buf.lock().unwrap();
            buf.clear();
            buf.reserve(sx * sy * sz);
            for z in 0..sz {
                for y in 0..sy {
                    for x in 0..sx {
                        buf.push(this.np.eval_3d(ox + x as f64, oy + y as f64, oz + z as f64));
                    }
                }
            }
            Ok(())
        });
        methods.add_method("get_map_slice", |lua, this, (_off, _sz, buf_opt): (Table, Table, Option<Table>)| {
            let buf = this.buf.lock().unwrap();
            let out = buf_opt.unwrap_or_else(|| lua.create_table_with_capacity(buf.len(), 0).unwrap());
            for (i, &v) in buf.iter().enumerate() {
                out.set(i + 1, v)?;
            }
            Ok(out)
        });
    }
}

/// Deterministic PCG32 PRNG matching Luanti's PcgRandom API.
#[derive(Clone)]
pub struct PcgRandomUD {
    state: Arc<Mutex<(u64, u64)>>,
}

impl PcgRandomUD {
    pub fn new(seed: u64, seq: u64) -> Self {
        let mut r = Self {
            state: Arc::new(Mutex::new((0u64, (seq << 1) | 1))),
        };
        let _ = r.next_u32();
        {
            let mut g = r.state.lock().unwrap();
            g.0 = g.0.wrapping_add(seed);
        }
        let _ = r.next_u32();
        r
    }

    fn next_u32(&mut self) -> u32 {
        let mut g = self.state.lock().unwrap();
        let old = g.0;
        g.0 = old.wrapping_mul(6364136223846793005).wrapping_add(g.1);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    fn next_range(&mut self, min: i64, max: i64) -> i64 {
        let (lo, hi) = if min <= max { (min, max) } else { (max, min) };
        let span = (hi as i128 - lo as i128 + 1).max(1) as u64;
        let r = ((self.next_u32() as u64) << 32) | (self.next_u32() as u64);
        lo + (r % span) as i64
    }
}

impl UserData for PcgRandomUD {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("next", |_, this, (min_o, max_o): (Option<i64>, Option<i64>)| {
            let mut c = this.clone();
            match (min_o, max_o) {
                (Some(a), Some(b)) => Ok(c.next_range(a, b)),
                (Some(a), None) => Ok(c.next_range(a, i32::MAX as i64)),
                _ => Ok(c.next_range(i32::MIN as i64, i32::MAX as i64)),
            }
        });
        methods.add_method("rand_normal_dist", |_, this, (min_o, max_o, trials_o): (Option<i64>, Option<i64>, Option<usize>)| {
            let mut c = this.clone();
            let lo = min_o.unwrap_or(i32::MIN as i64);
            let hi = max_o.unwrap_or(i32::MAX as i64);
            let trials = trials_o.unwrap_or(6).clamp(1, 32);
            let mut sum: i128 = 0;
            for _ in 0..trials {
                sum += c.next_range(lo, hi) as i128;
            }
            Ok((sum / trials as i128) as i64)
        });
        methods.add_method("get_state", |_, this, ()| {
            let g = this.state.lock().unwrap();
            Ok(format!("{:016x}{:016x}", g.0, g.1))
        });
        methods.add_method("set_state", |_, this, s: String| {
            if s.len() >= 32 {
                if let (Ok(a), Ok(b)) = (u64::from_str_radix(&s[0..16], 16), u64::from_str_radix(&s[16..32], 16)) {
                    let mut g = this.state.lock().unwrap();
                    g.0 = a;
                    g.1 = b | 1;
                }
            }
            Ok(())
        });
    }
}

/// Luanti PseudoRandom & SecureRandom.
#[derive(Clone)]
pub struct PseudoRandomUD(pub Arc<Mutex<u32>>);

impl UserData for PseudoRandomUD {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("next", |_, this, (min_o, max_o): (Option<i64>, Option<i64>)| {
            let mut g = this.0.lock().unwrap();
            *g = g.wrapping_mul(1103515245).wrapping_add(12345);
            let val = ((*g >> 16) & 0x7FFF) as i64;
            let lo = min_o.unwrap_or(0);
            let hi = max_o.unwrap_or(32767);
            let span = (hi - lo + 1).max(1);
            Ok(lo + (val % span))
        });
        methods.add_method("get_state", |_, this, ()| Ok(*this.0.lock().unwrap()));
    }
}

pub fn register_noise_and_rng(lua: &Lua) -> LuaResult<()> {
    let g = lua.globals();

    g.set("PcgRandom", lua.create_function(|_, (seed, seq): (Value, Option<Value>)| {
        let s = lua_value_to_u64_seed(seed);
        let q = seq.map(lua_value_to_u64_seed).unwrap_or(0);
        Ok(PcgRandomUD::new(s, q))
    })?)?;

    g.set("PseudoRandom", lua.create_function(|_, seed: Value| {
        let s = lua_value_to_u64_seed(seed) as u32;
        Ok(PseudoRandomUD(Arc::new(Mutex::new(s))))
    })?)?;

    g.set("SecureRandom", lua.create_function(|_, ()| {
        let s: u64 = rand::random();
        Ok(PcgRandomUD::new(s, 0xCAFEBABE))
    })?)?;

    let perlin_ctor = lua.create_function(|_, np: Value| Ok(PerlinNoiseUD(NoiseParams::from_lua(np))))?;
    g.set("PerlinNoise", perlin_ctor.clone())?;
    g.set("ValueNoise", perlin_ctor)?;

    let pmap_ctor = lua.create_function(|_, (np, size): (Value, Table)| {
        let sx: usize = size.get("x").unwrap_or(16);
        let sy: usize = size.get("y").unwrap_or(16);
        let sz: usize = size.get("z").unwrap_or(1);
        Ok(PerlinNoiseMapUD::new(NoiseParams::from_lua(np), [sx, sy, sz]))
    })?;
    g.set("PerlinNoiseMap", pmap_ctor.clone())?;
    g.set("ValueNoiseMap", pmap_ctor)?;

    // LuaJIT already includes `bit`, ensure global `bit` exists
    let _: Value = lua.load("if not bit then bit = require('bit') end").eval()?;
    Ok(())
}

#[cfg(test)]
mod noise_seed_tests {
    use super::*;

    #[test]
    fn rng_handles_large_u64_negative_float_and_nil_seeds() {
        let lua = unsafe { Lua::unsafe_new() };
        register_noise_and_rng(&lua).expect("register noise/rng");
        lua.load(
            r#"
            local p1 = PcgRandom("15905940020608058721")
            assert(type(p1:next(1, 100)) == "number")
            local p2 = PcgRandom(-42, 7)
            assert(type(p2:next(0, 10)) == "number")
            local p3 = PcgRandom(1e19)
            assert(type(p3:next()) == "number")
            local p4 = PcgRandom(nil)
            assert(type(p4:next()) == "number")
            local ps = PseudoRandom("15905940020608058721")
            assert(type(ps:next(0, 50)) == "number")
            local n = PerlinNoise({ offset = 0, scale = 1, seed = "15905940020608058721", octaves = 2 })
            assert(type(n:get_2d({ x = 1, y = 2 })) == "number")
        "#,
        )
        .exec()
        .expect("Lua RNG & noise seed handling must never fail on large u64 seeds");
    }
}
