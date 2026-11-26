# Cubex-RS

`cubex-rs` is a native Linux Rust voxel engine powered by `winit`, `wgpu` (Vulkan), `egui`, `rayon`, and embedded LuaJIT (`mlua` vendored). It implements the Luanti / Mineclonia Lua modding API (`core.*` / `minetest.*`) so Mineclonia mods, resource packs, schemas, and entities run directly on native GPU rendering.

## Features

- **Luanti & Mineclonia Runtime:** Embedded LuaJIT engine loading `builtin/`, `game.conf`, `mod.conf` / `modpack.conf` dependency graphs, `VoxelManip`, `.mts` schematics, node timers, ABMs, formspecs, and `ObjectRef` entities.
- **Voxel World & Lighting:** `16x16x16` cubic mapblocks with greedy opaque meshing, dual 0..15 sky/block light propagation, priority chunk remeshing, multi-biome mapgen, and compressed binary world persistence.
- **Configurable GPU Renderer (`wgpu`):**
  - Procedural atmospheric sky dome, day/sunset/night palette transitions, sun disc, stars, and drifting FBM clouds (`sky.rs`).
  - `2048x2048` directional sun shadow map with 3x3 PCF filtering (`shadow.rs`).
  - Handheld dynamic torch/light-source illumination, wind-swaying foliage, Fresnel water reflections with underwater depth tint, ACES filmic tone mapping, and bloom post-processing (`bloom.rs`).
  - Live Video Settings presets (`Performance`, `Balanced`, `Cinematic`, `Custom`) persisted in `~/.minetest/cubex_data/settings.toml`.
- **Survival Gameplay & Entities:**
  - Swept AABB voxel physics with `0.6`-block stepheight, sneaking edge-catch, swimming, ladders, hunger/saturation/exhaustion, right-click food eating, natural health regeneration, and starvation.
  - Physical mobs and falling nodes (`__builtin:falling_node`) with straight-down gravity and solid grid re-placement, plus `Keep Inventory on Death` (on by default) and per-viewport respawn screens.
- **Multiplayer, Split-Screen & Streaming:**
  - Local co-op up to 4 players across split-screen viewports or multiple displays, gamepad support (`gilrs`), touch controls, LAN discovery (`--port 8090` browser stream / game networking), `CBX-...` invites, `playit.gg` integration, and `--dedicated` headless server mode.

## Prerequisites

- **Rust toolchain:** `cargo` and `rustc` (2021 edition)
- **C compiler:** `gcc` / `clang` (`.cargo/config.toml` sets `CFLAGS=-std=gnu11` for vendored LuaJIT)
- **Graphics:** Vulkan-capable GPU driver
- **Game assets:** Luanti `builtin/` and `mineclonia` game directory (resolved from `~/.minetest/games/mineclonia`, `CUBEX_GAME_DIR`, and `CUBEX_BUILTIN_DIR`)

## Building & Running

```bash
# Release build
cargo build --release

# Launch the interactive title screen and 3D client
./target/release/cubex-rs

# Headless GPU benchmark and Mineclonia load check
./target/release/cubex-rs --smoke-test

# Automated headless playtest and screenshot capture (writes playtest_out/*.png)
./target/release/cubex-rs --playtest

# Dedicated headless multiplayer server
./target/release/cubex-rs --dedicated --port 30000
```

## Controls (Default)

| Action | Key / Input |
| --- | --- |
| Move | `W`, `A`, `S`, `D` |
| Jump / Swim Up | `Space` |
| Sneak | `Left Shift` |
| Sprint | `Left Ctrl` |
| Mine / Attack | Left Mouse Button |
| Use / Place / Eat Held Food | Right Mouse Button |
| Inventory | `E` |
| Drop Item | `Q` |
| Chat / Command | `T` / `/` |
| Cycle Camera Mode (1P / 3P) | `F5` |
| Debug Overlay | `F3` |
| Fullscreen | `F11` |
| Pause / Options Menu | `Escape` |

## Testing

Vulkan/headless `wgpu` tests share a single device setup sequentially, so run the suite single-threaded:

```bash
cargo check --all-targets
cargo test -- --test-threads=1
```
