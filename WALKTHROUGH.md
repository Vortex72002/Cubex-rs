# Cubex (`cubex-rs`) Developer Guide

**Cubex (`cubex-rs`)** is a native Linux Rust voxel engine (`winit` + `wgpu` + `egui` + `rayon`) embedding **LuaJIT (`mlua` vendored)** and implementing the **Luanti / Mineclonia Lua Modding API** (`core.*` / `minetest.*`) so Mineclonia mods, resource packs, and custom Lua overrides (`cubex_bridge`) run directly on Vulkan hardware.

---

## 1. Repository Layout

```text
cubex-rs/
├── .gitignore                 # Excludes /target, /playtest_out, *.log, editor swap/backup files
├── .cargo/config.toml         # CFLAGS="-std=gnu11" for vendored LuaJIT compatibility
├── Cargo.toml                 # Dependencies (wgpu, egui, mlua, rodio, gilrs, tokio) & fat-LTO release profile
├── README.md                  # Quickstart, prerequisites, controls, CLI modes, and test instructions
├── ARCHITECTURE.md            # Subsystem-to-Mineclonia API mapping and voxel conventions
├── WALKTHROUGH.md             # Module layout & development guide
├── src/
│   ├── lib.rs                 # Crate module exports
│   ├── main.rs                # winit + wgpu + egui loop, split-screen/multi-window viewports, CLI entrypoints
│   ├── paths.rs               # Resolves Mineclonia, Luanti builtin/, Craft & Ruin 32x, worlds, minetest.conf, playit
│   ├── config.rs              # minetest.conf reader/writer + EngineSettings & GraphicsPreset persistence
│   ├── registry.rs            # NodeDef, ItemDef, ToolCapabilities, Craft/Biome/Ore/Decoration/Entity registries
│   ├── audio/
│   │   └── mod.rs             # rodio .ogg playback, synthesized UI/pickup/eat waveforms, headless fallback
│   ├── input/
│   │   ├── mod.rs             # Input module exports
│   │   ├── gamepad.rs         # gilrs controller support & radial deadzone filtering
│   │   └── touch.rs           # On-screen touch layout & gesture state
│   ├── lua_api/
│   │   ├── mod.rs             # LuaRuntime bootstrap, player lifecycle, falling-node & entity step, item on_use
│   │   ├── core_api.rs        # Luanti C-API primitives, .mts schematics, craft resolver, formspec & sound queues
│   │   ├── mod_loader.rs      # game.conf & mod.conf / modpack.conf dependency sorting & loader
│   │   ├── noise.rs           # PcgRandom, PseudoRandom, SecureRandom, PerlinNoise & ValueNoise maps
│   │   └── userdata.rs        # Settings, ItemStack, MetaDataRef, InvRef, NodeTimerRef, VoxelManip, ObjectRef, AreaStore
│   ├── physics/
│   │   ├── mod.rs             # Physics exports, food lookup & unit tests
│   │   ├── collision.rs       # Swept AABB movement, stepheight, hunger/exhaustion/regen/starvation, eye-center suffocation
│   │   └── raycast.rs         # 3D DDA raycast, DigController, interact/place & food eating handlers
│   ├── textures/
│   │   └── mod.rs             # Resource pack scanner, ^[... modifiers, 2DArray atlas packer, skin & crack layers
│   ├── world/
│   │   ├── mod.rs             # 16x16x16 mapblocks, ChunkSnapshot, WorldStore, priority dirty queue, gamerules
│   │   ├── biome.rs           # Multi-biome sampler (SEA_LEVEL = 62)
│   │   ├── light.rs           # Dual 0..15 sky/block light propagation & incremental relight
│   │   ├── mapgen.rs          # Parallel multi-biome terrain & tree generation
│   │   ├── schematic.rs       # Binary .mts v4 decoder/encoder & world placement
│   │   └── persistence.rs     # Compressed world persistence (world.mt, chunks.bin, players.bin, nodemeta.bin)
│   ├── render/
│   │   ├── mod.rs             # wgpu Vulkan renderer, shadow & bloom passes, chunk upload budget, headless readback
│   │   ├── bloom.rs           # Bloom prefilter, downsample/upsample pyramid & composite post-pass (+ bloom.wgsl)
│   │   ├── camera.rs          # F5 CameraMode (1P / 3P Back / 3P Front) & collision pull-in rig
│   │   ├── entity_model.rs    # Mob quad/limb meshes, textures, hurt flash & __builtin:falling_node block mesh
│   │   ├── frustum.rs         # 6-plane Gribb-Hartmann frustum culler
│   │   ├── mesher.rs          # rayon parallel greedy chunk mesher across Luanti drawtypes
│   │   ├── pipeline.rs        # Selection wireframe, crack overlay & 3D break ParticleSystem
│   │   ├── player_model.rs    # Articulated 6-part 3rd-person player character
│   │   ├── shadow.rs          # 2048x2048 directional sun shadow map renderer (+ shadow.wgsl)
│   │   ├── sky.rs             # Procedural atmospheric sky gradient, sun, stars & FBM clouds (+ sky.wgsl)
│   │   ├── viewmodel.rs       # 1st-person arm, extruded held item, walk/swing/eat animations
│   │   └── shader.wgsl        # Voxel & line WGSL shaders (PCF shadows, torchlight, Fresnel water, fog, ACES)
│   ├── ui/
│   │   ├── mod.rs             # egui UI exports
│   │   ├── death_screen.rs    # Per-viewport crimson death overlay with Respawn & Main Menu actions
│   │   ├── formspec.rs        # Luanti formspec parser & interactive inventory slot transfers
│   │   ├── hud.rs             # Crosshair, hotbar, hearts/armor/hunger/XP bars, chat & F3 debug overlay
│   │   ├── loading_screen.rs  # World loading progress screen
│   │   ├── pause_menu.rs      # Unified ESC game menu & live Video/Audio/Controls/World settings
│   │   ├── splash_screen.rs   # Engine startup screen
│   │   ├── theme.rs           # Nearest-neighbor stone buttons, dirt/deepslate backgrounds, icon cache, § tooltips
│   │   ├── title_screen.rs    # 3D panorama title screen & Singleplayer/Multiplayer/Host/Options views
│   │   └── touch_hud.rs       # On-screen touch controls overlay
│   └── network/
│       ├── mod.rs             # Network exports
│       ├── invite.rs          # CBX-... Base64 invite encoder & decoder
│       ├── lan.rs             # UDP LAN broadcast & discovery scanner
│       ├── motd.rs            # Minecraft §-color & Luanti escape parser
│       ├── protocol.rs        # Async tokio client/server, chunk sync, chat & --dedicated server
│       ├── servers.rs         # Async server pinger, featured servers & icons
│       ├── tunnel.rs          # PlayitTunnelManager subprocess wrapper
│       └── web_client.rs      # LAN browser streaming & touch-friendly WebSocket/HTTP client
└── tests/
    ├── mineclonia_loader.rs   # Mineclonia mod tree & resource pack atlas checks
    ├── engine_tests.rs        # VoxelManip/light, greedy mesh, frustum, physics, formspecs, network & playit tests
    └── parity_tests.rs        # Schematics, mesher/light, dig/craft, viewmodel, persistence, biomes & camera checks
```

---

## 2. Building & CLI Modes

```bash
# Release binary (opt-level = 3, fat LTO, single codegen unit)
cargo build --release

# Interactive title screen & 3D client
./target/release/cubex-rs

# Headless GPU smoke test & Mineclonia mod loader verification
./target/release/cubex-rs --smoke-test

# Automated headless playtest & screenshot capture (writes playtest_out/*.png)
./target/release/cubex-rs --playtest

# Dedicated headless multiplayer server
./target/release/cubex-rs --dedicated --port 30000
```

## 3. Running Tests

```bash
cargo check --all-targets
cargo test -- --test-threads=1
```
