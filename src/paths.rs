use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct EnginePaths {
    pub project_root: PathBuf,
    pub builtin_dir: PathBuf,
    pub game_dir: PathBuf,
    pub minetest_dir: PathBuf,
    pub textures_dir: PathBuf,
    pub skins_dir: PathBuf,
    pub worlds_dir: PathBuf,
    pub conf_file: PathBuf,
    pub lua_override_dir: PathBuf,
    pub playit_bin: PathBuf,
    pub playit_config_dir: PathBuf,
    pub cubex_data_dir: PathBuf,
    pub settings_toml_file: PathBuf,
}

impl EnginePaths {
    pub fn resolve() -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let minetest_dir = home.join(".minetest");
        let fallback_project = home.join("cubex-project");
        let project_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

        let builtin_dir = std::env::var_os("CUBEX_BUILTIN_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let local = project_root.join("cubex-engine-src").join("builtin");
                if local.is_dir() {
                    local
                } else {
                    fallback_project.join("cubex-engine-src").join("builtin")
                }
            });

        let game_dir = std::env::var_os("CUBEX_GAME_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let mcl = minetest_dir.join("games").join("mineclonia");
                if mcl.is_dir() {
                    mcl
                } else {
                    minetest_dir.join("games").join("cubex")
                }
            });

        let worlds_dir = std::env::var_os("CUBEX_WORLD_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| minetest_dir.join("worlds"));

        let textures_dir = minetest_dir.join("textures");
        let skins_dir = std::env::var_os("CUBEX_SKINS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let local = project_root.join("assets").join("skins");
                if local.is_dir() {
                    return local;
                }
                if let Ok(exe) = std::env::current_exe() {
                    if let Some(exe_dir) = exe.parent() {
                        let near_exe = exe_dir.join("assets").join("skins");
                        if near_exe.is_dir() {
                            return near_exe;
                        }
                    }
                }
                let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("assets")
                    .join("skins");
                if manifest.is_dir() {
                    manifest
                } else {
                    local
                }
            });
        let conf_file = minetest_dir.join("minetest.conf");
        let lua_override_dir = fallback_project.join("lua_overrides").join("cubex_bridge");
        let playit_bin = fallback_project.join("tunnel").join("playit");
        let playit_config_dir = home.join(".config").join("playit_gg");
        let cubex_data_dir = minetest_dir.join("cubex_data");
        let settings_toml_file = cubex_data_dir.join("settings.toml");

        Self {
            project_root,
            builtin_dir,
            game_dir,
            minetest_dir,
            textures_dir,
            skins_dir,
            worlds_dir,
            conf_file,
            lua_override_dir,
            playit_bin,
            playit_config_dir,
            cubex_data_dir,
            settings_toml_file,
        }
    }

    pub fn ensure_dirs(&self) {
        let _ = std::fs::create_dir_all(&self.worlds_dir);
        let _ = std::fs::create_dir_all(&self.textures_dir);
        let _ = std::fs::create_dir_all(&self.skins_dir);
        let _ = std::fs::create_dir_all(&self.cubex_data_dir);
        let _ = std::fs::create_dir_all(&self.playit_config_dir);
    }
}
