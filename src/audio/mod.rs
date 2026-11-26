use parking_lot::Mutex;
use rand::Rng;
use rodio::buffer::SamplesBuffer;
use rodio::{Decoder, OutputStream, Sink};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, SyncSender};
use std::sync::Arc;
use std::thread;

use crate::config::AudioSettings;
use crate::registry::SoundSpec;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoundCategory {
    Master,
    Music,
    Blocks,
    Weather,
    Hostile,
    Friendly,
    Ambient,
    Ui,
}

impl SoundCategory {
    pub fn effective_volume(self, settings: &AudioSettings, gain: f32) -> f32 {
        let cat_vol = match self {
            Self::Master => 1.0,
            Self::Music => settings.music,
            Self::Blocks => settings.blocks,
            Self::Weather => settings.weather,
            Self::Hostile => settings.hostile,
            Self::Friendly => settings.friendly,
            Self::Ambient => settings.ambient,
            Self::Ui => settings.blocks.max(0.6),
        };
        (settings.master * cat_vol * gain).clamp(0.0, 2.0)
    }
}

enum AudioCmd {
    PlayFile {
        path: PathBuf,
        volume: f32,
        pitch: f32,
    },
    PlaySamples {
        channels: u16,
        sample_rate: u32,
        samples: Vec<f32>,
        volume: f32,
        pitch: f32,
    },
}

/// Non-blocking rodio audio engine with .ogg sound scanning and silent fallback when headless.
pub struct AudioEngine {
    /// Map of sound base name (e.g. "default_dig_choppy") -> list of variant .ogg file paths.
    pub sounds: HashMap<String, Vec<PathBuf>>,
    tx: Option<SyncSender<AudioCmd>>,
    device_ready: Arc<AtomicBool>,
    click_pcm: Vec<f32>,
    pickup_pcm: Vec<f32>,
    eat_pcm: Vec<f32>,
    last_played: Arc<Mutex<Option<String>>>,
}

impl AudioEngine {
    pub fn new(mod_paths: &HashMap<String, PathBuf>, game_dir: &Path) -> Self {
        let sounds = scan_all_sounds(mod_paths, game_dir);
        let device_ready = Arc::new(AtomicBool::new(false));
        let ready_flag = device_ready.clone();
        let (tx, rx) = mpsc::sync_channel::<AudioCmd>(64);

        // Spawn dedicated audio thread so OutputStream (!Send on ALSA) never leaves its thread
        // and headless CI environments without audio hardware never fail or panic.
        let _ = thread::Builder::new()
            .name("cubex-audio".into())
            .spawn(move || {
                let stream = OutputStream::try_default();
                let Ok((_output_stream, handle)) = stream else {
                    tracing::info!("No audio output device available; AudioEngine running in silent fallback mode");
                    while rx.recv().is_ok() {}
                    return;
                };
                ready_flag.store(true, Ordering::Relaxed);
                while let Ok(cmd) = rx.recv() {
                    let Ok(sink) = Sink::try_new(&handle) else {
                        continue;
                    };
                    match cmd {
                        AudioCmd::PlayFile { path, volume, pitch } => {
                            if let Ok(file) = File::open(&path) {
                                if let Ok(dec) = Decoder::new(BufReader::new(file)) {
                                    sink.set_volume(volume);
                                    sink.set_speed(pitch.clamp(0.25, 4.0));
                                    sink.append(dec);
                                    sink.detach();
                                }
                            }
                        }
                        AudioCmd::PlaySamples {
                            channels,
                            sample_rate,
                            samples,
                            volume,
                            pitch,
                        } => {
                            let buf = SamplesBuffer::new(channels, sample_rate, samples);
                            sink.set_volume(volume);
                            sink.set_speed(pitch.clamp(0.25, 4.0));
                            sink.append(buf);
                            sink.detach();
                        }
                    }
                }
            });

        Self {
            sounds,
            tx: Some(tx),
            device_ready,
            click_pcm: synth_click_waveform(),
            pickup_pcm: synth_pickup_waveform(),
            eat_pcm: synth_eat_waveform(),
            last_played: Arc::new(Mutex::new(None)),
        }
    }

    pub fn new_silent() -> Self {
        Self {
            sounds: HashMap::new(),
            tx: None,
            device_ready: Arc::new(AtomicBool::new(false)),
            click_pcm: synth_click_waveform(),
            pickup_pcm: synth_pickup_waveform(),
            eat_pcm: synth_eat_waveform(),
            last_played: Arc::new(Mutex::new(None)),
        }
    }

    pub fn empty_silent() -> Self {
        Self {
            sounds: HashMap::new(),
            tx: None,
            device_ready: Arc::new(AtomicBool::new(false)),
            click_pcm: Vec::new(),
            pickup_pcm: Vec::new(),
            eat_pcm: Vec::new(),
            last_played: Arc::new(Mutex::new(None)),
        }
    }

    pub fn has_device(&self) -> bool {
        self.device_ready.load(Ordering::Relaxed)
    }

    pub fn sound_count(&self) -> usize {
        self.sounds.len()
    }

    pub fn last_played_sound(&self) -> Option<String> {
        self.last_played.lock().clone()
    }

    pub fn resolve_sound_path(&self, name: &str) -> Option<PathBuf> {
        let clean = name.trim().strip_suffix(".ogg").unwrap_or(name.trim());
        if clean.is_empty() {
            return None;
        }
        let list = self.sounds.get(clean)?;
        if list.is_empty() {
            return None;
        }
        let idx = if list.len() == 1 {
            0
        } else {
            rand::thread_rng().gen_range(0..list.len())
        };
        list.get(idx).cloned()
    }

    pub fn play_named(
        &self,
        name: &str,
        gain: f32,
        pitch: f32,
        cat: SoundCategory,
        settings: &AudioSettings,
    ) -> bool {
        let vol = cat.effective_volume(settings, gain);
        if vol <= 0.001 {
            return false;
        }
        let Some(path) = self.resolve_sound_path(name) else {
            return false;
        };
        *self.last_played.lock() = Some(name.to_string());
        if let Some(ref tx) = self.tx {
            let _ = tx.try_send(AudioCmd::PlayFile {
                path,
                volume: vol,
                pitch: if pitch > 0.0 { pitch } else { 1.0 },
            });
        }
        true
    }

    pub fn play_ui_click(&self, settings: &AudioSettings) {
        let vol = SoundCategory::Ui.effective_volume(settings, 0.75);
        if vol <= 0.001 {
            return;
        }
        for candidate in ["gui_click", "mcl_click", "default_place_node_hard"] {
            if self.play_named(candidate, 0.6, 1.15, SoundCategory::Ui, settings) {
                return;
            }
        }
        *self.last_played.lock() = Some("__synth_ui_click".into());
        if let Some(ref tx) = self.tx {
            let _ = tx.try_send(AudioCmd::PlaySamples {
                channels: 1,
                sample_rate: 22050,
                samples: self.click_pcm.clone(),
                volume: vol,
                pitch: 1.0,
            });
        }
    }

    pub fn play_pickup(&self, settings: &AudioSettings) {
        let vol = SoundCategory::Blocks.effective_volume(settings, 0.7);
        if vol <= 0.001 {
            return;
        }
        for candidate in ["item_pickup", "mcl_item_pickup", "default_item_pickup"] {
            if self.play_named(candidate, 0.65, 1.0, SoundCategory::Blocks, settings) {
                return;
            }
        }
        *self.last_played.lock() = Some("__synth_pickup".into());
        if let Some(ref tx) = self.tx {
            let _ = tx.try_send(AudioCmd::PlaySamples {
                channels: 1,
                sample_rate: 22050,
                samples: self.pickup_pcm.clone(),
                volume: vol,
                pitch: 1.0,
            });
        }
    }

    pub fn play_eat(&self, settings: &AudioSettings) {
        let vol = SoundCategory::Friendly.effective_volume(settings, 0.85);
        if vol <= 0.001 {
            return;
        }
        for candidate in ["mcl_hunger_eat", "eat", "player_eat"] {
            let pitch = rand::thread_rng().gen_range(0.90f32..1.12f32);
            if self.play_named(candidate, 0.8, pitch, SoundCategory::Friendly, settings) {
                return;
            }
        }
        *self.last_played.lock() = Some("__synth_eat".into());
        if let Some(ref tx) = self.tx {
            let _ = tx.try_send(AudioCmd::PlaySamples {
                channels: 1,
                sample_rate: 22050,
                samples: self.eat_pcm.clone(),
                volume: vol,
                pitch: rand::thread_rng().gen_range(0.92f32..1.08f32),
            });
        }
    }

    pub fn play_ambient(&self, name: &str, gain: f32, settings: &AudioSettings) {
        self.play_named(name, gain, 1.0, SoundCategory::Ambient, settings);
    }

    pub fn play_footstep(&self, name: &str, gain: f32, settings: &AudioSettings) {
        let jitter = rand::thread_rng().gen_range(0.92f32..1.08f32);
        self.play_named(name, gain * 0.55, jitter, SoundCategory::Blocks, settings);
    }

    pub fn play_node_sound(&self, spec: &SoundSpec, cat: SoundCategory, settings: &AudioSettings) {
        if spec.name.trim().is_empty() {
            return;
        }
        let pitch = if spec.pitch > 0.0 {
            spec.pitch
        } else {
            rand::thread_rng().gen_range(0.94f32..1.06f32)
        };
        let gain = if spec.gain > 0.0 { spec.gain } else { 1.0 };
        self.play_named(&spec.name, gain, pitch, cat, settings);
    }
}

pub fn scan_all_sounds(
    mod_paths: &HashMap<String, PathBuf>,
    game_dir: &Path,
) -> HashMap<String, Vec<PathBuf>> {
    let mut map: HashMap<String, Vec<PathBuf>> = HashMap::new();
    scan_sound_dir(&game_dir.join("sounds"), &mut map);
    for mp in mod_paths.values() {
        scan_sound_dir(&mp.join("sounds"), &mut map);
    }
    for list in map.values_mut() {
        list.sort();
        list.dedup();
    }
    map
}

fn scan_sound_dir(dir: &Path, out: &mut HashMap<String, Vec<PathBuf>>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for ent in entries.flatten() {
        let p = ent.path();
        if p.is_dir() {
            scan_sound_dir(&p, out);
            continue;
        }
        let is_ogg = p
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("ogg"))
            .unwrap_or(false);
        if !is_ogg {
            continue;
        }
        let Some(stem) = p.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        // Always register exact stem (e.g. "default_dig_choppy.1" or "default_dig_crumbly")
        out.entry(stem.to_string()).or_default().push(p.clone());

        // Also register Luanti random-variant base name: "foo.1.ogg" -> "foo"
        if let Some((base, suffix)) = stem.rsplit_once('.') {
            if !base.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
                out.entry(base.to_string()).or_default().push(p);
            }
        }
    }
}

fn synth_click_waveform() -> Vec<f32> {
    let sample_rate = 22050usize;
    let len = (sample_rate as f32 * 0.045) as usize;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let t = i as f32 / sample_rate as f32;
        let env = (-t * 95.0).exp();
        let s = ((std::f32::consts::TAU * 640.0 * t).sin() * 0.6
            + (std::f32::consts::TAU * 1280.0 * t).sin() * 0.4)
            * env
            * 0.45;
        out.push(s);
    }
    out
}

fn synth_pickup_waveform() -> Vec<f32> {
    let sample_rate = 22050usize;
    let len = (sample_rate as f32 * 0.075) as usize;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let t = i as f32 / sample_rate as f32;
        let freq = 520.0 + t * 6500.0;
        let env = (1.0 - t / 0.075).max(0.0);
        let s = (std::f32::consts::TAU * freq * t).sin() * env * 0.35;
        out.push(s);
    }
    out
}

fn synth_eat_waveform() -> Vec<f32> {
    let sample_rate = 22050usize;
    let dur = 0.16f32;
    let len = (sample_rate as f32 * dur) as usize;
    let mut out = Vec::with_capacity(len);
    let mut rng_state = 0x5A17_4321u32;
    for i in 0..len {
        let t = i as f32 / sample_rate as f32;
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        let noise = ((rng_state >> 16) as f32 / 32768.0) - 1.0;
        let burst = ((t * 28.0).sin().abs()).powf(1.5);
        let env = (1.0 - t / dur).max(0.0);
        let low = (std::f32::consts::TAU * 190.0 * t).sin() * 0.35;
        out.push((noise * 0.45 + low) * burst * env * 0.45);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scans_ogg_variants_and_silent_fallback_never_panics() {
        let dir = std::env::temp_dir().join("cubex_rs_audio_scan_test");
        let _ = std::fs::remove_dir_all(&dir);
        let snd_dir = dir.join("mod_a").join("sounds");
        std::fs::create_dir_all(&snd_dir).unwrap();
        std::fs::write(snd_dir.join("stone_dig.1.ogg"), b"OggS").unwrap();
        std::fs::write(snd_dir.join("stone_dig.2.ogg"), b"OggS").unwrap();
        std::fs::write(snd_dir.join("ui_pop.ogg"), b"OggS").unwrap();

        let mut mod_paths = HashMap::new();
        mod_paths.insert("mod_a".into(), dir.join("mod_a"));
        let engine = AudioEngine::new(&mod_paths, &dir.join("game"));
        assert_eq!(engine.sounds.get("stone_dig").map(Vec::len), Some(2));
        assert!(engine.resolve_sound_path("stone_dig").is_some());
        assert!(engine.resolve_sound_path("ui_pop.ogg").is_some());

        let settings = AudioSettings::default();
        engine.play_ui_click(&settings);
        engine.play_pickup(&settings);
        engine.play_footstep("stone_dig", 0.8, &settings);
        engine.play_node_sound(
            &SoundSpec {
                name: "stone_dig".into(),
                gain: 0.9,
                pitch: 1.0,
            },
            SoundCategory::Blocks,
            &settings,
        );
        assert_eq!(engine.last_played_sound().as_deref(), Some("stone_dig"));
    }
}
