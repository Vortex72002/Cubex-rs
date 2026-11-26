use std::collections::HashMap;

pub const CONTENT_UNKNOWN: u16 = 125;
pub const CONTENT_AIR: u16 = 126;
pub const CONTENT_IGNORE: u16 = 127;
pub const RAM_MIC_ITEM_ID: &str = "cubex:ram_mic";
pub const NETHER_PORTAL_NODE_ID: &str = "cubex:nether_portal";

/// Returns `true` for standard terrain nodes (`dirt`, `grass`, `snow`, `sand`, `stone`, `wood`,
/// `leaves`, `water`, etc.) so they never inadvertently deal `damage_per_second` contact damage.
pub fn is_safe_terrain_node(name: &str, groups: &HashMap<String, i32>) -> bool {
    let lower = name.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }
    const HAZARD_KEYWORDS: &[&str] = &[
        "lava",
        "fire",
        "campfire",
        "magma",
        "cactus",
        "sweet_berry",
        "wither_rose",
        "acid",
        "spike",
        "poison",
        "powder_snow",
    ];
    if HAZARD_KEYWORDS.iter().any(|kw| lower.contains(kw)) {
        return false;
    }
    const SAFE_KEYWORDS: &[&str] = &[
        "dirt", "grass", "snow", "sand", "stone", "cobble", "wood", "tree", "plank", "leaves",
        "water", "gravel", "clay", "ice",
    ];
    if SAFE_KEYWORDS.iter().any(|kw| lower.contains(kw)) {
        return true;
    }
    ["soil", "sand", "stone", "wood", "tree", "leaves", "water"]
        .iter()
        .any(|g| groups.get(*g).copied().unwrap_or(0) > 0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawType {
    Normal,
    Airlike,
    Allfaces,
    AllfacesOptional,
    Glasslike,
    GlasslikeFramed,
    GlasslikeFramedOptional,
    Plantlike,
    Torchlike,
    Liquid,
    FlowingLiquid,
    NodeBox,
    Mesh,
    Signlike,
    Raillike,
    Fencelike,
    Firelike,
}

impl DrawType {
    pub fn from_str(s: &str) -> Self {
        match s {
            "airlike" => Self::Airlike,
            "allfaces" => Self::Allfaces,
            "allfaces_optional" => Self::AllfacesOptional,
            "glasslike" => Self::Glasslike,
            "glasslike_framed" => Self::GlasslikeFramed,
            "glasslike_framed_optional" => Self::GlasslikeFramedOptional,
            "plantlike" => Self::Plantlike,
            "torchlike" => Self::Torchlike,
            "liquid" => Self::Liquid,
            "flowingliquid" => Self::FlowingLiquid,
            "nodebox" => Self::NodeBox,
            "mesh" => Self::Mesh,
            "signlike" => Self::Signlike,
            "raillike" => Self::Raillike,
            "fencelike" => Self::Fencelike,
            "firelike" => Self::Firelike,
            _ => Self::Normal,
        }
    }

    pub fn is_solid_cube(self) -> bool {
        matches!(self, Self::Normal)
    }

    pub fn is_transparent(self) -> bool {
        !matches!(self, Self::Normal)
    }
}

#[derive(Debug, Clone)]
pub struct TileSpec {
    pub name: String,
    pub color: Option<[u8; 4]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SoundSpec {
    pub name: String,
    pub gain: f32,
    pub pitch: f32,
}

impl Default for SoundSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            gain: 1.0,
            pitch: 1.0,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct NodeSounds {
    pub footstep: Option<SoundSpec>,
    pub dig: Option<SoundSpec>,
    pub dug: Option<SoundSpec>,
    pub place: Option<SoundSpec>,
    pub fall: Option<SoundSpec>,
}

#[derive(Debug, Clone)]
pub struct NodeBoxSpec {
    pub box_type: String,
    pub fixed: Vec<[f32; 6]>,
    pub wall_top: Option<[f32; 6]>,
    pub wall_bottom: Option<[f32; 6]>,
    pub wall_side: Option<[f32; 6]>,
    pub connect_top: Vec<[f32; 6]>,
    pub connect_bottom: Vec<[f32; 6]>,
    pub connect_front: Vec<[f32; 6]>,
    pub connect_back: Vec<[f32; 6]>,
    pub connect_left: Vec<[f32; 6]>,
    pub connect_right: Vec<[f32; 6]>,
    pub disconnected_top: Vec<[f32; 6]>,
    pub disconnected_bottom: Vec<[f32; 6]>,
    pub disconnected_front: Vec<[f32; 6]>,
    pub disconnected_back: Vec<[f32; 6]>,
    pub disconnected_left: Vec<[f32; 6]>,
    pub disconnected_right: Vec<[f32; 6]>,
    pub disconnected_sides: Vec<[f32; 6]>,
    pub disconnected_all: Vec<[f32; 6]>,
}

impl Default for NodeBoxSpec {
    fn default() -> Self {
        Self {
            box_type: "regular".into(),
            fixed: Vec::new(),
            wall_top: None,
            wall_bottom: None,
            wall_side: None,
            connect_top: Vec::new(),
            connect_bottom: Vec::new(),
            connect_front: Vec::new(),
            connect_back: Vec::new(),
            connect_left: Vec::new(),
            connect_right: Vec::new(),
            disconnected_top: Vec::new(),
            disconnected_bottom: Vec::new(),
            disconnected_front: Vec::new(),
            disconnected_back: Vec::new(),
            disconnected_left: Vec::new(),
            disconnected_right: Vec::new(),
            disconnected_sides: Vec::new(),
            disconnected_all: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct DropItemEntry {
    pub items: Vec<String>,
    pub rarity: i32,
    pub tools: Vec<String>,
    pub tool_groups: Vec<Vec<String>>,
    pub inherit_color: bool,
}

#[derive(Debug, Clone, Default)]
pub enum NodeDrop {
    #[default]
    Default,
    Empty,
    Single(String),
    Table {
        max_items: Option<usize>,
        items: Vec<DropItemEntry>,
    },
}

impl NodeDrop {
    /// Resolve dropped itemstrings for a node using Luanti's rarity & tool/tool_group matching rules.
    pub fn resolve_drops(
        &self,
        node_name: &str,
        tool_name: Option<&str>,
        tool_groups: Option<&HashMap<String, i32>>,
    ) -> Vec<String> {
        match self {
            NodeDrop::Default => vec![node_name.to_string()],
            NodeDrop::Empty => Vec::new(),
            NodeDrop::Single(s) => {
                if s.trim().is_empty() {
                    Vec::new()
                } else {
                    vec![s.clone()]
                }
            }
            NodeDrop::Table { max_items, items } => {
                let mut out = Vec::new();
                let mut got_count = 0usize;
                for entry in items {
                    let good_rarity = if entry.rarity > 1 {
                        rand::random::<u32>() % (entry.rarity as u32) == 0
                    } else {
                        true
                    };
                    let mut good_tool = entry.tools.is_empty() && entry.tool_groups.is_empty();
                    if !entry.tools.is_empty() {
                        if let Some(tn) = tool_name {
                            for pat in &entry.tools {
                                if let Some(sub) = pat.strip_prefix('~') {
                                    if tn.contains(sub) {
                                        good_tool = true;
                                        break;
                                    }
                                } else if tn == pat {
                                    good_tool = true;
                                    break;
                                }
                            }
                        }
                    }
                    if !good_tool && !entry.tool_groups.is_empty() {
                        if let Some(tg) = tool_groups {
                            for req in &entry.tool_groups {
                                if !req.is_empty()
                                    && req.iter().all(|g| tg.get(g).copied().unwrap_or(0) != 0)
                                {
                                    good_tool = true;
                                    break;
                                }
                            }
                        }
                    }
                    if good_rarity && good_tool {
                        got_count += 1;
                        out.extend(entry.items.iter().cloned());
                        if let Some(max_n) = max_items {
                            if got_count >= *max_n {
                                break;
                            }
                        }
                    }
                }
                out
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ToolGroupCap {
    pub times: HashMap<i32, f32>,
    pub uses: i32,
    pub maxlevel: i32,
}

#[derive(Debug, Clone)]
pub struct ToolCapabilities {
    pub full_punch_interval: f32,
    pub max_drop_level: i32,
    pub groupcaps: HashMap<String, ToolGroupCap>,
    pub damage_groups: HashMap<String, i32>,
    pub punch_attack_uses: i32,
}

impl Default for ToolCapabilities {
    fn default() -> Self {
        Self {
            full_punch_interval: 1.4,
            max_drop_level: 0,
            groupcaps: HashMap::new(),
            damage_groups: HashMap::new(),
            punch_attack_uses: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DigParams {
    pub diggable: bool,
    pub time: f32,
    pub wear: u16,
}

/// Port of Luanti's `calculateResultWear` from `src/tool.cpp`.
pub fn calculate_result_wear(uses: u32, initial_wear: u16) -> u16 {
    if uses == 0 {
        return 0;
    }
    let wear_normal = 65536u32 / uses;
    let blocks_oversize = (65536u32 % uses) as u16;
    let mut wear_extra = 0u32;
    if blocks_oversize > 0 {
        let blocks_normal = (uses as u16).saturating_sub(blocks_oversize) as u32;
        let wear_extra_at = blocks_normal * wear_normal;
        if (initial_wear as u32) >= wear_extra_at {
            wear_extra = 1;
        }
    }
    (wear_normal + wear_extra).min(65535) as u16
}

/// 1:1 Rust port of Luanti's `getDigParams` (`src/tool.cpp`), including `dig_immediate`,
/// `maxlevel`, `level`, `leveldiff`, and wear calculation.
pub fn compute_dig_params(
    groups: &HashMap<String, i32>,
    tool: Option<&ToolCapabilities>,
    creative: bool,
) -> DigParams {
    compute_dig_params_with_wear(groups, tool, creative, 0)
}

pub fn compute_dig_params_with_wear(
    groups: &HashMap<String, i32>,
    tool: Option<&ToolCapabilities>,
    creative: bool,
    initial_wear: u16,
) -> DigParams {
    if creative {
        return DigParams {
            diggable: true,
            time: 0.0,
            wear: 0,
        };
    }

    let has_dig_immediate_cap = tool
        .map(|t| t.groupcaps.contains_key("dig_immediate"))
        .unwrap_or(false);
    if !has_dig_immediate_cap {
        match groups.get("dig_immediate").copied().unwrap_or(0) {
            2 => {
                return DigParams {
                    diggable: true,
                    time: 0.5,
                    wear: 0,
                };
            }
            3 => {
                return DigParams {
                    diggable: true,
                    time: 0.0,
                    wear: 0,
                };
            }
            _ => {}
        }
    }

    let Some(tp) = tool else {
        return DigParams {
            diggable: false,
            time: 0.0,
            wear: 0,
        };
    };

    let level = groups.get("level").copied().unwrap_or(0);
    let mut result_diggable = false;
    let mut result_time = 0.0f32;
    let mut result_wear = 0u16;

    for (groupname, cap) in &tp.groupcaps {
        let leveldiff = cap.maxlevel - level;
        if leveldiff < 0 {
            continue;
        }
        let rating = groups.get(groupname).copied().unwrap_or(0);
        let Some(&base_time) = cap.times.get(&rating) else {
            continue;
        };
        let mut time = base_time;
        if leveldiff > 1 {
            time /= leveldiff as f32;
        }
        if !result_diggable || time < result_time {
            result_time = time;
            result_diggable = true;
            if cap.uses <= 0 {
                result_wear = 0;
            } else {
                let real_uses = ((cap.uses as f64) * 3.0f64.powi(leveldiff))
                    .clamp(1.0, 65535.0) as u32;
                result_wear = calculate_result_wear(real_uses, initial_wear);
            }
        }
    }

    DigParams {
        diggable: result_diggable,
        time: result_time,
        wear: result_wear,
    }
}

#[derive(Debug, Clone)]
pub struct NodeDef {
    pub id: u16,
    pub name: String,
    pub description: String,
    pub drawtype: DrawType,
    pub walkable: bool,
    pub pointable: bool,
    pub diggable: bool,
    pub climbable: bool,
    pub buildable_to: bool,
    pub sunlight_propagates: bool,
    pub light_source: u8,
    pub paramtype: String,
    pub paramtype2: String,
    pub is_ground_content: bool,
    pub use_texture_alpha: String,
    pub tiles: Vec<TileSpec>,
    pub overlay_tiles: Vec<TileSpec>,
    pub node_boxes: Vec<[f32; 6]>,
    pub node_box_type: String,
    pub node_box_wall_top: Option<[f32; 6]>,
    pub node_box_wall_bottom: Option<[f32; 6]>,
    pub node_box_wall_side: Option<[f32; 6]>,
    pub connects_to: Vec<String>,
    pub connect_sides: Vec<String>,
    pub connect_top: Vec<[f32; 6]>,
    pub connect_bottom: Vec<[f32; 6]>,
    pub connect_front: Vec<[f32; 6]>,
    pub connect_back: Vec<[f32; 6]>,
    pub connect_left: Vec<[f32; 6]>,
    pub connect_right: Vec<[f32; 6]>,
    pub disconnected_top: Vec<[f32; 6]>,
    pub disconnected_bottom: Vec<[f32; 6]>,
    pub disconnected_front: Vec<[f32; 6]>,
    pub disconnected_back: Vec<[f32; 6]>,
    pub disconnected_left: Vec<[f32; 6]>,
    pub disconnected_right: Vec<[f32; 6]>,
    pub disconnected_sides: Vec<[f32; 6]>,
    pub disconnected_all: Vec<[f32; 6]>,
    pub selection_boxes: Vec<[f32; 6]>,
    pub collision_boxes: Vec<[f32; 6]>,
    pub selection_box: NodeBoxSpec,
    pub collision_box: NodeBoxSpec,
    pub mesh: Option<String>,
    pub groups: HashMap<String, i32>,
    pub color: Option<[u8; 4]>,
    pub palette: Option<String>,
    pub sounds: NodeSounds,
    pub drop: NodeDrop,
    pub liquidtype: String,
    pub liquid_viscosity: u8,
    pub liquid_alternative_flowing: String,
    pub liquid_alternative_source: String,
    pub floodable: bool,
    pub damage_per_second: i32,
    pub waving: u8,
    pub leveled: u8,
    pub visual_scale: f32,
    pub liquids_pointable: bool,
    pub inventory_image: String,
    pub wield_image: String,
    pub stack_max: u16,
}

impl NodeDef {
    /// Returns true only for fully opaque full-cube occluders.
    /// Non-normal drawtypes (leaves, glass, plants, liquids, nodeboxes), cutout/blended textures,
    /// air/ignore, and leaves/glass nodes return false so adjacent faces are never culled.
    pub fn is_opaque(&self) -> bool {
        if self.id == CONTENT_AIR || self.id == CONTENT_IGNORE {
            return false;
        }
        if self.drawtype != DrawType::Normal || self.use_texture_alpha != "opaque" {
            return false;
        }
        let lower = self.name.to_ascii_lowercase();
        !lower.contains("leaves") && !lower.contains("glass")
    }

    pub fn node_box_spec(&self) -> NodeBoxSpec {
        NodeBoxSpec {
            box_type: self.node_box_type.clone(),
            fixed: self.node_boxes.clone(),
            wall_top: self.node_box_wall_top,
            wall_bottom: self.node_box_wall_bottom,
            wall_side: self.node_box_wall_side,
            connect_top: self.connect_top.clone(),
            connect_bottom: self.connect_bottom.clone(),
            connect_front: self.connect_front.clone(),
            connect_back: self.connect_back.clone(),
            connect_left: self.connect_left.clone(),
            connect_right: self.connect_right.clone(),
            disconnected_top: self.disconnected_top.clone(),
            disconnected_bottom: self.disconnected_bottom.clone(),
            disconnected_front: self.disconnected_front.clone(),
            disconnected_back: self.disconnected_back.clone(),
            disconnected_left: self.disconnected_left.clone(),
            disconnected_right: self.disconnected_right.clone(),
            disconnected_sides: self.disconnected_sides.clone(),
            disconnected_all: self.disconnected_all.clone(),
        }
    }
}

impl Default for NodeDef {
    fn default() -> Self {
        Self {
            id: CONTENT_UNKNOWN,
            name: "unknown".into(),
            description: String::new(),
            drawtype: DrawType::Normal,
            walkable: true,
            pointable: true,
            diggable: true,
            climbable: false,
            buildable_to: false,
            sunlight_propagates: false,
            light_source: 0,
            paramtype: "none".into(),
            paramtype2: "none".into(),
            is_ground_content: true,
            use_texture_alpha: "opaque".into(),
            tiles: Vec::new(),
            overlay_tiles: Vec::new(),
            node_boxes: Vec::new(),
            node_box_type: "regular".into(),
            node_box_wall_top: None,
            node_box_wall_bottom: None,
            node_box_wall_side: None,
            connects_to: Vec::new(),
            connect_sides: Vec::new(),
            connect_top: Vec::new(),
            connect_bottom: Vec::new(),
            connect_front: Vec::new(),
            connect_back: Vec::new(),
            connect_left: Vec::new(),
            connect_right: Vec::new(),
            disconnected_top: Vec::new(),
            disconnected_bottom: Vec::new(),
            disconnected_front: Vec::new(),
            disconnected_back: Vec::new(),
            disconnected_left: Vec::new(),
            disconnected_right: Vec::new(),
            disconnected_sides: Vec::new(),
            disconnected_all: Vec::new(),
            selection_boxes: Vec::new(),
            collision_boxes: Vec::new(),
            selection_box: NodeBoxSpec::default(),
            collision_box: NodeBoxSpec::default(),
            mesh: None,
            groups: HashMap::new(),
            color: None,
            palette: None,
            sounds: NodeSounds::default(),
            drop: NodeDrop::Default,
            liquidtype: "none".into(),
            liquid_viscosity: 0,
            liquid_alternative_flowing: String::new(),
            liquid_alternative_source: String::new(),
            floodable: false,
            damage_per_second: 0,
            waving: 0,
            leveled: 0,
            visual_scale: 1.0,
            liquids_pointable: false,
            inventory_image: String::new(),
            wield_image: String::new(),
            stack_max: 64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ItemDef {
    pub name: String,
    pub item_type: String,
    pub description: String,
    pub inventory_image: String,
    pub wield_image: String,
    pub stack_max: u16,
    pub groups: HashMap<String, i32>,
    pub is_node: bool,
    pub tool_capabilities: Option<ToolCapabilities>,
    pub range: Option<f32>,
    pub liquids_pointable: bool,
}

impl Default for ItemDef {
    fn default() -> Self {
        Self {
            name: String::new(),
            item_type: "none".into(),
            description: String::new(),
            inventory_image: String::new(),
            wield_image: String::new(),
            stack_max: 64,
            groups: HashMap::new(),
            is_node: false,
            tool_capabilities: None,
            range: None,
            liquids_pointable: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CraftRecipe {
    pub recipe_type: String,
    pub output: String,
    pub recipe: Vec<Vec<String>>,
    pub replacements: Vec<(String, String)>,
    pub cooktime: f32,
    pub burntime: f32,
}

#[derive(Debug, Clone)]
pub struct BiomeDef {
    pub id: u32,
    pub name: String,
    pub node_top: String,
    pub depth_top: i32,
    pub node_filler: String,
    pub depth_filler: i32,
    pub node_stone: String,
    pub node_water_top: String,
    pub y_min: i32,
    pub y_max: i32,
    pub heat_point: f32,
    pub humidity_point: f32,
}

#[derive(Debug, Clone)]
pub struct OreDef {
    pub ore_type: String,
    pub ore: String,
    pub wherein: Vec<String>,
    pub clust_scarcity: i32,
    pub clust_num_ores: i32,
    pub clust_size: i32,
    pub y_min: i32,
    pub y_max: i32,
}

#[derive(Debug, Clone)]
pub struct DecorationDef {
    pub deco_type: String,
    pub name: String,
    pub place_on: Vec<String>,
    pub fill_ratio: f32,
    pub y_min: i32,
    pub y_max: i32,
    pub decoration: Vec<String>,
    pub schematic: Option<String>,
    pub flags: String,
    pub rotation: String,
}

#[derive(Debug, Clone)]
pub struct EntityDef {
    pub name: String,
    pub hp_max: i32,
    pub physical: bool,
    pub visual: String,
    pub textures: Vec<String>,
    pub collisionbox: [f32; 6],
}

/// Central registry holding all Mineclonia nodes, items, aliases, crafts, biomes, ores, decorations, and entities.
pub struct EngineRegistries {
    pub nodes_by_id: Vec<Option<NodeDef>>,
    pub name_to_id: HashMap<String, u16>,
    pub aliases: HashMap<String, String>,
    pub items: HashMap<String, ItemDef>,
    pub crafts: Vec<CraftRecipe>,
    pub biomes: Vec<BiomeDef>,
    pub ores: Vec<OreDef>,
    pub decorations: Vec<DecorationDef>,
    pub entities: HashMap<String, EntityDef>,
    next_id: u16,
}

impl Default for EngineRegistries {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineRegistries {
    pub fn new() -> Self {
        let mut reg = Self {
            nodes_by_id: vec![None; 256],
            name_to_id: HashMap::new(),
            aliases: HashMap::new(),
            items: HashMap::new(),
            crafts: Vec::new(),
            biomes: Vec::new(),
            ores: Vec::new(),
            decorations: Vec::new(),
            entities: HashMap::new(),
            next_id: 0,
        };

        // Pre-register Luanti built-in special content IDs
        reg.insert_fixed_node(
            CONTENT_UNKNOWN,
            NodeDef {
                id: CONTENT_UNKNOWN,
                name: "unknown".into(),
                drawtype: DrawType::Normal,
                walkable: true,
                tiles: vec![TileSpec {
                    name: "unknown_node.png".into(),
                    color: None,
                }],
                ..Default::default()
            },
        );
        reg.insert_fixed_node(
            CONTENT_AIR,
            NodeDef {
                id: CONTENT_AIR,
                name: "air".into(),
                drawtype: DrawType::Airlike,
                walkable: false,
                pointable: false,
                diggable: false,
                buildable_to: true,
                floodable: true,
                sunlight_propagates: true,
                is_ground_content: false,
                drop: NodeDrop::Empty,
                ..Default::default()
            },
        );
        reg.insert_fixed_node(
            CONTENT_IGNORE,
            NodeDef {
                id: CONTENT_IGNORE,
                name: "ignore".into(),
                drawtype: DrawType::Airlike,
                walkable: false,
                pointable: false,
                diggable: false,
                buildable_to: false,
                sunlight_propagates: false,
                is_ground_content: false,
                drop: NodeDrop::Empty,
                ..Default::default()
            },
        );
        reg.register_cubex_builtin_items();
        reg
    }

    pub fn register_cubex_builtin_items(&mut self) {
        if !self.items.contains_key(RAM_MIC_ITEM_ID) {
            self.register_item(ItemDef {
                name: RAM_MIC_ITEM_ID.into(),
                item_type: "craft".into(),
                description: "RAM Stick Mic".into(),
                inventory_image: "cubex_ram_mic.png".into(),
                wield_image: "cubex_ram_mic.png".into(),
                stack_max: 1,
                groups: HashMap::from([("cubex_prop".into(), 1)]),
                is_node: false,
                ..Default::default()
            });
        }
        if !self.name_to_id.contains_key(NETHER_PORTAL_NODE_ID) {
            self.register_node(NodeDef {
                name: NETHER_PORTAL_NODE_ID.into(),
                description: "Nether Portal".into(),
                drawtype: DrawType::NodeBox,
                walkable: false,
                pointable: true,
                diggable: true,
                climbable: false,
                buildable_to: false,
                sunlight_propagates: true,
                light_source: 11,
                paramtype: "light".into(),
                paramtype2: "facedir".into(),
                is_ground_content: false,
                use_texture_alpha: "blend".into(),
                tiles: vec![TileSpec {
                    name: "cubex_nether_portal.png".into(),
                    color: None,
                }],
                node_box_type: "fixed".into(),
                node_boxes: vec![[-0.5, -0.5, -0.125, 0.5, 0.5, 0.125]],
                selection_boxes: vec![[-0.5, -0.5, -0.125, 0.5, 0.5, 0.125]],
                collision_boxes: Vec::new(),
                groups: HashMap::from([("dig_immediate".into(), 3), ("portal".into(), 1)]),
                drop: NodeDrop::Empty,
                floodable: false,
                waving: 0,
                inventory_image: "cubex_nether_portal.png".into(),
                wield_image: "cubex_nether_portal.png".into(),
                stack_max: 64,
                ..Default::default()
            });
        }
    }

    fn insert_fixed_node(&mut self, id: u16, def: NodeDef) {
        let idx = id as usize;
        if idx >= self.nodes_by_id.len() {
            self.nodes_by_id.resize(idx + 1, None);
        }
        self.name_to_id.insert(def.name.clone(), id);
        self.nodes_by_id[idx] = Some(def);
    }

    fn alloc_id(&mut self) -> u16 {
        while matches!(self.next_id, CONTENT_UNKNOWN | CONTENT_AIR | CONTENT_IGNORE) {
            self.next_id += 1;
        }
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        id
    }

    pub fn register_node(&mut self, mut def: NodeDef) -> u16 {
        if is_safe_terrain_node(&def.name, &def.groups) {
            def.damage_per_second = 0;
        }
        if let Some(&existing) = self.name_to_id.get(&def.name) {
            def.id = existing;
            self.nodes_by_id[existing as usize] = Some(def);
            return existing;
        }
        let id = self.alloc_id();
        def.id = id;
        let idx = id as usize;
        if idx >= self.nodes_by_id.len() {
            self.nodes_by_id.resize(idx + 256, None);
        }
        self.name_to_id.insert(def.name.clone(), id);
        self.nodes_by_id[idx] = Some(def);
        id
    }

    pub fn register_item(&mut self, item: ItemDef) {
        self.items.insert(item.name.clone(), item);
    }

    pub fn unregister_item(&mut self, name: &str) {
        self.items.remove(name);
        if let Some(id) = self.name_to_id.remove(name) {
            if let Some(slot) = self.nodes_by_id.get_mut(id as usize) {
                *slot = None;
            }
        }
    }

    pub fn register_alias(&mut self, alias: impl Into<String>, target: impl Into<String>) {
        self.aliases.insert(alias.into(), target.into());
    }

    pub fn resolve_alias<'a>(&'a self, mut name: &'a str) -> &'a str {
        for _ in 0..16 {
            match self.aliases.get(name) {
                Some(next) if next != name => name = next.as_str(),
                _ => break,
            }
        }
        name
    }

    pub fn get_content_id(&self, name: &str) -> u16 {
        if let Some(&id) = self.name_to_id.get(name) {
            return id;
        }
        let resolved = self.resolve_alias(name);
        self.name_to_id.get(resolved).copied().unwrap_or(CONTENT_UNKNOWN)
    }

    pub fn get_node_def(&self, id: u16) -> Option<&NodeDef> {
        self.nodes_by_id.get(id as usize).and_then(|o| o.as_ref())
    }

    pub fn get_item_def(&self, name: &str) -> Option<&ItemDef> {
        if let Some(d) = self.items.get(name) {
            return Some(d);
        }
        let resolved = self.resolve_alias(name);
        self.items.get(resolved)
    }

    pub fn get_effective_tool_capabilities(
        &self,
        held_item_name: Option<&str>,
        hand_item_name: Option<&str>,
    ) -> Option<&ToolCapabilities> {
        if let Some(name) = held_item_name.filter(|s| !s.is_empty()) {
            if let Some(tc) = self
                .get_item_def(name)
                .and_then(|d| d.tool_capabilities.as_ref())
            {
                return Some(tc);
            }
        }
        if let Some(hand_name) = hand_item_name.filter(|s| !s.is_empty()) {
            if let Some(tc) = self
                .get_item_def(hand_name)
                .and_then(|d| d.tool_capabilities.as_ref())
            {
                return Some(tc);
            }
        }
        self.items
            .get("")
            .and_then(|d| d.tool_capabilities.as_ref())
    }

    pub fn get_name_from_content_id(&self, id: u16) -> &str {
        self.get_node_def(id)
            .map(|d| d.name.as_str())
            .unwrap_or("unknown")
    }

    pub fn node_count(&self) -> usize {
        self.name_to_id.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_def_is_opaque_for_leaves_glass_plants_and_stone() {
        let stone = NodeDef {
            id: 200,
            name: "mcl_core:stone".into(),
            drawtype: DrawType::Normal,
            use_texture_alpha: "opaque".into(),
            ..Default::default()
        };
        assert!(stone.is_opaque(), "opaque Normal stone should be an occluder");

        let leaves = NodeDef {
            id: 201,
            name: "mcl_trees:leaves_oak".into(),
            drawtype: DrawType::AllfacesOptional,
            use_texture_alpha: "clip".into(),
            ..Default::default()
        };
        assert!(!leaves.is_opaque(), "leaves must not occlude neighbor faces");

        let glass = NodeDef {
            id: 202,
            name: "mcl_core:glass".into(),
            drawtype: DrawType::Glasslike,
            use_texture_alpha: "clip".into(),
            ..Default::default()
        };
        assert!(!glass.is_opaque(), "glass must not occlude neighbor faces");

        let tallgrass = NodeDef {
            id: 203,
            name: "mcl_flowers:tallgrass".into(),
            drawtype: DrawType::Plantlike,
            use_texture_alpha: "clip".into(),
            ..Default::default()
        };
        assert!(!tallgrass.is_opaque(), "plants must not occlude neighbor faces");

        let opaque_named_leaves = NodeDef {
            id: 204,
            name: "mcl_trees:leaves_spruce".into(),
            drawtype: DrawType::Normal,
            use_texture_alpha: "opaque".into(),
            ..Default::default()
        };
        assert!(!opaque_named_leaves.is_opaque());
    }
}
