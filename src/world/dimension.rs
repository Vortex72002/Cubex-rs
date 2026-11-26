use glam::IVec3;
use serde::{Deserialize, Serialize};

/// Active world dimension identifier. Each dimension owns independent chunks, lighting, and streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum DimensionId {
    #[default]
    Overworld,
    Nether,
}

impl DimensionId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Overworld => "overworld",
            Self::Nether => "nether",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Overworld => "Overworld",
            Self::Nether => "The Nether",
        }
    }

    pub fn subdir(self) -> &'static str {
        self.as_str()
    }

    pub fn has_skylight(self) -> bool {
        match self {
            Self::Overworld => true,
            Self::Nether => false,
        }
    }

    pub fn is_nether(self) -> bool {
        self == Self::Nether
    }

    pub fn opposite(self) -> Self {
        match self {
            Self::Overworld => Self::Nether,
            Self::Nether => Self::Overworld,
        }
    }
}

/// Converts portal world coordinates between dimensions using authentic 1:8 Nether scaling.
pub fn scale_portal_coords(pos: IVec3, from: DimensionId, to: DimensionId) -> IVec3 {
    match (from, to) {
        (DimensionId::Overworld, DimensionId::Nether) => IVec3::new(
            pos.x.div_euclid(8),
            pos.y.clamp(36, 116),
            pos.z.div_euclid(8),
        ),
        (DimensionId::Nether, DimensionId::Overworld) => IVec3::new(
            pos.x.saturating_mul(8),
            pos.y.clamp(64, 120),
            pos.z.saturating_mul(8),
        ),
        _ => pos,
    }
}
