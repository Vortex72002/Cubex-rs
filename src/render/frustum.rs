use glam::{Mat4, Vec3, Vec4};

/// Six normalized view-frustum planes for fast per-chunk AABB culling.
#[derive(Debug, Clone)]
pub struct Frustum {
    pub planes: [Vec4; 6],
}

impl Frustum {
    pub fn from_view_proj(m: Mat4) -> Self {
        let r0 = m.row(0);
        let r1 = m.row(1);
        let r2 = m.row(2);
        let r3 = m.row(3);

        let raw = [
            r3 + r0, // left
            r3 - r0, // right
            r3 + r1, // bottom
            r3 - r1, // top
            r2,      // near (wgpu 0..1 NDC depth)
            r3 - r2, // far
        ];

        let mut planes = [Vec4::ZERO; 6];
        for (i, p) in raw.iter().enumerate() {
            let len = p.truncate().length();
            planes[i] = if len > 1e-6 { *p / len } else { *p };
        }
        Self { planes }
    }

    #[inline]
    pub fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        for p in &self.planes {
            let px = if p.x >= 0.0 { max.x } else { min.x };
            let py = if p.y >= 0.0 { max.y } else { min.y };
            let pz = if p.z >= 0.0 { max.z } else { min.z };
            if p.x * px + p.y * py + p.z * pz + p.w < 0.0 {
                return false;
            }
        }
        true
    }
}
