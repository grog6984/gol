use egui::Color32;

pub const PALETTE_SIZE: usize = 512;

#[derive(Clone, Debug, PartialEq)]
pub struct Palette {
    pub name: String,
    pub stops: Vec<(f32, Color32)>, // t in [0,1], sorted
}

impl Palette {
    pub fn new(name: &str, stops: &[(f32, Color32)]) -> Self {
        let mut stops = stops.to_vec();
        stops.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        Self {
            name: name.to_string(),
            stops,
        }
    }

    pub fn sample(&self, t: f32) -> Color32 {
        let t = t.clamp(0.0, 1.0);
        if self.stops.is_empty() {
            return Color32::BLACK;
        }
        if t <= self.stops[0].0 {
            return self.stops[0].1;
        }
        if t >= self.stops.last().unwrap().0 {
            return self.stops.last().unwrap().1;
        }
        for win in self.stops.windows(2) {
            let (t0, c0) = win[0];
            let (t1, c1) = win[1];
            if t >= t0 && t <= t1 {
                let u = if t1 == t0 { 0.0 } else { (t - t0) / (t1 - t0) };
                return lerp_color(c0, c1, u);
            }
        }
        self.stops.last().unwrap().1
    }

    pub fn build_rgba8(&self, out: &mut [u8]) {
        let n = out.len() / 4;
        for i in 0..n {
            let t = if n > 1 {
                i as f32 / (n - 1) as f32
            } else {
                0.0
            };
            let c = self.sample(t);
            out[i * 4 + 0] = c.r();
            out[i * 4 + 1] = c.g();
            out[i * 4 + 2] = c.b();
            out[i * 4 + 3] = c.a();
        }
    }
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let u = 1.0 - t;
    Color32::from_rgba_premultiplied(
        (a.r() as f32 * u + b.r() as f32 * t) as u8,
        (a.g() as f32 * u + b.g() as f32 * t) as u8,
        (a.b() as f32 * u + b.b() as f32 * t) as u8,
        (a.a() as f32 * u + b.a() as f32 * t) as u8,
    )
}

pub fn presets() -> Vec<Palette> {
    vec![
        Palette::new(
            "Classic",
            &[
                (0.0, Color32::BLACK),
                (0.05, Color32::from_rgb(50, 100, 255)),
                (0.3, Color32::from_rgb(0, 220, 120)),
                (0.7, Color32::from_rgb(255, 220, 0)),
                (1.0, Color32::from_rgb(255, 50, 50)),
            ],
        ),
        Palette::new(
            "Neon",
            &[
                (0.0, Color32::BLACK),
                (0.1, Color32::from_rgb(20, 0, 60)),
                (0.4, Color32::from_rgb(180, 0, 255)),
                (0.7, Color32::from_rgb(0, 255, 255)),
                (1.0, Color32::from_rgb(255, 255, 255)),
            ],
        ),
        Palette::new(
            "Ocean",
            &[
                (0.0, Color32::BLACK),
                (0.2, Color32::from_rgb(0, 40, 80)),
                (0.5, Color32::from_rgb(0, 120, 180)),
                (0.8, Color32::from_rgb(0, 255, 220)),
                (1.0, Color32::from_rgb(220, 255, 255)),
            ],
        ),
        Palette::new(
            "Fire",
            &[
                (0.0, Color32::BLACK),
                (0.2, Color32::from_rgb(60, 0, 0)),
                (0.5, Color32::from_rgb(180, 40, 0)),
                (0.8, Color32::from_rgb(255, 160, 0)),
                (1.0, Color32::from_rgb(255, 255, 220)),
            ],
        ),
        Palette::new(
            "Plasma",
            &[
                (0.0, Color32::from_rgb(20, 0, 40)),
                (0.25, Color32::from_rgb(120, 0, 160)),
                (0.5, Color32::from_rgb(255, 0, 120)),
                (0.75, Color32::from_rgb(255, 200, 0)),
                (1.0, Color32::from_rgb(255, 255, 255)),
            ],
        ),
        Palette::new("Grey", &[(0.0, Color32::BLACK), (1.0, Color32::WHITE)]),
        Palette::new(
            "Binary",
            &[
                (0.0, Color32::BLACK),
                (0.001, Color32::from_rgb(220, 220, 220)),
                (1.0, Color32::from_rgb(220, 220, 220)),
            ],
        ),
    ]
}
