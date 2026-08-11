/// Pattern expressed as a list of (x, y) live cells with an origin.
#[derive(Clone, Debug, Default)]
pub struct Pattern {
    pub name: String,
    pub cells: Vec<(i32, i32)>,
    pub width: i32,
    pub height: i32,
}

impl Pattern {
    pub fn new(name: &str, cells: &[(i32, i32)]) -> Self {
        let width = cells.iter().map(|c| c.0 + 1).max().unwrap_or(0);
        let height = cells.iter().map(|c| c.1 + 1).max().unwrap_or(0);
        Self {
            name: name.to_string(),
            cells: cells.to_vec(),
            width,
            height,
        }
    }

    pub fn parse_rle(name: &str, rle: &str, width: i32, height: i32) -> Self {
        let mut cells = Vec::new();
        let mut x = 0i32;
        let mut y = 0i32;
        let mut count_buf = String::new();
        for line in rle.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            for ch in line.chars() {
                match ch {
                    '0'..='9' => count_buf.push(ch),
                    'b' => {
                        let count = count_buf.parse::<i32>().unwrap_or(1);
                        x += count;
                        count_buf.clear();
                    }
                    'o' => {
                        let count = count_buf.parse::<i32>().unwrap_or(1);
                        for _ in 0..count {
                            cells.push((x, y));
                            x += 1;
                        }
                        count_buf.clear();
                    }
                    '$' => {
                        let count = count_buf.parse::<i32>().unwrap_or(1);
                        y += count;
                        x = 0;
                        count_buf.clear();
                    }
                    '!' => break,
                    _ => {}
                }
            }
        }
        Self {
            name: name.to_string(),
            cells,
            width,
            height,
        }
    }

    pub fn translate(&self, dx: i32, dy: i32) -> Self {
        let mut p = self.clone();
        for (x, y) in &mut p.cells {
            *x += dx;
            *y += dy;
        }
        p
    }

    pub fn centered(&self) -> Self {
        let min_x = self.cells.iter().map(|c| c.0).min().unwrap_or(0);
        let min_y = self.cells.iter().map(|c| c.1).min().unwrap_or(0);
        self.translate(-min_x - self.width / 2, -min_y - self.height / 2)
    }
}

pub fn presets() -> Vec<Pattern> {
    vec![
        Pattern::new("Empty", &[]),
        Pattern::new("Glider", &[(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)]),
        Pattern::new(
            "LWSS",
            &[
                (1, 0),
                (4, 0),
                (0, 1),
                (0, 2),
                (4, 2),
                (0, 3),
                (1, 3),
                (2, 3),
                (3, 3),
            ],
        ),
        Pattern::new(
            "MWSS",
            &[
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (1, 1),
                (5, 1),
                (5, 2),
                (0, 2),
                (5, 3),
                (0, 3),
                (2, 4),
                (3, 4),
                (4, 4),
                (5, 4),
            ],
        ),
        Pattern::parse_rle("HWSS", "3b5o$o2b5o$7o$o6bo$o!", 8, 5),
        Pattern::new("R-pentomino", &[(1, 0), (2, 0), (0, 1), (1, 1), (1, 2)]),
        Pattern::new(
            "Diehard",
            &[(6, 0), (0, 1), (1, 1), (1, 2), (5, 2), (6, 2), (7, 2)],
        ),
        Pattern::new(
            "Acorn",
            &[(1, 0), (3, 1), (0, 2), (1, 2), (4, 2), (5, 2), (6, 2)],
        ),
        Pattern::parse_rle(
            "Gosper Glider Gun",
            "24bo11b$22bobo11b$12b2o6b2o12b2o$11bo3bo4b2o12b2o$2o8bo5bo3b2o14b$2o8bo3bob2o4bobo11b$
10bo5bo7bobo9b$11bo3bo9bo10b$12b2o!",
            36,
            9,
        ),
        Pattern::parse_rle(
            "Pulsar",
            "3b3o3b3o3b$o3bobobobo3bo$o3bobobobo3bo$o3bobobobo3bo$3b3o3b3o3b$
3b3o3b3o3b$o3bobobobo3bo$o3bobobobo3bo$o3bobobobo3bo$3b3o3b3o3b!",
            13,
            13,
        ),
        Pattern::parse_rle("Pentadecathlon", "2bo4bo2b$2ob4o2o$2bo4bo2b!", 10, 3),
        Pattern::parse_rle(
            "Switch Engine",
            "o6bo3b$3o3b3o2b$3bobo3b2o$2b2ob2o2b2o2$2b2ob2o2b2o$3bobo3b2o$
3o3b3o2b$o6bo!",
            11,
            9,
        ),
        Pattern::parse_rle(
            "Period-30 Glider Gun",
            "18bo7b$19bo6b$17b3o6b4$bo20b$obo4b2o13b$o5b3o12b$6bo2bo7b3o$
8b2o6bo3bo$8b2o5bo5bo$8b2o6bo3bo$6bo2bo8b3o$o5b3o$obo4b2o$bo!",
            24,
            17,
        ),
        Pattern::parse_rle(
            "Puffer Train",
            "8bo3bo3b$7b3ob3o2b$6b2o5b2ob$5b3o5b3o$4b3o7b3o$5b3o5b3o$6b2o5b2ob$
7b3ob3o2b$8bo3bo3b$2b2o15b$2ob2o14b$4o15b$b2o!",
            18,
            13,
        ),
    ]
}
