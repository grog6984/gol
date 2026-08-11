/// Conway-like Life rule encoded as B/S bit masks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rule {
    pub birth: u32,   // bit n set -> dead cell with n neighbours becomes alive
    pub survive: u32, // bit n set -> live cell with n neighbours survives
}

impl Rule {
    pub const fn new(birth: u32, survive: u32) -> Self {
        Self { birth, survive }
    }

    pub fn from_bs(b: &[u8], s: &[u8]) -> Self {
        let mut birth = 0u32;
        let mut survive = 0u32;
        for &n in b {
            birth |= 1 << n;
        }
        for &n in s {
            survive |= 1 << n;
        }
        Self { birth, survive }
    }

    /// Parse strings like "B3/S23" or "23/3" (MCell/Life32 notation, survival first).
    pub fn parse(text: &str) -> Option<Self> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }

        // Try "B3/S23" (Golly / Hensel notation)
        let upper = trimmed.to_ascii_uppercase();
        if let Some(rest) = upper.strip_prefix('B') {
            let parts: Vec<&str> = rest.split('/').collect();
            if parts.len() == 2 {
                let s_part = parts[1].strip_prefix('S').unwrap_or(parts[1]);
                return Some(Self::from_bs(&digits(parts[0]), &digits(s_part)));
            }
        }

        // Try "S23/B3" (reverse)
        if let Some(rest) = upper.strip_prefix('S') {
            let parts: Vec<&str> = rest.split('/').collect();
            if parts.len() == 2 {
                let b_part = parts[1].strip_prefix('B').unwrap_or(parts[1]);
                return Some(Self::from_bs(&digits(b_part), &digits(parts[0])));
            }
        }

        // Try "23/3" (survival/birth)
        let parts: Vec<&str> = trimmed.split('/').collect();
        if parts.len() == 2 {
            return Some(Self::from_bs(&digits(parts[1]), &digits(parts[0])));
        }

        None
    }

    /// Format as "B3/S23".
    pub fn format(&self) -> String {
        format!(
            "B{}/S{}",
            mask_digits(self.birth),
            mask_digits(self.survive)
        )
    }

    pub fn conway() -> Self {
        Self::from_bs(&[3], &[2, 3])
    }

    pub fn highlife() -> Self {
        Self::from_bs(&[3, 6], &[2, 3])
    }

    pub fn day_and_night() -> Self {
        Self::from_bs(&[3, 6, 7, 8], &[3, 4, 6, 7, 8])
    }

    pub fn seeds() -> Self {
        Self::from_bs(&[2], &[])
    }

    pub fn life_without_death() -> Self {
        Self::from_bs(&[3], &[0, 1, 2, 3, 4, 5, 6, 7, 8])
    }

    pub fn diamoeba() -> Self {
        Self::from_bs(&[3, 5, 6, 7, 8], &[5, 6, 7, 8])
    }

    pub fn anneal() -> Self {
        Self::from_bs(&[4, 6, 7, 8], &[3, 5, 6, 7, 8])
    }

    pub fn gnarl() -> Self {
        Self::from_bs(&[1], &[1])
    }

    pub fn mor_anneal() -> Self {
        Self::from_bs(&[4, 6, 8], &[3, 5, 7])
    }
}

fn digits(s: &str) -> Vec<u8> {
    s.chars()
        .filter(|c| c.is_ascii_digit())
        .map(|c| c as u8 - b'0')
        .collect()
}

fn mask_digits(mask: u32) -> String {
    (0..=8)
        .filter(|n| (mask >> n) & 1 != 0)
        .map(|n| std::char::from_digit(n, 10).unwrap())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_conway() {
        let r = Rule::parse("B3/S23").unwrap();
        assert_eq!(r.birth, 1 << 3);
        assert_eq!(r.survive, (1 << 2) | (1 << 3));
        assert_eq!(r.format(), "B3/S23");
    }

    #[test]
    fn parse_slash() {
        let r = Rule::parse("23/3").unwrap();
        assert_eq!(r, Rule::conway());
    }
}
