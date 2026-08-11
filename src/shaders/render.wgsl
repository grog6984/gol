struct Cam {
    center: vec2f,
    viewport: vec2f,
    grid: vec2f,
    scale: f32,
    circle_threshold: f32,
    wrap: u32,
};

struct VOut {
    @builtin(position) pos: vec4f,
};

@group(0) @binding(0) var state: texture_storage_2d<r32uint, read>;
@group(0) @binding(1) var palette: texture_2d<f32>;
@group(0) @binding(2) var pal_sampler: sampler;
@group(0) @binding(3) var<uniform> cam: Cam;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    var tri = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0)
    );
    var out: VOut;
    out.pos = vec4f(tri[vi], 0.0, 1.0);
    return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
    let half = cam.viewport * 0.5;
    // Continuous grid coordinates at the current fragment (physical pixels).
    let gx = cam.center.x + (in.pos.x - half.x) / cam.scale;
    let gy = cam.center.y + (in.pos.y - half.y) / cam.scale;
    let ix = i32(floor(gx));
    let iy = i32(floor(gy));
    let gxi = i32(cam.grid.x);
    let gyi = i32(cam.grid.y);

    var age: u32;
    if cam.wrap != 0u {
        // Tile the finite world so panning past an edge shows the opposite side.
        let wx = ((ix % gxi) + gxi) % gxi;
        let wy = ((iy % gyi) + gyi) % gyi;
        age = textureLoad(state, vec2i(wx, wy)).r;
    } else {
        if ix < 0 || iy < 0 || ix >= gxi || iy >= gyi {
            return textureSample(palette, pal_sampler, vec2f(0.0, 0.5));
        }
        age = textureLoad(state, vec2i(ix, iy)).r;
    }

    if age == 0u {
        // Dead cells use the palette's background color (t = 0).
        return textureSample(palette, pal_sampler, vec2f(0.0, 0.5));
    }

    // Draw zoomed-in live cells as circles.
    if cam.scale >= cam.circle_threshold {
        let fx = gx - floor(gx);
        let fy = gy - floor(gy);
        let dx = fx - 0.5;
        let dy = fy - 0.5;
        let r = 0.42;
        if dx * dx + dy * dy > r * r {
            return textureSample(palette, pal_sampler, vec2f(0.0, 0.5));
        }
    }

    // Each time a cell's age crosses another power-of-two threshold it steps to a
    // new palette color. Fresh cells (n = 0) use the high-contrast right end.
    let n = i32(floor(log2(f32(age))));
    let t = clamp(1.0 - f32(n) * 0.08, 0.0, 1.0);
    let color = textureSample(palette, pal_sampler, vec2f(t, 0.5));
    return vec4f(color.rgb, 1.0);
}