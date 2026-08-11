struct Params {
    fraction: f32,
    seed: u32,
    grid_x: u32,
    grid_y: u32,
};

@group(0) @binding(0) var src: texture_storage_2d<r32uint, read>;
@group(0) @binding(1) var dst: texture_storage_2d<r32uint, write>;
@group(0) @binding(2) var<uniform> params: Params;

fn hash(p: vec2u) -> u32 {
    var x = p.x * 73856093u + p.y * 19349663u + params.seed * 83492791u;
    x = (x ^ (x >> 16u)) * 0x45d9f3bu;
    x = (x ^ (x >> 16u)) * 0x45d9f3bu;
    x = x ^ (x >> 16u);
    return x;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if gid.x >= params.grid_x || gid.y >= params.grid_y {
        return;
    }
    let coord = vec2i(i32(gid.x), i32(gid.y));
    let old = textureLoad(src, coord).r;
    let h = hash(gid.xy);
    let flip = f32(h % 1000u) < params.fraction * 1000.0;
    let next = select(old, select(1u, 0u, old > 0u), flip);
    textureStore(dst, coord, vec4u(next, 0u, 0u, 0u));
}
