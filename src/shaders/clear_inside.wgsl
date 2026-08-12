struct Selection {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    grid_w: u32,
    grid_h: u32,
};

@group(0) @binding(0) var state: texture_storage_2d<r32uint, write>;
@group(0) @binding(1) var<uniform> sel: Selection;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if gid.x >= sel.grid_w || gid.y >= sel.grid_h {
        return;
    }
    let inside = gid.x >= sel.x0 && gid.x <= sel.x1 && gid.y >= sel.y0 && gid.y <= sel.y1;
    if inside {
        textureStore(state, vec2i(gid.xy), vec4u(0u, 0u, 0u, 0u));
    }
}
