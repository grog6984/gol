struct Params {
    birth: u32,
    survive: u32,
    wrap: u32,
    _pad: u32,
};

@group(0) @binding(0) var src: texture_storage_2d<r32uint, read>;
@group(0) @binding(1) var dst: texture_storage_2d<r32uint, write>;
@group(0) @binding(2) var<uniform> params: Params;

const MAX_AGE: u32 = 100000u;
const TILE: i32 = 16;
const HALO: i32 = 1;
const LOCAL: i32 = TILE + 2 * HALO;

var<workgroup> tile: array<u32, LOCAL * LOCAL>;

@compute @workgroup_size(TILE, TILE)
fn main(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let size = vec2i(textureDimensions(src));
    let coord = vec2i(gid.xy);
    let base = coord - vec2i(lid.xy) - HALO;

    // Load a TILE+2 halo tile into shared memory.
    for (var ly = i32(lid.y); ly < LOCAL; ly += TILE) {
        for (var lx = i32(lid.x); lx < LOCAL; lx += TILE) {
            let gx = base.x + lx;
            let gy = base.y + ly;
            var v = 0u;
            if params.wrap != 0u {
                let wx = ((gx % size.x) + size.x) % size.x;
                let wy = ((gy % size.y) + size.y) % size.y;
                v = textureLoad(src, vec2i(wx, wy)).r;
            } else if gx >= 0 && gy >= 0 && gx < size.x && gy < size.y {
                v = textureLoad(src, vec2i(gx, gy)).r;
            }
            tile[u32(ly * LOCAL + lx)] = v;
        }
    }
    workgroupBarrier();

    if coord.x >= size.x || coord.y >= size.y {
        return;
    }

    let lx = i32(lid.x) + HALO;
    let ly = i32(lid.y) + HALO;

    var count = 0u;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if dx == 0 && dy == 0 {
                continue;
            }
            if tile[u32((ly + dy) * LOCAL + (lx + dx))] > 0u {
                count += 1u;
            }
        }
    }

    let old = tile[u32(ly * LOCAL + lx)];
    let bit = 1u << count;
    var next: u32;
    if old > 0u {
        next = select(0u, min(old + 1u, MAX_AGE), (params.survive & bit) != 0u);
    } else {
        next = select(0u, 1u, (params.birth & bit) != 0u);
    }

    textureStore(dst, coord, vec4u(next, 0u, 0u, 0u));
}
