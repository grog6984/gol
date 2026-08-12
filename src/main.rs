mod palette;
mod patterns;
mod rules;
mod script;
mod sim;

use eframe::egui;
use egui::{Color32, Rect, Sense};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::palette::presets as palette_presets;
use crate::patterns::{Pattern, presets as pattern_presets};
use crate::rules::Rule;
use crate::script::{ScriptEngine, ScriptResult};
use crate::sim::{ReadbackResult, Sim};

fn main() {
    // Force X11 so fullscreen is truly borderless (Wayland often keeps a thin frame).
    #[cfg(target_os = "linux")]
    unsafe {
        std::env::set_var("WINIT_UNIX_BACKEND", "x11");
    }

    env_logger::init();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_maximized(true)
            .with_title("Game of Life")
            .with_drag_and_drop(false),
        vsync: true,
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            present_mode: wgpu::PresentMode::AutoVsync,
            wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(
                eframe::egui_wgpu::WgpuSetupCreateNew {
                    instance_descriptor: wgpu::InstanceDescriptor {
                        backends: wgpu::Backends::VULKAN,
                        ..Default::default()
                    },
                    device_descriptor: Arc::new(|adapter| {
                        let mut desc = wgpu::DeviceDescriptor::default();
                        let supported = adapter.features();
                        if supported
                            .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
                        {
                            desc.required_features |=
                                wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
                        }
                        desc
                    }),
                    ..Default::default()
                },
            ),
            ..Default::default()
        },
        ..Default::default()
    };
    eframe::run_native(
        "Game of Life",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
    .expect("eframe failed");
}

struct App {
    sim: Arc<Mutex<Sim>>,
    script_engine: ScriptEngine,
    script_source: String,
    script_error: Option<String>,

    generation: u64,
    running: bool,
    steps_per_frame: usize,
    show_ui: bool,
    fullscreen: bool,

    scale: f32,
    center: egui::Vec2,
    wrap: bool,

    selected_pattern_idx: usize,
    selected_palette_idx: usize,
    rule_text: String,
    rule_error: Option<String>,

    drawing: bool,
    last_mouse_pos: Option<(i32, i32)>,
    pending_edits: Vec<crate::sim::Edit>,
    screen_fitted: bool,
    show_quit_dialog: bool,
    last_mouse_move: Instant,
    cursor_hidden: bool,

    selection: Option<[i32; 4]>,
    selection_start: Option<egui::Pos2>,
    selection_current: Option<egui::Pos2>,
    selection_edge: Option<SelectionEdge>,
    hovered_edge: Option<SelectionEdge>,
    world_rect: Rect,
    export_status: Option<(String, Instant)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SelectionEdge {
    Left,
    Right,
    Top,
    Bottom,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

const INITIAL_GRID: u32 = 1024;

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let render_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("this app requires the wgpu backend");
        log::info!("wgpu adapter: {:?}", render_state.adapter.get_info());
        log::info!("wgpu features: {:?}", render_state.adapter.features());
        let device = render_state.device.clone();
        let queue = render_state.queue.clone();
        let surface_format = render_state.target_format;

        let rule = Rule::conway();
        let palette = &palette_presets()[0];
        let sim = Sim::new(
            device.clone(),
            queue.clone(),
            INITIAL_GRID,
            INITIAL_GRID,
            surface_format,
            rule,
            palette,
        );

        let mut script_engine = ScriptEngine::new().expect("script engine");
        let script_source = DEFAULT_SCRIPT.to_string();
        let script_error = script_engine
            .set_source(&script_source)
            .err()
            .map(|e| format!("{e:?}"));

        let mut app = Self {
            sim: Arc::new(Mutex::new(sim)),
            script_engine,
            script_source,
            script_error,
            generation: 0,
            running: false,
            steps_per_frame: 1,
            show_ui: true,
            fullscreen: false,
            scale: 1.0,
            center: egui::Vec2::splat(INITIAL_GRID as f32 * 0.5),
            wrap: true,
            selected_pattern_idx: 0,
            selected_palette_idx: 0,
            rule_text: Rule::conway().format(),
            rule_error: None,
            drawing: false,
            last_mouse_pos: None,
            pending_edits: Vec::new(),
            screen_fitted: false,
            show_quit_dialog: false,
            last_mouse_move: Instant::now(),
            cursor_hidden: false,
            selection: None,
            selection_start: None,
            selection_current: None,
            selection_edge: None,
            hovered_edge: None,
            world_rect: Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0)),
            export_status: None,
        };
        let initial = pattern_presets()[app.selected_pattern_idx].clone();
        app.reset_to_pattern(&initial);
        app
    }

    fn apply_script_result(&mut self, result: ScriptResult) {
        if let Some(rule) = result.rule {
            self.rule_text = rule.format();
            self.sim.lock().unwrap().set_rule(rule);
        }
        if let Some(name) = result.palette {
            if let Some(idx) = palette_presets().iter().position(|p| p.name == name) {
                self.selected_palette_idx = idx;
                self.sim
                    .lock()
                    .unwrap()
                    .set_palette(&palette_presets()[idx]);
            }
        }
    }

    fn run_steps(&mut self, n: usize) {
        // The timeline script is called once per frame at the current generation.
        // If you need per-generation script precision, set steps_per_frame to 1.
        let result = self
            .script_engine
            .on_step(self.generation)
            .unwrap_or_default();
        self.apply_script_result(result);

        let cb = self.sim.lock().unwrap().step(n as u32);
        self.generation += n as u64;
        self.sim.lock().unwrap().submit(vec![cb]);
    }

    fn cell_at(&self, rect: Rect, pos: egui::Pos2) -> Option<(i32, i32)> {
        if !rect.contains(pos) {
            return None;
        }
        let rel = pos - rect.min;
        let vp = rect.size() * 0.5;
        let gx = self.center.x + (rel.x - vp.x) / self.scale;
        let gy = self.center.y + (rel.y - vp.y) / self.scale;
        let ix = gx.floor() as i32;
        let iy = gy.floor() as i32;
        if ix < 0 || iy < 0 {
            return None;
        }
        let (w, h) = self.sim.lock().unwrap().size;
        if (ix as u32) < w && (iy as u32) < h {
            Some((ix, iy))
        } else {
            None
        }
    }

    fn zoom_at(&mut self, rect: Rect, pos: egui::Pos2, factor: f32) {
        let (gw, gh) = self.sim.lock().unwrap().size;
        let vp = rect.size();
        let min_scale = (vp.x / gw as f32).max(vp.y / gh as f32);
        let new_scale = (self.scale * factor).clamp(min_scale, 64.0);
        if new_scale != self.scale {
            let rel = pos - rect.min;
            let vpc = rect.size() * 0.5;
            let gx = self.center.x + (rel.x - vpc.x) / self.scale;
            let gy = self.center.y + (rel.y - vpc.y) / self.scale;
            // Keep the point under the mouse fixed while zooming, but stop panning
            // once we hit the minimum (1:1) scale so zooming out further does not
            // drift the view.
            if factor > 1.0 || new_scale > min_scale {
                self.center = egui::Vec2::new(
                    gx - (rel.x - vpc.x) / new_scale,
                    gy - (rel.y - vpc.y) / new_scale,
                );
            }
            self.scale = new_scale;
        }
        self.clamp_camera(rect);
    }

    fn clamp_camera(&mut self, rect: Rect) {
        let (gw, gh) = self.sim.lock().unwrap().size;
        let gw = gw as f32;
        let gh = gh as f32;
        let vp = rect.size();
        let min_scale = (vp.x / gw).max(vp.y / gh);
        if self.scale < min_scale {
            self.scale = min_scale;
        }
        if !self.wrap {
            // Keep the finite grid inside the viewport; at min zoom this is the centered 1:1 view.
            let half_x = vp.x / (2.0 * self.scale);
            let half_y = vp.y / (2.0 * self.scale);
            let min_cx = half_x;
            let max_cx = (gw - half_x).max(min_cx);
            let min_cy = half_y;
            let max_cy = (gh - half_y).max(min_cy);
            self.center.x = self.center.x.clamp(min_cx, max_cx);
            self.center.y = self.center.y.clamp(min_cy, max_cy);
        }
    }

    fn resize_to_world(&mut self, _ctx: &egui::Context, rect: Rect) {
        // The world only ever grows, never shrinks, so existing content is never
        // cropped. When the viewport is smaller than the world, the camera zooms
        // out to fit (see clamp_camera's min_scale). When it's larger (e.g. going
        // into fullscreen), the world grows to exactly fill the screen (no frame)
        // and preserves the current cells by centering them in the new grid.
        let (gw, gh) = self.sim.lock().unwrap().size;
        let w = (rect.width().ceil() as u32).max(1);
        let h = (rect.height().ceil() as u32).max(1);
        if w > gw || h > gh {
            let nw = w.max(gw);
            let nh = h.max(gh);
            let shift_x = ((nw as i64 - gw as i64) / 2) as f32;
            let shift_y = ((nh as i64 - gh as i64) / 2) as f32;
            self.center += egui::Vec2::new(shift_x, shift_y);
            self.sim.lock().unwrap().resize(nw, nh);
        }
        self.clamp_camera(rect);
    }

    fn paint_at(&mut self, x: i32, y: i32, erase: bool) {
        let (w, h) = self.sim.lock().unwrap().size;
        if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
            self.pending_edits.push(crate::sim::Edit {
                x: x as u32,
                y: y as u32,
                value: if erase { 0 } else { 1 },
            });
        }
    }

    fn clear(&mut self) {
        let mut sim = self.sim.lock().unwrap();
        sim.cpu_state.fill(0);
        sim.upload_cpu_state();
        self.generation = 0;
    }

    fn randomize(&mut self, flip_fraction: f32) {
        let was_running = self.running;
        self.running = false;
        self.flush_edits();
        let seed = rand::random::<u32>();
        let cb = self.sim.lock().unwrap().randomize_gpu(flip_fraction, seed);
        self.sim.lock().unwrap().submit(vec![cb]);
        self.running = was_running;
    }

    fn flush_edits(&mut self) {
        if self.pending_edits.is_empty() {
            return;
        }
        let edits = std::mem::take(&mut self.pending_edits);
        let cb = self.sim.lock().unwrap().apply_edits(&edits);
        self.sim.lock().unwrap().submit(vec![cb]);
    }

    fn world_to_screen(&self, rect: Rect, gx: f32, gy: f32) -> egui::Pos2 {
        let vp = rect.size() * 0.5;
        egui::pos2(
            rect.min.x + (gx - self.center.x) * self.scale + vp.x,
            rect.min.y + (gy - self.center.y) * self.scale + vp.y,
        )
    }

    fn finish_selection(&mut self, rect: Rect) {
        if let (Some(a), Some(b)) = (self.selection_start, self.selection_current) {
            if let (Some((ax, ay)), Some((bx, by))) = (self.cell_at(rect, a), self.cell_at(rect, b))
            {
                self.selection = Some([
                    ax.min(bx),
                    ay.min(by),
                    ax.max(bx),
                    ay.max(by),
                ]);
            } else {
                self.selection = None;
            }
        }
        self.selection_start = None;
        self.selection_current = None;
    }

    fn selection_screen_rect(&self, rect: Rect) -> Option<egui::Rect> {
        let sel = self.selection?;
        let min = self.world_to_screen(rect, sel[0] as f32, sel[1] as f32);
        let max = self.world_to_screen(rect, sel[2] as f32 + 1.0, sel[3] as f32 + 1.0);
        Some(egui::Rect::from_min_max(min, max))
    }

    fn hover_selection_edge(&self, rect: Rect, pos: Option<egui::Pos2>) -> Option<SelectionEdge> {
        let pos = pos?;
        let sel = self.selection_screen_rect(rect)?;
        let margin = 6.0;
        let near_left = (pos.x - sel.left()).abs() < margin;
        let near_right = (pos.x - sel.right()).abs() < margin;
        let near_top = (pos.y - sel.top()).abs() < margin;
        let near_bottom = (pos.y - sel.bottom()).abs() < margin;
        let inside_x = pos.x >= sel.left() - margin && pos.x <= sel.right() + margin;
        let inside_y = pos.y >= sel.top() - margin && pos.y <= sel.bottom() + margin;
        if !inside_x || !inside_y {
            return None;
        }
        match (near_top, near_bottom, near_left, near_right) {
            (true, false, true, false) => Some(SelectionEdge::TopLeft),
            (true, false, false, true) => Some(SelectionEdge::TopRight),
            (false, true, true, false) => Some(SelectionEdge::BottomLeft),
            (false, true, false, true) => Some(SelectionEdge::BottomRight),
            (true, false, _, _) => Some(SelectionEdge::Top),
            (false, true, _, _) => Some(SelectionEdge::Bottom),
            (_, _, true, false) => Some(SelectionEdge::Left),
            (_, _, false, true) => Some(SelectionEdge::Right),
            _ => None,
        }
    }

    fn edge_cursor(edge: SelectionEdge) -> egui::CursorIcon {
        match edge {
            SelectionEdge::Left | SelectionEdge::Right => egui::CursorIcon::ResizeHorizontal,
            SelectionEdge::Top | SelectionEdge::Bottom => egui::CursorIcon::ResizeVertical,
            SelectionEdge::TopLeft | SelectionEdge::BottomRight => {
                egui::CursorIcon::ResizeNwSe
            }
            SelectionEdge::TopRight | SelectionEdge::BottomLeft => {
                egui::CursorIcon::ResizeNeSw
            }
        }
    }

    fn resize_selection(&mut self, edge: SelectionEdge, rect: Rect, pos: egui::Pos2) {
        let (cx, cy) = match self.cell_at(rect, pos) {
            Some(p) => p,
            None => return,
        };
        let Some(sel) = self.selection.as_mut() else { return };
        match edge {
            SelectionEdge::Left => sel[0] = cx.min(sel[2]),
            SelectionEdge::Right => sel[2] = cx.max(sel[0]),
            SelectionEdge::Top => sel[1] = cy.min(sel[3]),
            SelectionEdge::Bottom => sel[3] = cy.max(sel[1]),
            SelectionEdge::TopLeft => {
                sel[0] = cx.min(sel[2]);
                sel[1] = cy.min(sel[3]);
            }
            SelectionEdge::TopRight => {
                sel[2] = cx.max(sel[0]);
                sel[1] = cy.min(sel[3]);
            }
            SelectionEdge::BottomLeft => {
                sel[0] = cx.min(sel[2]);
                sel[3] = cy.max(sel[1]);
            }
            SelectionEdge::BottomRight => {
                sel[2] = cx.max(sel[0]);
                sel[3] = cy.max(sel[1]);
            }
        }
    }

    fn clear_selection_if_outside(&mut self, rect: Rect, pos: egui::Pos2) {
        if let Some(sel) = self.selection_screen_rect(rect) {
            if !sel.contains(pos) {
                self.selection = None;
            }
        }
    }

    fn clear_outside_selection(&mut self) {
        let Some(sel) = self.selection else { return };
        self.flush_edits();
        let Some((x0, y0, x1, y1)) = self.selection_rect_cells(sel) else {
            return;
        };
        let cb = self.sim.lock().unwrap().clear_outside(x0, y0, x1, y1);
        self.sim.lock().unwrap().submit(vec![cb]);
    }

    fn clear_inside_selection(&mut self) {
        let Some(sel) = self.selection else { return };
        self.flush_edits();
        let Some((x0, y0, x1, y1)) = self.selection_rect_cells(sel) else {
            return;
        };
        let cb = self.sim.lock().unwrap().clear_inside(x0, y0, x1, y1);
        self.sim.lock().unwrap().submit(vec![cb]);
    }

    /// Clamp a selection rectangle to the world bounds as inclusive cell coords.
    fn selection_rect_cells(&self, sel: [i32; 4]) -> Option<(u32, u32, u32, u32)> {
        let (w, h) = self.sim.lock().unwrap().size;
        let x0 = sel[0].clamp(0, w as i32 - 1) as u32;
        let y0 = sel[1].clamp(0, h as i32 - 1) as u32;
        let x1 = sel[2].clamp(0, w as i32 - 1) as u32;
        let y1 = sel[3].clamp(0, h as i32 - 1) as u32;
        if x1 < x0 || y1 < y0 {
            return None;
        }
        Some((x0, y0, x1, y1))
    }

    fn save_selection_png(&mut self) {
        self.flush_edits();
        let cells = match self.selection {
            Some(sel) => self.selection_rect_cells(sel),
            None => self.selection_rect_cells(self.visible_cells(self.world_rect)),
        };
        let Some((x0, y0, x1, y1)) = cells else { return };
        let mut sim = self.sim.lock().unwrap();
        if sim.png_readback.is_some() {
            self.export_status = Some(("Export already in progress".to_string(), Instant::now()));
            return;
        }
        let path = next_download_path();
        let palette = sim.palette.clone();
        sim.request_selection_png(x0, y0, x1, y1, path.clone(), palette);
        let label = if self.selection.is_some() {
            "Saving selection…".to_string()
        } else {
            "Saving viewport…".to_string()
        };
        self.export_status = Some((label, Instant::now()));
    }

    fn copy_selection_to_clipboard(&mut self) {
        let Some(sel) = self.selection else { return };
        self.flush_edits();
        let Some((x0, y0, x1, y1)) = self.selection_rect_cells(sel) else {
            return;
        };
        let mut sim = self.sim.lock().unwrap();
        if sim.png_readback.is_some() {
            self.export_status = Some(("Export already in progress".to_string(), Instant::now()));
            return;
        }
        let palette = sim.palette.clone();
        sim.request_selection_clipboard(x0, y0, x1, y1, palette);
        self.export_status = Some(("Copying selection…".to_string(), Instant::now()));
    }

    fn select_visible(&mut self) {
        self.selection_start = None;
        self.selection_current = None;
        self.selection_edge = None;
        self.selection = Some(self.visible_cells(self.world_rect));
    }

    fn select_world(&mut self) {
        self.selection_start = None;
        self.selection_current = None;
        self.selection_edge = None;
        let (w, h) = self.sim.lock().unwrap().size;
        if w == 0 || h == 0 {
            self.selection = None;
            return;
        }
        self.selection = Some([0, 0, w as i32 - 1, h as i32 - 1]);
    }

    fn visible_cells(&self, rect: Rect) -> [i32; 4] {
        let (w, h) = self.sim.lock().unwrap().size;
        let vp = rect.size() * 0.5;
        let world = |pos: egui::Pos2| -> (f32, f32) {
            let rel = pos - rect.min;
            (
                self.center.x + (rel.x - vp.x) / self.scale,
                self.center.y + (rel.y - vp.y) / self.scale,
            )
        };
        let (ax, ay) = world(rect.left_top());
        let (bx, by) = world(rect.right_bottom());
        let x0 = (ax.floor() as i32).clamp(0, w as i32 - 1);
        let y0 = (ay.floor() as i32).clamp(0, h as i32 - 1);
        let x1 = (bx.floor() as i32).clamp(0, w as i32 - 1);
        let y1 = (by.floor() as i32).clamp(0, h as i32 - 1);
        [x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1)]
    }

    fn draw_selection(&self, ui: &egui::Ui, rect: Rect) {
        let accent = egui::Color32::from_rgb(70, 160, 255);
        let screen_rect = match (
            self.selection_current,
            self.selection_start,
            self.selection,
        ) {
            (Some(cur), Some(start), _) => egui::Rect::from_two_pos(start, cur),
            (None, None, Some(_)) => match self.selection_screen_rect(rect) {
                Some(r) => r,
                None => return,
            },
            _ => return,
        };
        let painter = ui.painter();
        painter.rect_filled(screen_rect, 2.0, accent.gamma_multiply(0.10));
        let stroke = egui::Stroke::new(1.5_f32, accent);
        painter.rect_stroke(screen_rect, 2.0, stroke, egui::StrokeKind::Outside);
        let handle = 6.0;
        let handle_stroke = egui::Stroke::new(1.0_f32, egui::Color32::WHITE);
        for corner in [
            screen_rect.left_top(),
            screen_rect.right_top(),
            screen_rect.left_bottom(),
            screen_rect.right_bottom(),
        ] {
            let r = egui::Rect::from_center_size(corner, egui::Vec2::splat(handle));
            painter.rect_filled(r, 1.0, accent);
            painter.rect_stroke(r, 1.0, handle_stroke, egui::StrokeKind::Outside);
        }

        // Highlight the edge or corner being hovered or dragged for resizing.
        if let Some(edge) = self.selection_edge.or(self.hovered_edge) {
            let hi = egui::Stroke::new(2.5_f32, egui::Color32::WHITE);
            let lt = screen_rect.left_top();
            let rt = screen_rect.right_top();
            let lb = screen_rect.left_bottom();
            let rb = screen_rect.right_bottom();
            match edge {
                SelectionEdge::Left => {
                    painter.line_segment([lt, lb], hi);
                }
                SelectionEdge::Right => {
                    painter.line_segment([rt, rb], hi);
                }
                SelectionEdge::Top => {
                    painter.line_segment([lt, rt], hi);
                }
                SelectionEdge::Bottom => {
                    painter.line_segment([lb, rb], hi);
                }
                SelectionEdge::TopLeft => {
                    painter.line_segment([lt, rt], hi);
                    painter.line_segment([lt, lb], hi);
                }
                SelectionEdge::TopRight => {
                    painter.line_segment([lt, rt], hi);
                    painter.line_segment([rt, rb], hi);
                }
                SelectionEdge::BottomLeft => {
                    painter.line_segment([lb, rb], hi);
                    painter.line_segment([lt, lb], hi);
                }
                SelectionEdge::BottomRight => {
                    painter.line_segment([lb, rb], hi);
                    painter.line_segment([rt, rb], hi);
                }
            }
        }

        if self.selection.is_some() {
            let cw = (screen_rect.width() / self.scale).round() as u64;
            let ch = (screen_rect.height() / self.scale).round() as u64;
            let text = format!("{cw}×{ch}  ·  Y clear out  ·  C clear in  ·  Ctrl+C copy  ·  Ctrl+S save  ·  Ctrl+A view  ·  drag edges  ·  Esc");
            let galley = painter.layout_no_wrap(
                text,
                egui::FontId::monospace(11.0),
                egui::Color32::from_rgb(235, 235, 235),
            );
            let mut bg = screen_rect.left_top() - egui::vec2(0.0, galley.size().y + 6.0);
            bg.x = bg.x.clamp(rect.left(), rect.right() - galley.size().x);
            bg.y = bg.y.max(rect.top());
            painter.rect_filled(
                egui::Rect::from_min_size(bg, galley.size() + egui::vec2(8.0, 4.0)),
                3.0,
                egui::Color32::from_black_alpha(190),
            );
            painter.galley(bg + egui::vec2(4.0, 2.0), galley, egui::Color32::WHITE);
        }
    }

    fn reset_to_pattern(&mut self, pat: &Pattern) {
        self.clear();
        {
            let mut sim = self.sim.lock().unwrap();
            let cx = (sim.size.0 / 2) as i32;
            let cy = (sim.size.1 / 2) as i32;
            let (w, h) = sim.size;
            for (px, py) in &pat.centered().cells {
                let x = cx + px;
                let y = cy + py;
                if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
                    let i = (y as u32 * w + x as u32) as usize;
                    sim.cpu_state[i] = 1;
                }
            }
            sim.upload_cpu_state();
            let cx_f = sim.size.0 as f32 * 0.5;
            let cy_f = sim.size.1 as f32 * 0.5;
            self.center = egui::Vec2::new(cx_f, cy_f);
        }
        self.generation = 0;
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.show_quit_dialog {
            let confirm = ctx.input(|i| {
                i.key_pressed(egui::Key::Q)
                    || i.key_pressed(egui::Key::Enter)
                    || i.key_pressed(egui::Key::Y)
            });
            let cancel = ctx.input(|i| {
                i.key_pressed(egui::Key::Escape) || i.key_pressed(egui::Key::N)
            });
            if confirm {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            } else if cancel {
                self.show_quit_dialog = false;
            }
        } else {
            let (f1, space, fkey) = ctx.input(|i| {
                (
                    i.key_pressed(egui::Key::F1),
                    i.key_pressed(egui::Key::Space),
                    i.key_pressed(egui::Key::F),
                )
            });
            if f1 {
                self.show_ui = !self.show_ui;
            }
            if space {
                self.running = !self.running;
            }
            if fkey {
                self.fullscreen = !self.fullscreen;
                if self.fullscreen {
                    self.show_ui = false;
                    self.last_mouse_move = Instant::now();
                }
                // Use the WM fullscreen hint so the window covers the monitor and
                // the desktop panels are hidden. Also strip decorations so the
                // WM's thin resize border around the fullscreen window is removed.
                // The grow-only world fills the screen exactly, so edge-to-edge.
                ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(self.fullscreen));
                ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(!self.fullscreen));
                ctx.request_repaint();
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Q)) {
                self.show_quit_dialog = true;
            }
            for (key, pct) in [
                (egui::Key::Num1, 0.01),
                (egui::Key::Num2, 0.10),
                (egui::Key::Num3, 0.20),
                (egui::Key::Num4, 0.50),
                (egui::Key::Num5, 1.00),
            ] {
                if ctx.input(|i| i.key_pressed(key)) {
                    self.randomize(pct);
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::ArrowUp)) {
                self.steps_per_frame = (self.steps_per_frame + 1).min(20);
            }
            if ctx.input(|i| i.key_pressed(egui::Key::ArrowDown)) {
                self.steps_per_frame = (self.steps_per_frame.saturating_sub(1)).max(1);
            }

            // Selection editing shortcuts (paused only).
            if !self.running {
                let (y_pressed, ctrl_s, esc) = ctx.input(|i| {
                    (
                        i.key_pressed(egui::Key::Y) && !i.modifiers.command,
                        i.modifiers.command && i.key_pressed(egui::Key::S),
                        i.key_pressed(egui::Key::Escape),
                    )
                });
                if y_pressed {
                    self.clear_outside_selection();
                }
                if ctrl_s {
                    self.save_selection_png();
                }
                if esc {
                    self.selection = None;
                    self.selection_start = None;
                    self.selection_current = None;
                    self.selection_edge = None;
                }
            }

            // Selection and palette shortcuts, gated so they never fire while a
            // text field has focus (e.g. typing a rule or script).
            if !ctx.wants_keyboard_input() {
                let (c_pressed, ctrl_c, ctrl_a, shift_ctrl_a, left, right) = ctx.input(|i| {
                    (
                        i.key_pressed(egui::Key::C)
                            && !i.modifiers.command
                            && !i.modifiers.shift,
                        i.modifiers.command && !i.modifiers.shift && i.key_pressed(egui::Key::C),
                        i.modifiers.command && !i.modifiers.shift && i.key_pressed(egui::Key::A),
                        i.modifiers.command && i.modifiers.shift && i.key_pressed(egui::Key::A),
                        i.key_pressed(egui::Key::ArrowLeft),
                        i.key_pressed(egui::Key::ArrowRight),
                    )
                });
                if !self.running {
                    if c_pressed {
                        self.clear_inside_selection();
                    }
                    if ctrl_c {
                        self.copy_selection_to_clipboard();
                    }
                    if ctrl_a {
                        self.select_visible();
                    }
                    if shift_ctrl_a {
                        self.select_world();
                    }
                }
                if left || right {
                    let n = palette_presets().len();
                    if n > 0 {
                        let delta = if right { 1 } else { n - 1 };
                        self.selected_palette_idx = (self.selected_palette_idx + delta) % n;
                        self.sim
                            .lock()
                            .unwrap()
                            .set_palette(&palette_presets()[self.selected_palette_idx]);
                    }
                }
            }

            // Finish any pending PNG export / clipboard copy.
            if let Some(result) = self.sim.lock().unwrap().poll_png() {
                match result {
                    Ok(ReadbackResult::Saved(path)) => {
                        self.export_status =
                            Some((format!("Saved {}", path.display()), Instant::now()));
                        if let Some(name) = path.file_name() {
                            if let Some(name) = name.to_str() {
                                println!("{name}");
                            }
                        }
                    }
                    Ok(ReadbackResult::Image {
                        width,
                        height,
                        rgba,
                    }) => {
                        let status = match copy_image_to_clipboard(width, height, rgba) {
                            Ok(()) => "Copied selection to clipboard".to_string(),
                            Err(e) => format!("Copy failed: {e}"),
                        };
                        self.export_status = Some((status, Instant::now()));
                    }
                    Err(e) => {
                        self.export_status =
                            Some((format!("Export failed: {e}"), Instant::now()));
                    }
                }
            }
        }

        // Hide the OS mouse cursor in fullscreen after 500 ms of inactivity.
        if self.fullscreen {
            let moved = ctx.input(|i| i.pointer.delta().length_sq() > 0.0);
            let now = Instant::now();
            if moved {
                self.last_mouse_move = now;
                if self.cursor_hidden {
                    ctx.send_viewport_cmd(egui::ViewportCommand::CursorVisible(true));
                    self.cursor_hidden = false;
                }
            } else if !self.cursor_hidden
                && now.duration_since(self.last_mouse_move) > Duration::from_millis(500)
            {
                ctx.send_viewport_cmd(egui::ViewportCommand::CursorVisible(false));
                self.cursor_hidden = true;
            }
        } else if self.cursor_hidden {
            ctx.send_viewport_cmd(egui::ViewportCommand::CursorVisible(true));
            self.cursor_hidden = false;
        }

        // Adapt the game area to the native screen resolution once at startup.
        // Some compositors ignore `with_maximized`, so force it after the first frame.
        if !self.screen_fitted {
            let sr = ctx.input(|i| i.screen_rect());
            if sr.width() > 100.0 && sr.height() > 100.0 {
                ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
                    sr.width(),
                    sr.height(),
                )));
                ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(true));
                self.screen_fitted = true;
            }
        }

        if self.running {
            self.run_steps(self.steps_per_frame);
        }

        // Population readback disabled to keep the UI thread responsive.
        // See Sim::poll_population / request_population_readback if you want to re-enable it.

        if self.show_ui {
            egui::Window::new("Game of Life")
                .default_pos([8.0, 8.0])
                .resizable(true)
                .show(ctx, |ui| {
                    ui.separator();

                    ui.horizontal(|ui| {
                        if ui
                            .button(if self.running { "Pause" } else { "Play" })
                            .clicked()
                        {
                            self.running = !self.running;
                        }
                        if ui.button("Step").clicked() && !self.running {
                            self.run_steps(self.steps_per_frame);
                        }
                        if ui.button("Clear").clicked() {
                            self.clear();
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.add(
                            egui::Slider::new(&mut self.steps_per_frame, 1..=20)
                                .text("steps/frame"),
                        );
                        let wrap_resp = ui.checkbox(&mut self.wrap, "Connect opposing edges");
                        if wrap_resp.changed() {
                            self.sim.lock().unwrap().set_wrap(self.wrap);
                        }
                        wrap_resp.on_hover_text("Wrap edges (torus) or treat them as a dead abyss");
                    });

                    ui.separator();
                    ui.label("Rule (B/S)");
                    ui.horizontal(|ui| {
                        ui.text_edit_singleline(&mut self.rule_text);
                        if ui.button("Apply").clicked() {
                            match Rule::parse(&self.rule_text) {
                                Some(r) => {
                                    self.rule_error = None;
                                    self.sim.lock().unwrap().set_rule(r);
                                }
                                None => self.rule_error = Some("Invalid rule".into()),
                            }
                        }
                    });
                    if let Some(err) = &self.rule_error {
                        ui.colored_label(Color32::RED, err);
                    }
                    egui::ComboBox::from_label("Preset rules")
                        .selected_text("pick")
                        .show_ui(ui, |ui| {
                            let presets = [
                                ("Conway", Rule::conway()),
                                ("HighLife", Rule::highlife()),
                                ("Day & Night", Rule::day_and_night()),
                                ("Seeds", Rule::seeds()),
                                ("Life w/o Death", Rule::life_without_death()),
                                ("Diamoeba", Rule::diamoeba()),
                                ("Anneal", Rule::anneal()),
                                ("Gnarl", Rule::gnarl()),
                                ("MorAnneal", Rule::mor_anneal()),
                            ];
                            for (name, r) in presets {
                                if ui.selectable_label(false, name).clicked() {
                                    self.rule_text = r.format();
                                    self.sim.lock().unwrap().set_rule(r);
                                }
                            }
                        });

                    ui.separator();
                    ui.label("Palette");
                    let palette_names: Vec<_> =
                        palette_presets().iter().map(|p| p.name.clone()).collect();
                    egui::ComboBox::from_label("Preset palettes")
                        .selected_text(&palette_names[self.selected_palette_idx])
                        .show_ui(ui, |ui| {
                            for (i, name) in palette_names.iter().enumerate() {
                                if ui
                                    .selectable_label(i == self.selected_palette_idx, name)
                                    .clicked()
                                {
                                    self.selected_palette_idx = i;
                                    self.sim.lock().unwrap().set_palette(&palette_presets()[i]);
                                }
                            }
                        });

                    ui.separator();
                    ui.label("Start structure");
                    let pattern_names: Vec<_> =
                        pattern_presets().iter().map(|p| p.name.clone()).collect();
                    egui::ComboBox::from_label("Pattern")
                        .selected_text(&pattern_names[self.selected_pattern_idx])
                        .show_ui(ui, |ui| {
                            for (i, name) in pattern_names.iter().enumerate() {
                                if ui
                                    .selectable_label(i == self.selected_pattern_idx, name)
                                    .clicked()
                                {
                                    self.selected_pattern_idx = i;
                                    self.reset_to_pattern(&pattern_presets()[i]);
                                }
                            }
                        });
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Flip random cells:");
                        for pct in [1, 10, 20, 50, 100] {
                            if ui.button(format!("{}%", pct)).clicked() {
                                self.randomize(pct as f32 / 100.0);
                            }
                        }
                    });

                    ui.separator();
                    ui.label(format!("Generation: {}", self.generation));
                    ui.label("Population: -");

                    ui.separator();
                    ui.collapsing("Script timeline", |ui| {
                        ui.label("function onStep(t) { ... }");
                        ui.add_sized(
                            [ui.available_width(), 200.0],
                            egui::TextEdit::multiline(&mut self.script_source),
                        );
                        if ui.button("Apply script").clicked() {
                            match self.script_engine.set_source(&self.script_source) {
                                Ok(()) => self.script_error = None,
                                Err(e) => self.script_error = Some(e),
                            }
                        }
                        if let Some(err) = &self.script_error {
                            ui.colored_label(Color32::RED, err);
                        }
                    });

                    ui.separator();
                    ui.label("Shortcuts: Space = run/pause  F1 = UI  F = fullscreen  1-5 = flip %  Q = quit");
                    ui.label("Up/Down = step count  Left/Right = palette  Wheel = zoom  middle/right-drag = pan");
                    ui.label("Ctrl+drag = select area   drag edges = resize   Y = clear outside   C = clear inside");
                    ui.label("Ctrl+C = copy   Ctrl+S = export (selection or fullscreen)");
                    ui.label("Ctrl+A = select view   Shift+Ctrl+A = select whole world   Esc / click outside = clear selection");
                });
        }

        if self.show_quit_dialog {
            egui::Window::new("Quit Game of Life")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    ui.label("Do you want to quit?");
                    ui.horizontal(|ui| {
                        if ui.button("Yes").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                        if ui.button("No").clicked() {
                            self.show_quit_dialog = false;
                        }
                    });
                });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            // Panel-local coordinates (origin at the window's top-left) so the
            // world matches egui's pointer/painting space exactly. Using the
            // global inner_rect here offsets everything by the window position,
            // which broke the moat and the selection mapping in windowed mode.
            let rect = ui.available_rect_before_wrap();
            self.world_rect = rect;
            let ppp = ctx.pixels_per_point();
            self.resize_to_world(ctx, rect);
            self.clamp_camera(rect);

            // Mouse wheel: zoom centered on the cursor.
            let (scroll, hover) = ctx.input(|i| (i.raw_scroll_delta.y, i.pointer.hover_pos()));
            if scroll != 0.0 {
                if let Some(pos) = hover {
                    if rect.contains(pos) {
                        let factor = if scroll > 0.0 { 1.2 } else { 1.0 / 1.2 };
                        self.zoom_at(rect, pos, factor);
                        ctx.request_repaint();
                    }
                }
            }

            let response = ui.interact(rect, ui.id().with("grid"), Sense::click_and_drag());

            // Middle-drag or right-drag pans (allowed even while running).
            if response.dragged_by(egui::PointerButton::Middle)
                || response.dragged_by(egui::PointerButton::Secondary)
            {
                let delta = ctx.input(|i| i.pointer.delta());
                self.center -= delta / self.scale;
                self.clamp_camera(rect);
                ctx.request_repaint();
            }

            // Editing only while paused.
            if !self.running {
                let ctrl = ctx.input(|i| i.modifiers.command);

                // Show resize cursor when hovering a selection edge.
                if !ctrl && self.selection.is_some() && self.selection_edge.is_none() {
                    self.hovered_edge = self.hover_selection_edge(rect, response.hover_pos());
                    if let Some(edge) = self.hovered_edge {
                        ctx.output_mut(|o| o.cursor_icon = Self::edge_cursor(edge));
                    }
                } else {
                    self.hovered_edge = None;
                }

                if ctrl {
                    // Ctrl+drag starts a fresh rectangular selection; Ctrl+click clears it.
                    self.selection_edge = None;
                    if response.drag_started_by(egui::PointerButton::Primary) {
                        self.selection = None;
                        self.selection_start = response.interact_pointer_pos();
                        self.selection_current = response.interact_pointer_pos();
                    } else if response.dragged_by(egui::PointerButton::Primary) {
                        self.selection_current = response.interact_pointer_pos();
                    } else if response.drag_stopped() {
                        self.finish_selection(rect);
                    }
                    if response.clicked_by(egui::PointerButton::Primary) {
                        self.selection = None;
                        self.selection_start = None;
                        self.selection_current = None;
                    }
                } else if let Some(edge) = self.selection_edge {
                    // Dragging a selection edge resizes it.
                    if !ctx.input(|i| i.pointer.button_down(egui::PointerButton::Primary)) {
                        self.selection_edge = None;
                    } else if response.dragged_by(egui::PointerButton::Primary) {
                        if let Some(pos) = response.interact_pointer_pos() {
                            self.resize_selection(edge, rect, pos);
                        }
                    }
                } else if response.is_pointer_button_down_on()
                    && self.hovered_edge.is_some()
                    && ctx.input(|i| i.pointer.button_down(egui::PointerButton::Primary))
                {
                    // Start resizing as soon as the edge is pressed. The drag
                    // threshold would otherwise move the pointer out of the hover
                    // margin before `drag_started_by` fires.
                    self.selection_edge = self.hovered_edge;
                } else if response.drag_started_by(egui::PointerButton::Primary) {
                    self.drawing = true;
                    if let Some(pos) = response.interact_pointer_pos() {
                        self.clear_selection_if_outside(rect, pos);
                        if let Some((x, y)) = self.cell_at(rect, pos) {
                            self.paint_at(x, y, false);
                            self.last_mouse_pos = Some((x, y));
                        }
                    }
                } else if response.dragged_by(egui::PointerButton::Primary) {
                    if let Some(pos) = response.interact_pointer_pos() {
                        if let Some((x, y)) = self.cell_at(rect, pos) {
                            if let Some((lx, ly)) = self.last_mouse_pos {
                                for (cx, cy) in line_cells(lx, ly, x, y) {
                                    self.paint_at(cx, cy, false);
                                }
                            } else {
                                self.paint_at(x, y, false);
                            }
                            self.last_mouse_pos = Some((x, y));
                        }
                    }
                } else if response.drag_stopped() && self.drawing {
                    self.drawing = false;
                    self.last_mouse_pos = None;
                    self.flush_edits();
                }

                // Right-click erases a single cell (and clears selection if clicked outside it).
                if response.clicked_by(egui::PointerButton::Secondary) {
                    if let Some(pos) = response.interact_pointer_pos() {
                        self.clear_selection_if_outside(rect, pos);
                        if let Some((x, y)) = self.cell_at(rect, pos) {
                            self.paint_at(x, y, true);
                            self.flush_edits();
                        }
                    }
                }

                // Left-click paints a single cell (and clears selection if clicked outside it).
                if response.clicked_by(egui::PointerButton::Primary) && !self.drawing && !ctrl {
                    if let Some(pos) = response.interact_pointer_pos() {
                        self.clear_selection_if_outside(rect, pos);
                        if let Some((x, y)) = self.cell_at(rect, pos) {
                            self.paint_at(x, y, false);
                            self.flush_edits();
                        }
                    }
                }
            }

            let cb = GolCallback {
                sim: Arc::clone(&self.sim),
                center: [self.center.x, self.center.y],
                viewport_px: [rect.width() * ppp, rect.height() * ppp],
                scale_px: self.scale * ppp,
                wrap: self.wrap,
            };
            self.flush_edits();
            let callback = eframe::egui_wgpu::Callback::new_paint_callback(rect, cb);
            ui.painter().add(callback);

            // Selection overlay (paused only).
            if !self.running && (self.selection.is_some() || self.selection_start.is_some()) {
                self.draw_selection(ui, rect);
            }

            // Transient export status message.
            if let Some((msg, at)) = &self.export_status {
                if at.elapsed() < Duration::from_secs(4) {
                    let painter = ui.painter();
                    let galley = painter.layout_no_wrap(
                        msg.clone(),
                        egui::FontId::monospace(12.0),
                        egui::Color32::from_rgb(200, 235, 255),
                    );
                    let pos = egui::pos2(
                        rect.center().x - galley.size().x * 0.5,
                        rect.bottom() - 42.0,
                    );
                    painter.rect_filled(
                        egui::Rect::from_min_size(
                            pos - egui::vec2(10.0, 6.0),
                            galley.size() + egui::vec2(20.0, 12.0),
                        ),
                        4.0,
                        egui::Color32::from_black_alpha(200),
                    );
                    painter.galley(pos - egui::vec2(4.0, 0.0), galley, egui::Color32::WHITE);
                } else {
                    self.export_status = None;
                }
            }
        });

        if self.running
            || self.drawing
            || self.sim.lock().unwrap().has_pending_readback()
            || self.sim.lock().unwrap().png_readback.is_some()
            || self.selection_start.is_some()
            || self.selection_edge.is_some()
        {
            ctx.request_repaint();
        }
    }
}

#[derive(Clone)]
struct GolCallback {
    sim: Arc<Mutex<Sim>>,
    center: [f32; 2],
    viewport_px: [f32; 2],
    scale_px: f32,
    wrap: bool,
}

impl eframe::egui_wgpu::CallbackTrait for GolCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _screen_descriptor: &eframe::egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        _callback_resources: &mut eframe::egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        self.sim
            .lock()
            .unwrap()
            .set_camera(self.center, self.viewport_px, self.scale_px, self.wrap);
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        _callback_resources: &eframe::egui_wgpu::CallbackResources,
    ) {
        self.sim.lock().unwrap().render(render_pass);
    }
}

fn copy_image_to_clipboard(width: u32, height: u32, rgba: Vec<u8>) -> Result<(), String> {
    let mut clipboard = arboard::Clipboard::new().map_err(|e| e.to_string())?;
    clipboard
        .set_image(arboard::ImageData {
            width: width as usize,
            height: height as usize,
            bytes: std::borrow::Cow::Owned(rgba),
        })
        .map_err(|e| e.to_string())
}

fn next_download_path() -> std::path::PathBuf {
    let mut dir = std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    dir.push("Downloads");
    std::fs::create_dir_all(&dir).ok();
    for n in 1.. {
        let p = dir.join(format!("gol_img_{n}.png"));
        if !p.exists() {
            return p;
        }
    }
    unreachable!()
}

fn line_cells(x0: i32, y0: i32, x1: i32, y1: i32) -> Vec<(i32, i32)> {
    let mut pts = Vec::new();
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;
    let mut x = x0;
    let mut y = y0;
    loop {
        pts.push((x, y));
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
    pts
}

const DEFAULT_SCRIPT: &str = r#"
// Timeline script. Runs once at load, then onStep(t) is called each generation.
// Use setRule("B3/S23"), setRuleEx(birthMask, surviveMask), setPalette("Neon"), log(...).

function onStep(t) {
    // Example: switch to HighLife after 500 generations.
    // if (t === 500) {
    //     setRule("B36/S23");
    //     setPalette("Fire");
    // }
}
"#;
