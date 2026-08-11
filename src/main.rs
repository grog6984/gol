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
use crate::sim::Sim;

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
    grid_sized: bool,
    show_quit_dialog: bool,
    last_mouse_move: Instant,
    cursor_hidden: bool,
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
            grid_sized: false,
            show_quit_dialog: false,
            last_mouse_move: Instant::now(),
            cursor_hidden: false,
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
        // Size the world exactly once to the initial screen resolution.
        // After that, viewport changes only move the camera; the world state never resizes.
        if !self.grid_sized {
            let w = (rect.width().ceil() as u32).max(1);
            let h = (rect.height().ceil() as u32).max(1);
            self.sim.lock().unwrap().resize(w, h);
            self.center = egui::Vec2::new(w as f32 * 0.5, h as f32 * 0.5);
            self.scale = 1.0;
            self.grid_sized = true;
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
                ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(self.fullscreen));
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
                    ui.label("Shortcuts: Space = run/pause  F1 = UI  F = fullscreen  1-5 = flip %");
                    ui.label("Up/Down = step count   Wheel = zoom   middle/right-drag = pan");
                    ui.label("Left-click/drag = paint   Right-click = erase   Q = quit");
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
            let rect = ui.available_rect_before_wrap();
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
                if response.drag_started_by(egui::PointerButton::Primary) {
                    self.drawing = true;
                    if let Some(pos) = response.interact_pointer_pos() {
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

                // Right-click erases a single cell.
                if response.clicked_by(egui::PointerButton::Secondary) {
                    if let Some(pos) = response.interact_pointer_pos() {
                        if let Some((x, y)) = self.cell_at(rect, pos) {
                            self.paint_at(x, y, true);
                            self.flush_edits();
                        }
                    }
                }

                // Left-click paints a single cell.
                if response.clicked_by(egui::PointerButton::Primary) && !self.drawing {
                    if let Some(pos) = response.interact_pointer_pos() {
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
        });

        if self.running
            || self.drawing
            || self.sim.lock().unwrap().has_pending_readback()
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
