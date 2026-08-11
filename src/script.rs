use crate::rules::Rule;
use rquickjs::{Context, Function, Runtime};
use std::cell::RefCell;

thread_local! {
    static NEXT_RULE: RefCell<Option<Rule>> = RefCell::new(None);
    static NEXT_PALETTE: RefCell<Option<String>> = RefCell::new(None);
}

#[derive(Default, Clone, Debug)]
pub struct ScriptResult {
    pub rule: Option<Rule>,
    pub palette: Option<String>,
}

pub struct ScriptEngine {
    #[allow(dead_code)]
    runtime: Runtime,
    ctx: Context,
}

impl ScriptEngine {
    pub fn new() -> Result<Self, String> {
        let runtime = Runtime::new().map_err(|e| format!("runtime: {e}"))?;
        let ctx = Context::builder()
            .build(&runtime)
            .map_err(|e| format!("context: {e}"))?;

        ctx.with(|ctx| {
            let globals = ctx.globals();

            globals
                .set(
                    "setRule",
                    Function::new(ctx.clone(), |rule: String| {
                        if let Some(r) = Rule::parse(&rule) {
                            NEXT_RULE.with(|c| *c.borrow_mut() = Some(r));
                        }
                    })
                    .unwrap(),
                )
                .unwrap();

            globals
                .set(
                    "setRuleEx",
                    Function::new(ctx.clone(), |birth: u32, survive: u32| {
                        NEXT_RULE.with(|c| *c.borrow_mut() = Some(Rule::new(birth, survive)));
                    })
                    .unwrap(),
                )
                .unwrap();

            globals
                .set(
                    "setPalette",
                    Function::new(ctx.clone(), |name: String| {
                        NEXT_PALETTE.with(|c| *c.borrow_mut() = Some(name));
                    })
                    .unwrap(),
                )
                .unwrap();

            globals
                .set(
                    "log",
                    Function::new(ctx.clone(), |msg: String| {
                        println!("[script] {msg}");
                    })
                    .unwrap(),
                )
                .unwrap();

            Ok(())
        })
        .map_err(|e: rquickjs::Error| format!("globals: {e}"))?;

        Ok(Self { runtime, ctx })
    }

    pub fn set_source(&mut self, source: &str) -> Result<(), String> {
        let mut options = rquickjs::context::EvalOptions::default();
        options.global = true;
        self.ctx
            .with(|ctx| ctx.eval_with_options::<(), _>(source, options))
            .map_err(|e| format!("{e:?}"))?;
        Ok(())
    }

    pub fn on_step(&self, generation: u64) -> Result<ScriptResult, String> {
        NEXT_RULE.with(|c| *c.borrow_mut() = None);
        NEXT_PALETTE.with(|c| *c.borrow_mut() = None);

        self.ctx
            .with(|ctx| {
                let globals = ctx.globals();
                if let Ok(func) = globals.get::<_, Function>("onStep") {
                    let _ = func.call::<(u64,), ()>((generation,));
                }
                Ok(())
            })
            .map_err(|e: rquickjs::Error| format!("{e}"))?;

        let rule = NEXT_RULE.with(|c| *c.borrow());
        let palette = NEXT_PALETTE.with(|c| c.borrow().clone());
        Ok(ScriptResult { rule, palette })
    }
}
