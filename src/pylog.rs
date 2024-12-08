use std::{fmt::Arguments, mem, sync::atomic::{AtomicUsize, Ordering}};

pub use std::{format_args, module_path, panic::Location};

use anyhow::anyhow;
use log::{LevelFilter, Log, Record, Metadata};
pub use log::{Level, STATIC_MAX_LEVEL};


struct NopLogger;

impl Log for NopLogger {
    fn enabled(&self, _: &Metadata) -> bool {
        false
    }

    fn log(&self, _: &Record) {}
    fn flush(&self) {}
}

static mut LOGGER: &dyn Log = &NopLogger{};

static STATE: AtomicUsize = AtomicUsize::new(0);
const DISABLED: usize = 0;
const ENABLING: usize = 1;
const ENABLED: usize = 2;

static MAX_LOG_LEVEL_FILTER: AtomicUsize = AtomicUsize::new(0);

pub fn max_level() -> LevelFilter {
    unsafe { mem::transmute(MAX_LOG_LEVEL_FILTER.load(Ordering::Relaxed)) }
}

fn set_max_level(level: LevelFilter) {
    MAX_LOG_LEVEL_FILTER.store(level as usize, Ordering::Relaxed);
}

pub fn enable_log(filter: LevelFilter) -> anyhow::Result<()> {
    match STATE.compare_exchange(
        DISABLED,
        ENABLING,
        Ordering::Acquire,
        Ordering::Relaxed,
    ) {
        Ok(DISABLED) => {
            set_max_level(filter);
            let pyo3_logger = Box::new(pyo3_log::Logger::default().filter(filter));
            pyo3_logger.reset_handle();
            unsafe {
                LOGGER = Box::leak(pyo3_logger)
            }
            STATE.store(ENABLED, Ordering::Release);
            Ok(())
        }
        Err(ENABLING) => {
            while STATE.load(Ordering::Relaxed) == ENABLING {
                std::hint::spin_loop();
            }
            Err(anyhow!("enable pylog error"))
        }
        _ => Ok(()),
    }
}

pub fn logger() -> &'static dyn Log {
    if STATE.load(Ordering::Acquire) != ENABLED {
        static NOP: NopLogger = NopLogger;
        &NOP
    } else {
        unsafe { LOGGER }
    }
}

pub fn log(
    args: Arguments,
    level: Level,
    &(target, module_path, loc): &(&str, &'static str, &'static Location),
)
{
    let mut builder = Record::builder();

    builder
        .args(args)
        .level(level)
        .target(target)
        .module_path_static(Some(module_path))
        .file_static(Some(loc.file()))
        .line(Some(loc.line()));

    let r = builder.build();
    logger().log(&r);
}

#[macro_export]
macro_rules! pylog {
    // log!(target: "my_target", Level::Info, "a {} event", "log");
    (target: $target:expr, $lvl:expr, $($arg:tt)+) => ({
        let lvl = $lvl;
        if lvl <= $crate::pylog::STATIC_MAX_LEVEL && lvl <= $crate::pylog::max_level() {
            $crate::pylog::log(
                $crate::pylog::format_args!($($arg)+),
                lvl,
                &($target, $crate::pylog::module_path!(), $crate::pylog::Location::caller())
            );
        }
    });

    // log!(Level::Info, "a log event")
    ($lvl:expr, $($arg:tt)+) => ($crate::pylog!(target: $crate::pylog::module_path!(), $lvl, $($arg)+));
}

#[macro_export]
macro_rules! pyinfo {
    (target: $target:expr, $($arg:tt)+) => ($crate::pylog!(target: $target, $crate::pylog::Level::Info, $($arg)+));

    ($($arg:tt)+) => ($crate::pylog!($crate::pylog::Level::Info, $($arg)+))
}

#[macro_export]
macro_rules! pywarn {
    (target: $target:expr, $($arg:tt)+) => ($crate::pylog!(target: $target, $crate::pylog::Level::Warn, $($arg)+));

    ($($arg:tt)+) => ($crate::pylog!($crate::pylog::Level::Warn, $($arg)+))
}

#[macro_export]
macro_rules! pyerror {
    (target: $target:expr, $($arg:tt)+) => ($crate::pylog!(target: $target, $crate::pylog::Level::Error, $($arg)+));

    ($($arg:tt)+) => ($crate::pylog!($crate::pylog::Level::Error, $($arg)+))
}

#[macro_export]
macro_rules! pydebug {
    (target: $target:expr, $($arg:tt)+) => ($crate::pylog!(target: $target, $crate::pylog::Level::Debug, $($arg)+));

    ($($arg:tt)+) => ($crate::pylog!($crate::pylog::Level::Debug, $($arg)+))
}