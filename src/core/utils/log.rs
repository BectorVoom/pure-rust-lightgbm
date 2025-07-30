/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

use std::cell::RefCell;

/// Logging levels matching the C++ LightGBM implementation.
/// Higher values indicate more verbose logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// Fatal error level - terminates the program
    Fatal = -1,
    /// Warning level - indicates potential issues
    Warning = 0,
    /// Information level - general information messages
    Info = 1,
    /// Debug level - detailed debugging information
    Debug = 2,
}

impl From<i32> for LogLevel {
    fn from(value: i32) -> Self {
        match value {
            -1 => LogLevel::Fatal,
            0 => LogLevel::Warning,
            1 => LogLevel::Info,
            2 => LogLevel::Debug,
            _ => LogLevel::Info, // Default to Info for invalid values
        }
    }
}

/// Type alias for logging callback functions.
pub type LogCallback = fn(&str);

thread_local! {
    static LOG_LEVEL: RefCell<LogLevel> = RefCell::new(LogLevel::Info);
    static LOG_CALLBACK: RefCell<Option<LogCallback>> = RefCell::new(None);
}

/// Central logging facility for LightGBM operations.
/// Provides thread-local log level configuration and callback support.
#[derive(Debug)]
pub struct Log;

impl Log {
    /// Sets the current thread's logging level.
    pub fn reset_log_level(level: LogLevel) {
        LOG_LEVEL.with(|l| *l.borrow_mut() = level);
    }

    /// Sets the logging callback function for the current thread.
    /// If None, logs go to stdout; if Some, logs go to the callback.
    pub fn reset_callback(callback: Option<LogCallback>) {
        LOG_CALLBACK.with(|c| *c.borrow_mut() = callback);
    }

    /// Logs a debug message if the current log level allows it.
    pub fn debug(message: &str) {
        Self::write(LogLevel::Debug, "Debug", message);
    }

    /// Logs a formatted debug message using format_args!.
    pub fn debug_fmt(args: std::fmt::Arguments<'_>) {
        let message = format!("{}", args);
        Self::debug(&message);
    }

    /// Logs an info message if the current log level allows it.
    pub fn info(message: &str) {
        Self::write(LogLevel::Info, "Info", message);
    }

    /// Logs a formatted info message using format_args!.
    pub fn info_fmt(args: std::fmt::Arguments<'_>) {
        let message = format!("{}", args);
        Self::info(&message);
    }

    /// Logs a warning message if the current log level allows it.
    pub fn warning(message: &str) {
        Self::write(LogLevel::Warning, "Warning", message);
    }

    /// Logs a formatted warning message using format_args!.
    pub fn warning_fmt(args: std::fmt::Arguments<'_>) {
        let message = format!("{}", args);
        Self::warning(&message);
    }

    /// Logs a fatal error message and terminates the program with panic.
    /// This function never returns (marked with `!`).
    pub fn fatal(message: &str) -> ! {
        let formatted_message = format!("[LightGBM] [Fatal] {}", message);

        // Write to stderr and flush like the C++ version
        eprintln!("{}", formatted_message);

        // Panic with the message like C++ throws runtime_error
        panic!("{}", message);
    }
    
    /// Set verbosity level (equivalent to SetVerbosity in C++)
    /// Sets the verbosity level using integer values (matches C++ API).
    /// -1: Fatal, 0: Warning, 1: Info, 2+: Debug
    pub fn set_verbosity(verbosity: i32) {
        let level = match verbosity {
            i if i < 0 => LogLevel::Fatal,
            0 => LogLevel::Warning,
            1 => LogLevel::Info,
            _ => LogLevel::Debug,
        };
        
        LOG_LEVEL.with(|l| {
            *l.borrow_mut() = level;
        });
    }

    /// Logs a formatted fatal error message and terminates the program.
    /// This function never returns (marked with `!`).
    pub fn fatal_fmt(args: std::fmt::Arguments<'_>) -> ! {
        let message = format!("{}", args);
        Self::fatal(&message);
    }

    fn write(level: LogLevel, level_str: &str, message: &str) {
        let should_log = LOG_LEVEL.with(|l| level <= *l.borrow());

        if should_log {
            let callback = LOG_CALLBACK.with(|c| *c.borrow());

            match callback {
                Some(cb) => {
                    // Use callback like C++ version
                    let formatted = format!("[LightGBM] [{}] {}", level_str, message);
                    cb(&formatted);
                    cb("\n");
                }
                None => {
                    // Write to stdout like C++ version
                    println!("[LightGBM] [{}] {}", level_str, message);
                    use std::io::{self, Write};
                    io::stdout().flush().unwrap_or(());
                }
            }
        }
    }
}

// Macros that replicate the C++ CHECK functionality
/// Checks that a condition is true, panics with fatal error if false.
/// Equivalent to CHECK() macro in C++ LightGBM.
#[macro_export]
macro_rules! check {
    ($condition:expr) => {
        if !($condition) {
            $crate::core::utils::log::Log::fatal(&format!(
                "Check failed: {} at {}:{}",
                stringify!($condition),
                file!(),
                line!()
            ));
        }
    };
}

/// Checks that two values are equal, panics with fatal error if not.
/// Equivalent to CHECK_EQ() macro in C++ LightGBM.
#[macro_export]
macro_rules! check_eq {
    ($a:expr, $b:expr) => {
        check!($a == $b);
    };
}

/// Checks that two values are not equal, panics with fatal error if they are.
/// Equivalent to CHECK_NE() macro in C++ LightGBM.
#[macro_export]
macro_rules! check_ne {
    ($a:expr, $b:expr) => {
        check!($a != $b);
    };
}

/// Checks that first value is greater than or equal to second, panics if not.
/// Equivalent to CHECK_GE() macro in C++ LightGBM.
#[macro_export]
macro_rules! check_ge {
    ($a:expr, $b:expr) => {
        check!($a >= $b);
    };
}

/// Checks that first value is less than or equal to second, panics if not.
/// Equivalent to CHECK_LE() macro in C++ LightGBM.
#[macro_export]
macro_rules! check_le {
    ($a:expr, $b:expr) => {
        check!($a <= $b);
    };
}

/// Checks that first value is greater than second, panics if not.
/// Equivalent to CHECK_GT() macro in C++ LightGBM.
#[macro_export]
macro_rules! check_gt {
    ($a:expr, $b:expr) => {
        check!($a > $b);
    };
}

/// Checks that first value is less than second, panics if not.
/// Equivalent to CHECK_LT() macro in C++ LightGBM.
#[macro_export]
macro_rules! check_lt {
    ($a:expr, $b:expr) => {
        check!($a < $b);
    };
}

/// Checks that a raw pointer is not null, panics with fatal error if it is.
/// Equivalent to CHECK_NOTNULL() macro in C++ LightGBM.
#[macro_export]
macro_rules! check_notnull {
    ($pointer:expr) => {
        if $pointer.is_null() {
            $crate::core::utils::log::Log::fatal(&format!(
                "{} Can't be NULL at {}:{}",
                stringify!($pointer),
                file!(),
                line!()
            ));
        }
    };
}

// For Option types (more idiomatic Rust)
/// Checks that an Option is Some, panics with fatal error if None.
/// Rust-idiomatic equivalent to CHECK_NOTNULL for Option types.
#[macro_export]
macro_rules! check_some {
    ($option:expr) => {
        if $option.is_none() {
            $crate::core::utils::log::Log::fatal(&format!(
                "{} Can't be None at {}:{}",
                stringify!($option),
                file!(),
                line!()
            ));
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_conversion() {
        assert_eq!(LogLevel::from(-1), LogLevel::Fatal);
        assert_eq!(LogLevel::from(0), LogLevel::Warning);
        assert_eq!(LogLevel::from(1), LogLevel::Info);
        assert_eq!(LogLevel::from(2), LogLevel::Debug);
        assert_eq!(LogLevel::from(999), LogLevel::Info); // Default case
    }

    #[test]
    fn test_log_level_reset() {
        Log::reset_log_level(LogLevel::Debug);
        LOG_LEVEL.with(|l| assert_eq!(*l.borrow(), LogLevel::Debug));

        Log::reset_log_level(LogLevel::Warning);
        LOG_LEVEL.with(|l| assert_eq!(*l.borrow(), LogLevel::Warning));
    }

    #[test]
    fn test_callback_reset() {
        fn test_callback(_msg: &str) {}

        Log::reset_callback(Some(test_callback));
        LOG_CALLBACK.with(|c| assert!(c.borrow().is_some()));

        Log::reset_callback(None);
        LOG_CALLBACK.with(|c| assert!(c.borrow().is_none()));
    }

    #[test]
    fn test_log_ordering() {
        assert!(LogLevel::Fatal < LogLevel::Warning);
        assert!(LogLevel::Warning < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Debug);
    }

    #[test]
    fn test_basic_logging() {
        // Test that logging functions don't panic
        Log::debug("Test debug message");
        Log::info("Test info message");
        Log::warning("Test warning message");
    }

    #[test]
    #[should_panic(expected = "Test fatal message")]
    fn test_fatal_panic() {
        Log::fatal("Test fatal message");
    }

    #[test]
    fn test_check_macros() {
        // These should not panic
        check!(true);
        check_eq!(1, 1);
        check_ne!(1, 2);
        check_ge!(2, 1);
        check_le!(1, 2);
        check_gt!(2, 1);
        check_lt!(1, 2);
    }

    #[test]
    #[should_panic]
    fn test_check_failure() {
        check!(false);
    }

    #[test]
    fn test_formatted_logging() {
        Log::debug_fmt(format_args!("Debug: {}", 42));
        Log::info_fmt(format_args!("Info: {}", "test"));
        Log::warning_fmt(format_args!("Warning: {}", 3.14));
    }
}
