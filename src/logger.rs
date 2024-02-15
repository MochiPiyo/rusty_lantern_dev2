use std::sync::Mutex;

use lazy_static::lazy_static;



lazy_static! {
    pub static ref LOGGER: Mutex<Logger> = Mutex::new(Logger::new());
}

enum Log {
    Debug(String),
    Warning(String),
    Error(String),
    FatalError(String),
}
enum LoggerMode {
    Debug
}

pub struct Logger {
    mode: LoggerMode,
    history: Vec<Log>,
    // outputtype // console, text, GUI
}
impl Logger {
    pub fn new() -> Self {
        Self {
            mode: LoggerMode::Debug,
            history: Vec::new(),
        }
    }

    fn on_update(&self) {
        // とりあえず，println!()
        let string = match self.history.last() {
            None => panic!("none"),
            Some(Log::Debug(s)) => s,
            Some(Log::Warning(s)) => s,
            Some(Log::Error(s)) => s,
            Some(Log::FatalError(s)) => s,
        };
        println!("{}", string);
    }

    pub fn debug(&mut self, string: String) {
        self.history.push(Log::Debug(string));
        self.on_update();
    }
    pub fn warning(&mut self, string: String) {
        self.history.push(Log::Warning(string));
        self.on_update();
    }
    pub fn error(&mut self, string: String) {
        self.history.push(Log::Error(string));
        self.on_update();
    }
    pub fn fatal_error(&mut self, string: String) {
        self.history.push(Log::FatalError(string));
        self.on_update();
    }
}
