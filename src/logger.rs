use std::sync::Mutex;

use lazy_static::lazy_static;



lazy_static! {
    pub static ref LOGGER: Logger = Logger::new();
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

pub struct _Logger {
    mode: LoggerMode,
    history: Vec<Log>,
    // outputtype // console, text, GUI
}
pub struct Logger {
    logger: Mutex<_Logger>,
}
impl Logger {
    pub fn new() -> Self {
        let inner = _Logger {
            mode: LoggerMode::Debug,
            history: Vec::new(),
        };
        Logger {
            logger: Mutex::new(inner),
        }
    }

    fn on_update(&self) {
        // とりあえず，println!()
        let lock = self.logger.lock().unwrap();
        let string = match lock.history.last() {
            None => panic!("none"),
            Some(Log::Debug(s)) => s,
            Some(Log::Warning(s)) => s,
            Some(Log::Error(s)) => s,
            Some(Log::FatalError(s)) => s,
        };
        println!("{}", string);
    }

    // とりあえずformat!するためにStringで受ける。あとでマクロにする
    pub fn debug(&self, string: String) {
        self.logger.lock().unwrap().history.push(Log::Debug(string));
        self.on_update();
    }
    pub fn warning(&self, string: String) {
        self.logger.lock().unwrap().history.push(Log::Warning(string));
        self.on_update();
    }
    pub fn error(&self, string: String) {
        self.logger.lock().unwrap().history.push(Log::Error(string));
        self.on_update();
    }
    pub fn fatal_error(&self, string: String) {
        self.logger.lock().unwrap().history.push(Log::FatalError(string));
        self.on_update();
    }
}
