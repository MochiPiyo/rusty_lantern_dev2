use std::sync::Mutex;

use lazy_static::lazy_static;

use crate::logger::Logger;


pub struct MachineConfig {
    pub enable_multi_thread: bool,
    pub multi_thread_threshold: usize,
    pub thread_num: Option<usize>, // None is all

    pub logger: Logger,

    // pub wgpu_device: Vec<usize, WgpuDevice>, 
    // pub cuda_device: 
}
lazy_static! {
    pub static ref MACHINE_CONFIG: Mutex<MachineConfig> = Mutex::new(MachineConfig {
        enable_multi_thread: true,
        multi_thread_threshold: 1_000_000,
        thread_num: None,

        logger: Logger::new(),
    });
}