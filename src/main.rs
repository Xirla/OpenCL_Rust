use opencl3;
use std;

const PLATFORM_ID: usize = 0;
const DEVICE_ID: usize = 0;

fn main() {

  // get platforms
  let platforms = opencl3::platform::get_platforms().expect(
    "Failed to get platforms."
  );
  let platform = platforms[PLATFORM_ID];
  println!("Using platform: {}", platform.name().unwrap());

  // get devices
  let devices = platform.get_devices(opencl3::device::CL_DEVICE_TYPE_ALL).expect(
    "Failed to get devices."
  );
  let device = opencl3::device::Device::new(devices[DEVICE_ID]);
  println!("Using device: {}", device.name().unwrap());

  // create context
  let context = opencl3::context::Context::from_device(&device).expect(
    "Failed to create context."
  );

  let source = std::fs::read_to_string("./src/vadd.cl").unwrap();

  // create program
  let mut program = opencl3::program::Program::create_from_source(&context, &source).expect(
    "Failed to create program."
  );
  
  // build program
  program.build(&devices, "").expect(
    "Failed to build program."
  );

  // create kernel
  let kernel = opencl3::kernel::Kernel::create(&program, "vadd").expect(
    "Failed to create kernel."
  );

  // create command queue
  let command_queue = opencl3::command_queue::CommandQueue::create(&context, device.id(), opencl3::command_queue::CL_QUEUE_PROFILING_ENABLE).expect(
    "Failed to create command queue."
  );

  let a: [i32; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  let b: [i32; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  let mut c: [i32; 10] = [0; 10];

  // create buffer
  let mut a_buffer = opencl3::memory::Buffer::<opencl3::types::cl_int>::create(&context, opencl3::memory::CL_MEM_READ_ONLY, 10, std::ptr::null_mut()).expect(
    "Failed to create buffer a."
  );
  let mut b_buffer = opencl3::memory::Buffer::<opencl3::types::cl_int>::create(&context, opencl3::memory::CL_MEM_READ_ONLY, 10, std::ptr::null_mut()).expect(
    "Failed to create buffer b."
  );
  let mut c_buffer = opencl3::memory::Buffer::<opencl3::types::cl_int>::create(&context, opencl3::memory::CL_MEM_WRITE_ONLY, 10, std::ptr::null_mut()).expect(
    "Failed to create buffer c."
  );

  // write buffer
  command_queue.enqueue_write_buffer(&mut a_buffer, opencl3::types::CL_TRUE, 0, &a, &[]).expect(
    "Failed to write buffer a."
  );
  command_queue.enqueue_write_buffer(&mut b_buffer, opencl3::types::CL_TRUE, 0, &b, &[]).expect(
    "Failed to write buffer b."
  );

  // execute kernel
  let kernel_event = opencl3::kernel::ExecuteKernel::new(&kernel)
  .set_arg(&a_buffer)
  .set_arg(&b_buffer)
  .set_arg(&c_buffer)
  .set_global_work_size(10)
  .set_local_work_size(1)
  .set_global_work_offset(0)
  .enqueue_nd_range(&command_queue)
  .expect("Failed to enqueue kernel execution.");

  command_queue.enqueue_read_buffer(&mut c_buffer, opencl3::types::CL_TRUE, 0, &mut c, &[kernel_event.get()]).expect(
    "Failed to read buffer c."
  );

  for (i, val) in c.iter().enumerate() {
    println!("c[{}] = {}", i, val);
  }
}
