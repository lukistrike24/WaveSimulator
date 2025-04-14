# wave_simulator.py
# WARNING: All explicit error handling has been removed for brevity.
# This code is fragile and not suitable for production use.
import numpy as np
import pyopencl as cl
import time
import os
import math

# --- Defaults ---
DEFAULT_NX = 1000
DEFAULT_NY = 1000
DEFAULT_MAX_TIME_STEPS = 20000
DEFAULT_VIZ_INTERVAL = 25
DEFAULT_C = 1.0
DEFAULT_DX = 0.1
DEFAULT_COURANT_NUM = 0.1
DEFAULT_NPML = 20
DEFAULT_MAX_DAMPING = 80.0
DEFAULT_DAMPING_POWER = 4

# --- Kernel Code (Unchanged) ---
kernel_code = """
#define M_PI_F 3.14159265358979323846f
__kernel void wave_step_general(
    const int nx, const int ny, const float coupling, const float dt,
    const float dt_sq, const float t, const int num_sources,
    __global const int* src_x_coords, __global const int* src_y_coords,
    __global const float* src_amps, __global const float* src_omegas,
    __global const int* obstacle_mask, __global const float* gamma,
    __global const float* u_prev, __global const float* u_curr,
    __global float* u_next)
{
    int i = get_global_id(0); int j = get_global_id(1); int idx = i + j * nx;
    u_next[idx] = 0.0f;
    int material = obstacle_mask[idx];

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1 && material == 0) {
        float gamma_val = gamma[idx];
        float dt_half = dt / 2.0f;
        float factor1 = 1.0f, factor2 = 1.0f;
        if (gamma_val > 1e-9f) {
             factor1 = 1.0f / (1.0f + gamma_val * dt_half);
             factor2 = 1.0f - gamma_val * dt_half;
        }
        int idx_ip = (i + 1) + j * nx; int idx_im = (i - 1) + j * nx;
        int idx_jp = i + (j + 1) * nx; int idx_jm = i + (j - 1) * nx;
        float laplacian = u_curr[idx_ip] + u_curr[idx_im] +
                          u_curr[idx_jp] + u_curr[idx_jm] - 4.0f * u_curr[idx];
        u_next[idx] = factor1 * (2.0f * u_curr[idx] - factor2 * u_prev[idx] + coupling * laplacian);

        for (int s = 0; s < num_sources; ++s) {
            if (i == src_x_coords[s] && j == src_y_coords[s]) {
                u_next[idx] += factor1 * dt_sq * src_amps[s] * sin(src_omegas[s] * t);
            }
        }
    }
}
"""

# --- Simulation Logic Class (Simplified) ---
class SimulatorLogic:
    # Removed error_callback
    def __init__(self, config, status_callback, frame_callback):
        self.config = config
        self.status_callback = status_callback
        self.frame_callback = frame_callback
        self._is_running = True
        self.ctx = None
        self.queue = None
        self.prg = None
        self.wave_step_kernel = None
        self.buffers_to_release = []

    def stop(self):
        self._is_running = False
        self.status_callback("Stop signal received.")

    def setup_opencl(self):
        # Removed try...except, assumes success
        self.status_callback("Setting up OpenCL context...")
        self.ctx = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.ctx)
        device_name = self.queue.device.name
        self.status_callback(f"Using OpenCL device: {device_name}")
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.wave_step_kernel = self.prg.wave_step_general
        # return True # Implicitly succeeds

    def release_resources(self):
        self.status_callback("Releasing OpenCL resources...")
        for buf in self.buffers_to_release:
             # Removed try...except, assumes buf exists and release works
             if buf: buf.release()
        self.buffers_to_release.clear()
        self.wave_step_kernel = None
        self.prg = None
        self.queue = None
        self.ctx = None
        self.status_callback("OpenCL resources released.")

    def run(self):
        sim_start_time = time.time()
        self._is_running = True
        self.setup_opencl() # Assume it works

        # --- Extract Parameters ---
        NX = self.config.get('nx', DEFAULT_NX); NY = self.config.get('ny', DEFAULT_NY)
        MAX_TIME_STEPS = self.config.get('max_time_steps', DEFAULT_MAX_TIME_STEPS)
        VIZ_INTERVAL = self.config.get('viz_interval', DEFAULT_VIZ_INTERVAL)
        C = self.config.get('c', DEFAULT_C); DX = self.config.get('dx', DEFAULT_DX)
        COURANT_NUM = self.config.get('courant_num', DEFAULT_COURANT_NUM)
        NPML = self.config.get('npml', DEFAULT_NPML)
        MAX_DAMPING = self.config.get('max_damping', DEFAULT_MAX_DAMPING)
        DAMPING_POWER = self.config.get('damping_power', DEFAULT_DAMPING_POWER)
        building_blocks = self.config.get('building_blocks', [])

        # --- Calculated Parameters (Assume valid config) ---
        DT = COURANT_NUM * DX / (C * math.sqrt(2)); C_SQ = C * C
        DT_SQ = DT * DT; COUPLING = (C * DT / DX)**2

        # --- Pre-process Building Blocks ---
        obstacle_mask_h = np.zeros((NY, NX), dtype=np.int32)
        src_pos_x, src_pos_y, src_amps, src_omegas = [], [], [], []
        for block in building_blocks:
            if block['type'] == 'obstacle' and block['shape'] == 'rectangle':
                pos, size = block.get('position', (0,0)), block.get('size', (1,1))
                x_min, y_min = pos; width, height = size
                x_start, y_start = max(0, x_min), max(0, y_min)
                x_end, y_end = min(NX, x_min + width), min(NY, y_min + height)
                if x_start < x_end and y_start < y_end:
                    obstacle_mask_h[y_start:y_end, x_start:x_end] = block.get('material_type', 1)
            elif block['type'] == 'source' and block.get('source_type') == 'point':
                pos = block.get('position', (NX//2, NY//2)); amp = block.get('amplitude', 0.0); freq = block.get('frequency', 1.0)
                pos_x, pos_y = max(0, min(NX - 1, pos[0])), max(0, min(NY - 1, pos[1]))
                src_pos_x.append(pos_x); src_pos_y.append(pos_y); src_amps.append(amp); src_omegas.append(2.0 * math.pi * freq)

        num_sources = len(src_pos_x)
        src_x_h = np.array(src_pos_x, dtype=np.int32) if num_sources > 0 else np.array([0], dtype=np.int32)
        src_y_h = np.array(src_pos_y, dtype=np.int32) if num_sources > 0 else np.array([0], dtype=np.int32)
        src_amp_h = np.array(src_amps, dtype=np.float32) if num_sources > 0 else np.array([0.0], dtype=np.float32)
        src_omega_h = np.array(src_omegas, dtype=np.float32) if num_sources > 0 else np.array([0.0], dtype=np.float32)

        # --- Calculate Damping Profile (Gamma) ---
        gamma_h = np.zeros((NY, NX), dtype=np.float32)
        if NPML > 0:
            for i in range(NX):
                 for j in range(NY):
                     dist_x = 0.0; dist_y = 0.0
                     if i < NPML: dist_x = (NPML - i) / NPML
                     elif i >= NX - NPML: dist_x = (i - (NX - NPML - 1)) / NPML
                     if j < NPML: dist_y = (NPML - j) / NPML
                     elif j >= NY - NPML: dist_y = (j - (NY - NPML - 1)) / NPML
                     dist = max(dist_x, dist_y)
                     if dist > 0: gamma_h[j, i] = MAX_DAMPING * (dist ** DAMPING_POWER)

        # --- Initialize Grids & Buffers ---
        mf = cl.mem_flags
        u_prev_h = np.zeros((NY, NX), dtype=np.float32)
        u_curr_h = np.zeros((NY, NX), dtype=np.float32) # Host buffer for results

        # Assume buffer creation works
        self.status_callback("Creating OpenCL buffers...")
        gamma_d = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gamma_h)
        obstacle_mask_d = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=obstacle_mask_h)
        u_prev_d = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=u_prev_h)
        u_curr_d = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=u_curr_h)
        u_next_d = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=u_curr_h.nbytes)
        src_x_d = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=src_x_h)
        src_y_d = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=src_y_h)
        src_amp_d = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=src_amp_h)
        src_omega_d = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=src_omega_h)
        self.buffers_to_release = [gamma_d, obstacle_mask_d, u_prev_d, u_curr_d, u_next_d,
                                   src_x_d, src_y_d, src_amp_d, src_omega_d]

        d_buffers = [u_prev_d, u_curr_d, u_next_d]

        # --- Main Simulation Loop ---
        self.status_callback("Starting simulation loop...")
        for k in range(MAX_TIME_STEPS):
            if not self._is_running: break
            current_time = k * DT

            # Removed try...except block for kernel execution/copy
            self.wave_step_kernel(self.queue, (NX, NY), None,
                             np.int32(NX), np.int32(NY), np.float32(COUPLING),
                             np.float32(DT), np.float32(DT_SQ), np.float32(current_time),
                             np.int32(num_sources), src_x_d, src_y_d, src_amp_d, src_omega_d,
                             obstacle_mask_d, gamma_d,
                             d_buffers[0], d_buffers[1], d_buffers[2]).wait()

            d_buffers = [d_buffers[1], d_buffers[2], d_buffers[0]] # Cycle buffers

            if k % VIZ_INTERVAL == 0 or k == MAX_TIME_STEPS - 1:
                cl.enqueue_copy(self.queue, u_curr_h, d_buffers[1]).wait() # Assume copy works
                self.frame_callback(u_curr_h.copy(), current_time, k)

        # --- End of Loop ---
        sim_end_time = time.time()
        final_status = f"Simulation finished/stopped. Total time: {sim_end_time - sim_start_time:.2f}s"
        self.status_callback(final_status)
        self.release_resources()

# --- Guard for direct execution (Simplified Test) ---
if __name__ == '__main__':
    print("SimulatorLogic - Simplified Test Run")
    def print_status(msg): print(f"[Status] {msg}")
    def print_frame(data, t, k): print(f"[Frame {k} @ t={t:.3f}] Shape: {data.shape}, Max Amp: {np.max(np.abs(data)):.4f}")
    # Removed print_error

    test_config = {
        'nx': 128, 'ny': 128, 'max_time_steps': 200, 'viz_interval': 20,
        'building_blocks': [
             {'type': 'source','source_type': 'point','position': (32, 64),'frequency': 5.0,'amplitude': 100.0},
             {'type': 'obstacle','shape': 'rectangle','position': (90, 50),'size': (5, 40),'material_type': 1}
        ]
    }
    # Removed error callback argument
    simulator = SimulatorLogic(test_config, print_status, print_frame)
    simulator.run()