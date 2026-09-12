#pragma once
#include <cstdint>

// Config struct passed by value (copied to constant memory automatically)
struct KernelConfig {
    int rf_per_pulse;
    int meltdown_temp;
    int vent_divisor;
    int vent_base;
    int pipe_divisor;
    int pipe_base;
    int absorber_cooling; 
};

template <size_t W, size_t H>
void launch_gpu_batch_eval(
    const uint8_t* d_grids, 
    int batch_size, 
    KernelConfig config,
    long long* d_out_rf, 
    int* d_out_temp, 
    bool* d_out_melted
);