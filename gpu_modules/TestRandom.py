random_test = """
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>
#include <thrust/device_vector.h>
extern "C" {

__device__ float first(curandState_t *rand_state)
{
    return curand_uniform(rand_state);
}

__device__ float second(curandState_t *rand_state)
{
    float rand = first(rand_state);
    return rand;
}

__device__ float third(curandState_t *rand_state)
{
    float rand = second(rand_state);
    return rand;
}

__global__ void random_simulation(int *seed, float *input, float *output)
{
    // initiate the random sampler, must have.
    int iteration = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    curandState s;
    curand_init((*seed)* iteration, 0, 0, &s);

    // get the number of trials
    int num_trials      = (float)input[0];

    //printf("number of trials = %d", num_trials);

    // keep only the GPU nodes that satisfy the number of trial
    if (iteration>num_trials) return;

    float First = curand_uniform(&s);
    //float Second = first(&s);
    //float Third = second(&s);

    atomicAdd(&output[iteration],(float)First);
   // atomicAdd(&output[iteration+num_trials],(float)Second);
   // atomicAdd(&output[iteration+2*num_trials],(float)Third);

}
}
"""
