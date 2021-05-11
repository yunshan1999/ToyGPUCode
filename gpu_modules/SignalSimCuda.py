# #############################################################
# Signal simulation details shall be written here!
# this currently is just a test and demonstration!
# #############################################################
# Created @ 2021-03-16
# by Qing Lin
# email: qinglin@ustc.edu.cn
###############################################################

pandax4t_signal_sim = """
#include <curand_kernel.h>
#include <stdio.h>
#include <thrust/device_vector.h>
extern "C" {

__global__ void test(int *seed, int *input)
{
    // initiate the random sampler, must have.
    int iteration = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    curandState s;
    curand_init((*seed)* iteration, 0, 0, &s);
    
    // get the number of trials
    int num_trials      = input[0];
    
    //printf("number of trials = %d", num_trials);
    
    // keep only the GPU nodes that satisfy the number of trial
    if (iteration>num_trials) return;
    
    
    // print
    printf("This is the %d-th GPU thread!\\n", iteration);
    return;
}


}
"""




