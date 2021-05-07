import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np

mod = SourceModule("""
__global__ void diag_kernel(float *dest, int stride, int N)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < N) {
    float* p = (float*)((char*)dest + tid*stride) + 1;
        *(p+1) = 1.0f;
    }
}
""")

diag_kernel = mod.get_function("diag_kernel")

a = np.zeros((10,10), dtype=np.float32)
a_N = np.int32(a.shape[0])
a_stride = np.int32(a.strides[0])
a_bytes = a.size * a.dtype.itemsize
a_gpu = drv.mem_alloc(a_bytes)
drv.memcpy_htod(a_gpu, a)
tArgs               = [
    a_gpu,
    a_stride,
    a_N
]
diag_kernel(*tArgs, block=(32,1,1))
drv.memcpy_dtoh(a, a_gpu)

print(a)
