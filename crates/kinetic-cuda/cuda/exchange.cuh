// Sums of partial outputs between two devices that each hold a share of a
// layer, without the host: each writes its values straight into the other's
// mailbox, then raises a flag there, and waits for the other's flag in its
// own. Every device runs the same exchanges in the same order, each
// numbered by a count both keep, so the flags only grow and a wait cannot
// confuse one exchange with another; the mailbox has two halves used in
// turn, so a device that runs ahead writes to the half its peer is done
// with.

#define EXCHANGE_THREADS 256

// What a device does with the peer's values.
#define EXCHANGE_ADD 0   // adds them to its own
#define EXCHANGE_TAKE 1  // replaces its own with them
#define EXCHANGE_KEEP 2  // keeps its own; it only gives

struct ExchangeArgs {
    float* y;
    // The peer's mailbox, flags and this device's, `cap` values a half.
    float* peer_box;
    unsigned* peer_flags;
    const float* own_box;
    const unsigned* own_flags;
    // The exchanges done, and the blocks of this one done.
    unsigned* count;
    unsigned cap;
    unsigned n;
    unsigned mode;
};

extern "C" __global__ void __launch_bounds__(EXCHANGE_THREADS)
    exchange(ExchangeArgs a) {
    __shared__ unsigned seq;
    // Every block reads the count before the last one moves it on.
    if (threadIdx.x == 0) seq = *reinterpret_cast<volatile unsigned*>(a.count) + 1;
    __syncthreads();
    const unsigned half = (seq & 1) * a.cap;
    const unsigned each = (a.n + gridDim.x - 1) / gridDim.x;
    const unsigned lo = blockIdx.x * each;
    const unsigned hi = min(a.n, lo + each);
    for (unsigned i = lo + threadIdx.x; i < hi; i += blockDim.x) a.peer_box[half + i] = a.y[i];
    // the values reach the peer before the flag that says they are there
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *reinterpret_cast<volatile unsigned*>(a.peer_flags + blockIdx.x) = seq;
        const volatile unsigned* flag = a.own_flags + blockIdx.x;
        while (static_cast<int>(*flag - seq) < 0) {
        }
        __threadfence();
    }
    __syncthreads();
    const volatile float* box = a.own_box + half;
    for (unsigned i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        if (a.mode == EXCHANGE_ADD) a.y[i] += box[i];
        else if (a.mode == EXCHANGE_TAKE) a.y[i] = box[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(a.count + 1, 1u) == gridDim.x - 1) {
            a.count[1] = 0;
            __threadfence();
            *reinterpret_cast<volatile unsigned*>(a.count) = seq;
        }
    }
}
