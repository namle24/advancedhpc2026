import numba
from numba import cuda
def main():
    if not cuda.is_available():
        print("CUDA is not available")
        return
    cuda.detect()
    device = cuda.select_device(0)
    dev_id = device.id
    dev_name = device.name.decode('utf-8') if isinstance(device.name, bytes) else device.name
    print("Device ID:", dev_id)
    print("Device Name:", dev_name)
    try:
        mp_count = device.MULTIPROCESSOR_COUNT
        print("Multiprocessors:", mp_count)
    except AttributeError:
        pass
    free_mem, total_mem = cuda.current_context().get_memory_info()
    print("Total Memory (bytes):", total_mem)
    print("Free Memory (bytes):", free_mem)
    print("Total Memory (GB):", total_mem / (1024 ** 3))
    print("Free Memory (GB):", free_mem / (1024 ** 3))
if __name__ == "__main__":
    main()
