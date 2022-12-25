import kraft
import kraft.device


def main():
    t = kraft.Tensor([1, 2, 3], device=kraft.device.get_gpu_device())

    print(kraft.device.is_gpu_available())
    print(t)
    print(t.shape)


if __name__ == "__main__":
    main()
