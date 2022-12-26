import kraft
import kraft.device
import kraft.optim
import kraft.autograd.ops


def main():
    device = kraft.device.get_gpu_device()

    data = kraft.randn([4, 16, 128, 128], dtype=kraft.float32, requires_grad=True, device=device)

    w = kraft.randn([8, 16, 3, 3], dtype=kraft.float32, requires_grad=True, device=device)
    b = kraft.randn([8], dtype=kraft.float32, requires_grad=True, device=device)

    out = kraft.autograd.ops.conv2d(data, W=w, b=b)
    out = kraft.autograd.ops.max_pool2d(out, kernel_size=2)
    out = out.sum()
    out.backward()

    print(data.grad)


if __name__ == "__main__":
    main()
