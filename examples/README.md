# Examples

All example scripts rely on a shared accelerator helper that selects the active GPU backend at runtime.
Import :mod:`_accelerator` to access ``DEVICE`` (a :class:`torch.device`) and ``DEVICE_STR`` (a string alias)
so that both CUDA and HIP backends are supported out of the box::

    from _accelerator import DEVICE, DEVICE_STR

    pipeline = pipeline.to(DEVICE)
    generator = torch.Generator(device=DEVICE_STR)

When running on ROCm builds of PyTorch the helper automatically maps CUDA-only strings to the ``hip:N`` alias,
matching the logic used by the C++ extension. On CUDA builds ``cuda:N`` continues to be used.
