from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import platform


@dataclass(frozen=True)
class RuntimeInfo:
    backend: str
    device: str
    uses_vram: bool
    python_version: str
    platform_name: str
    torch_available: bool
    torch_version: str | None = None
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_device_name: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_runtime() -> RuntimeInfo:
    python_version = platform.python_version()
    platform_name = platform.platform()

    if importlib.util.find_spec("torch") is None:
        return RuntimeInfo(
            backend="python",
            device="cpu",
            uses_vram=False,
            python_version=python_version,
            platform_name=platform_name,
            torch_available=False,
            note="torch belum terpasang, jadi runtime memakai Python CPU biasa.",
        )

    try:
        import torch  # type: ignore
    except Exception as error:
        return RuntimeInfo(
            backend="python",
            device="cpu",
            uses_vram=False,
            python_version=python_version,
            platform_name=platform_name,
            torch_available=False,
            note=f"torch terdeteksi tapi gagal diimport: {error}",
        )

    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        return RuntimeInfo(
            backend="torch",
            device="cuda",
            uses_vram=True,
            python_version=python_version,
            platform_name=platform_name,
            torch_available=True,
            torch_version=str(torch.__version__),
            cuda_available=True,
            cuda_device_count=int(torch.cuda.device_count()),
            cuda_device_name=str(torch.cuda.get_device_name(0)),
            note="torch dan CUDA aktif. Model bisa memakai VRAM GPU.",
        )

    return RuntimeInfo(
        backend="torch",
        device="cpu",
        uses_vram=False,
        python_version=python_version,
        platform_name=platform_name,
        torch_available=True,
        torch_version=str(torch.__version__),
        cuda_available=False,
        cuda_device_count=0,
        cuda_device_name=None,
        note="torch tersedia, tetapi CUDA tidak aktif. Model akan tetap berjalan di CPU.",
    )


def format_runtime_info(runtime: RuntimeInfo) -> str:
    lines = [
        "Runtime Summary",
        f"Backend: {runtime.backend}",
        f"Device: {runtime.device}",
        f"Uses VRAM: {runtime.uses_vram}",
        f"Python: {runtime.python_version}",
        f"Platform: {runtime.platform_name}",
        f"Torch available: {runtime.torch_available}",
    ]
    if runtime.torch_version:
        lines.append(f"Torch version: {runtime.torch_version}")
    if runtime.cuda_available:
        lines.append(f"CUDA device count: {runtime.cuda_device_count}")
        lines.append(f"CUDA device name: {runtime.cuda_device_name}")
    if runtime.note:
        lines.append(f"Note: {runtime.note}")
    return "\n".join(lines)
