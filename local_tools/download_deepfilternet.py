import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen


RELEASE_API = "https://api.github.com/repos/Rikorose/DeepFilterNet/releases/tags/v0.5.6"
WINDOWS_ASSET = "deep-filter-0.5.6-x86_64-pc-windows-msvc.exe"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download the DeepFilterNet Windows CLI binary."
    )
    parser.add_argument(
        "--output-path",
        default="local_tools/deepfilternet/deep-filter.exe",
    )
    args = parser.parse_args(argv)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    asset = _find_windows_asset()
    _download(asset["browser_download_url"], output_path)
    print(f"downloaded {asset['name']} -> {output_path}")
    return 0


def _find_windows_asset() -> dict:
    request = Request(RELEASE_API, headers={"User-Agent": "audio-dataset-agent"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    for asset in payload["assets"]:
        if asset["name"] == WINDOWS_ASSET:
            return asset
    raise RuntimeError(f"DeepFilterNet release asset not found: {WINDOWS_ASSET}")


def _download(url: str, output_path: Path) -> None:
    request = Request(url, headers={"User-Agent": "audio-dataset-agent"})
    with urlopen(request, timeout=120) as response:
        output_path.write_bytes(response.read())


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
