#!/usr/bin/env python3
"""Update and build llama.cpp variants.

This script implements the workflows described in AGENTS.md.
"""

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

# -----------------------------
# Config (paths/URLs at top)
# -----------------------------
GITHUB_API_RELEASE_LATEST = (
    "https://api.github.com/repos/{owner}/{repo}/releases/latest"
)
GITHUB_API_COMMIT = "https://api.github.com/repos/{owner}/{repo}/commits/{ref}"
GITHUB_API_COMMITS = "https://api.github.com/repos/{owner}/{repo}/commits"
GITHUB_API_TAGS = "https://api.github.com/repos/{owner}/{repo}/tags"

PREBUILT_VULKAN_REPO = ("ggml-org", "llama.cpp")
PREBUILT_ROCM_REPO = ("lemonade-sdk", "llamacpp-rocm")

PREBUILT_VULKAN_ASSET_MATCHES = [
    ["ubuntu", "vulkan", "x64"],
    ["linux", "vulkan", "x64"],
    ["ubuntu", "vulkan"],
    ["linux", "vulkan"],
]
PREBUILT_ROCM_ASSET_MATCHES = [
    ["ubuntu", "rocm", "gfx1151", "x64"],
    ["linux", "rocm", "gfx1151", "x64"],
    ["ubuntu", "rocm", "gfx1151"],
    ["linux", "rocm", "gfx1151"],
]

INSTALL_BASE = os.path.expanduser("~/Apps")
LLAMA_CPP_SRC = "/home/sid/Projects/ai/llama.cpp"

TOOLBOXES_REPO = "/home/sid/Projects/ai/amd-strix-halo-toolboxes"
TOOLBOXES_DIR = "/mnt/data/projects/ai/amd-strix-halo-toolboxes/toolboxes"
TOOLBOX_DOCKERFILE_VULKAN = os.path.join(TOOLBOXES_DIR, "Dockerfile.vulkan-radv")
TOOLBOX_DOCKERFILE_ROCM = os.path.join(TOOLBOXES_DIR, "Dockerfile.rocm7-nightlies")
TOOLBOX_DOCKERFILE_VULKAN_URL = (
    "https://raw.githubusercontent.com/kyuz0/amd-strix-halo-toolboxes/"
    "refs/heads/main/toolboxes/Dockerfile.vulkan-radv"
)
TOOLBOX_DOCKERFILE_ROCM_URL = (
    "https://raw.githubusercontent.com/kyuz0/amd-strix-halo-toolboxes/"
    "refs/heads/main/toolboxes/Dockerfile.rocm7-nightlies"
)
TOOLBOX_BUILD_CONTEXT = TOOLBOXES_DIR

CONTAINER_RUNTIME = os.environ.get("LLAMA_CPP_CONTAINER_RUNTIME", "").strip()

LLAMA_CPP_UPSTREAM_REPO = ("ggml-org", "llama.cpp")
LLAMA_CPP_DEFAULT_BRANCH = "master"

# Suffixes (adjust here if naming changes)
SUFFIX_PREBUILT_VULKAN = "vulkan-prebuilt"
SUFFIX_PREBUILT_ROCM = "rocm-prebuilt"
SUFFIX_BUILD_VULKAN = "vulkan"
SUFFIX_TOOLBOX_VULKAN = "vulkan"
SUFFIX_TOOLBOX_ROCM = "rocm-7nightlies"

# -----------------------------
# Helpers
# -----------------------------


def eprint(*args):
    print(*args, file=sys.stderr)


def run(cmd, cwd=None, check=True, capture=False):
    if capture:
        result = subprocess.run(
            cmd, cwd=cwd, check=check, text=True, stdout=subprocess.PIPE
        )
        return result.stdout.strip()
    subprocess.run(cmd, cwd=cwd, check=check)
    return ""


def ensure_empty_dir(path: Path, force: bool):
    if path.exists():
        if not force:
            raise RuntimeError(f"Install dir already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def rmtree_writeable(path: Path):
    if not path.exists():
        return

    def onerror(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR)
        except OSError:
            pass
        func(p)

    shutil.rmtree(path, onerror=onerror)


def clean_llama_cpp_tree(src: Path):
    try:
        run(["git", "clean", "-xdf"], cwd=src)
    except subprocess.CalledProcessError as exc:
        eprint(f"git clean failed: {exc}. Trying to remove build dir...")
        rmtree_writeable(src / "build")
        run(["git", "clean", "-xdf"], cwd=src)


def find_executable(names):
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def resolve_build_tools():
    c_compiler = find_executable(["cc", "gcc", "clang"])
    cxx_compiler = find_executable(["c++", "g++", "clang++"])
    if not c_compiler or not cxx_compiler:
        missing = []
        if not c_compiler:
            missing.append("C compiler (cc/gcc/clang)")
        if not cxx_compiler:
            missing.append("C++ compiler (c++/g++/clang++)")
        raise RuntimeError(
            "Missing build tools: "
            + ", ".join(missing)
            + ". Install a compiler toolchain and retry."
        )

    if shutil.which("ninja"):
        generator = ["-G", "Ninja"]
    elif shutil.which("make"):
        generator = ["-G", "Unix Makefiles"]
    else:
        raise RuntimeError("Missing build tool: install ninja or make and retry.")

    return generator, c_compiler, cxx_compiler


def require_vulkan_deps():
    missing = []
    if not shutil.which("glslc"):
        missing.append("glslc (Vulkan SDK shader compiler)")

    header_paths = [
        Path("/usr/include/vulkan/vulkan.h"),
        Path("/usr/local/include/vulkan/vulkan.h"),
    ]
    if not any(p.exists() for p in header_paths):
        missing.append("Vulkan headers (vulkan.h)")

    lib_paths = [
        Path("/usr/lib/libvulkan.so"),
        Path("/usr/lib64/libvulkan.so"),
        Path("/usr/local/lib/libvulkan.so"),
        Path("/usr/local/lib64/libvulkan.so"),
    ]
    if not any(p.exists() for p in lib_paths):
        missing.append("Vulkan loader library (libvulkan.so)")

    if missing:
        raise RuntimeError(
            "Missing Vulkan dependencies: "
            + ", ".join(missing)
            + ". Install Vulkan SDK/dev packages and retry."
        )


def require_container_runtime():
    runtime = container_cmd()
    if not shutil.which(runtime):
        raise RuntimeError(
            f"Container runtime not found: {runtime}. Install it or set LLAMA_CPP_CONTAINER_RUNTIME."
        )


def is_toolbox_env() -> bool:
    return Path("/run/.toolboxenv").exists()


def ensure_toolbox_or_run():
    if is_toolbox_env():
        return
    if not shutil.which("toolbox"):
        raise RuntimeError(
            "Toolbox not found in PATH. Install it or run inside a toolbox container."
        )
    script_path = Path(__file__).resolve()
    cmd = ["toolbox", "run", "--", sys.executable, str(script_path), *sys.argv[1:]]
    run(cmd)
    raise SystemExit(0)


def http_get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "update-llama-cpp"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP error {exc.code} fetching {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Network error fetching {url}: {exc.reason}. "
            "Check DNS/network connectivity or proxy settings."
        ) from exc


def http_get_json_with_headers(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "update-llama-cpp"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8")), resp.headers
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP error {exc.code} fetching {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Network error fetching {url}: {exc.reason}. "
            "Check DNS/network connectivity or proxy settings."
        ) from exc


def github_latest_release(owner: str, repo: str):
    url = GITHUB_API_RELEASE_LATEST.format(owner=owner, repo=repo)
    return http_get_json(url)


def github_commit(owner: str, repo: str, ref: str):
    url = GITHUB_API_COMMIT.format(owner=owner, repo=repo, ref=ref)
    return http_get_json(url)


ARCHIVE_EXTENSIONS = (".tar.gz", ".tgz", ".tar.xz", ".tar.bz2", ".zip")
DEFAULT_EXCLUDE_SUBSTRINGS = [".sha256", "checksum", "checksums", "sig", ".asc"]


def is_archive_name(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(ext) for ext in ARCHIVE_EXTENSIONS)


def pick_asset(assets, match_sets, exclude_substrings=None):
    excludes = [s.lower() for s in (exclude_substrings or DEFAULT_EXCLUDE_SUBSTRINGS)]
    for required in match_sets:
        required_lower = [s.lower() for s in required]
        for asset in assets:
            name = asset.get("name", "")
            lower = name.lower()
            if not is_archive_name(lower):
                continue
            if any(s in lower for s in excludes):
                continue
            if all(s in lower for s in required_lower):
                return asset
    available = [a.get("name", "") for a in assets]
    raise RuntimeError(
        f"No asset matching any of: {match_sets}. Available assets: {available}"
    )


def download_file(url: str, dest: Path):
    req = urllib.request.Request(url, headers={"User-Agent": "update-llama-cpp"})
    try:
        with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP error {exc.code} downloading {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Network error downloading {url}: {exc.reason}. "
            "Check DNS/network connectivity or proxy settings."
        ) from exc


def extract_archive(archive_path: Path, dest_dir: Path):
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
        return
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
        return
    raise RuntimeError(f"Unsupported archive type: {archive_path}")


def chmod_bins(install_dir: Path):
    bin_dir = install_dir / "bin"
    if bin_dir.is_dir():
        for p in bin_dir.iterdir():
            if p.is_file():
                p.chmod(p.stat().st_mode | 0o111)
        return
    for p in install_dir.rglob("*"):
        if p.is_file():
            p.chmod(p.stat().st_mode | 0o111)


def flatten_single_top_dir(install_dir: Path):
    entries = [p for p in install_dir.iterdir()]
    if len(entries) != 1:
        return
    top = entries[0]
    if not top.is_dir():
        return
    for child in top.iterdir():
        shutil.move(str(child), install_dir)
    top.rmdir()


def copy_build_bin_to_install(build_dir: Path, install_dir: Path):
    bin_dir = build_dir / "bin"
    if not bin_dir.is_dir():
        raise RuntimeError(f"Build output not found: {bin_dir}")
    for p in bin_dir.iterdir():
        if p.is_file():
            shutil.copy2(p, install_dir / p.name)


def short_sha(full_sha: str) -> str:
    return full_sha[:7]


def parse_link_last_page(link_header: str) -> int:
    if not link_header:
        return 0
    for part in link_header.split(","):
        if 'rel="last"' in part:
            match = re.search(r"[?&]page=(\d+)", part)
            if match:
                return int(match.group(1))
    return 0


def get_remote_llama_cpp_build_ver() -> str:
    url = (
        GITHUB_API_COMMITS.format(
            owner=LLAMA_CPP_UPSTREAM_REPO[0],
            repo=LLAMA_CPP_UPSTREAM_REPO[1],
        )
        + f"?sha={LLAMA_CPP_DEFAULT_BRANCH}&per_page=1"
    )
    data, headers = http_get_json_with_headers(url)
    last_page = parse_link_last_page(headers.get("Link", ""))
    if last_page:
        return str(last_page)
    return str(len(data))


def find_tag_for_commit(owner: str, repo: str, commit_sha: str) -> str | None:
    page = 1
    last_page = None
    while True:
        url = (
            GITHUB_API_TAGS.format(owner=owner, repo=repo)
            + f"?per_page=100&page={page}"
        )
        data, headers = http_get_json_with_headers(url)
        for tag in data:
            tag_sha = tag.get("commit", {}).get("sha")
            if not tag_sha:
                continue
            if commit_sha.startswith(tag_sha) or tag_sha.startswith(commit_sha):
                return tag.get("name")
        if last_page is None:
            last_page = parse_link_last_page(headers.get("Link", ""))
        if not data or (last_page and page >= last_page):
            break
        page += 1
    return None


def resolve_commit_sha(owner: str, repo: str, commit_ref: str) -> str | None:
    if len(commit_ref) >= 7:
        try:
            commit = github_commit(owner, repo, commit_ref)
            sha = commit.get("sha")
            if sha:
                return sha
        except RuntimeError:
            pass

    page = 1
    last_page = None
    while True:
        url = (
            GITHUB_API_COMMITS.format(owner=owner, repo=repo)
            + f"?per_page=100&page={page}"
        )
        data, headers = http_get_json_with_headers(url)
        for item in data:
            sha = item.get("sha")
            if sha and sha.startswith(commit_ref):
                return sha
        if last_page is None:
            last_page = parse_link_last_page(headers.get("Link", ""))
        if not data or (last_page and page >= last_page):
            break
        page += 1
    return None


def resolve_rocm_release_version(release: dict) -> str:
    body = release.get("body", "") or ""
    match = re.search(
        r"Llama\.cpp Commit Hash[^0-9a-fA-F]*([0-9a-fA-F]{5,40})",
        body,
    )
    if not match:
        raise RuntimeError("ROCm release notes missing Llama.cpp Commit Hash")
    commit_ref = match.group(1)
    commit_sha = resolve_commit_sha(
        LLAMA_CPP_UPSTREAM_REPO[0], LLAMA_CPP_UPSTREAM_REPO[1], commit_ref
    )
    if not commit_sha:
        raise RuntimeError("Unable to resolve full commit sha for ROCm release")
    tag = find_tag_for_commit(
        LLAMA_CPP_UPSTREAM_REPO[0], LLAMA_CPP_UPSTREAM_REPO[1], commit_sha
    )
    if tag:
        return tag
    eprint(
        "Warning: Could not resolve tag for commit "
        f"{commit_ref}; using commit hash for version."
    )
    return commit_ref


def container_cmd() -> str:
    if CONTAINER_RUNTIME:
        return CONTAINER_RUNTIME
    return "podman" if shutil.which("podman") else "docker"


def docker_build(dockerfile: Path, tag: str, context: Path):
    runtime = container_cmd()
    run([runtime, "build", "-f", str(dockerfile), "-t", tag, str(context)])


def resolve_dockerfile(path: str, url: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    if not Path(TOOLBOX_BUILD_CONTEXT).exists():
        raise RuntimeError(
            "Toolbox build context not found; cannot use remote Dockerfile"
        )
    tmp_dir = Path(tempfile.mkdtemp(prefix="llama-dockerfile-"))
    tmp_file = tmp_dir / p.name
    download_file(url, tmp_file)
    return tmp_file


def prebuilt_install(
    owner: str,
    repo: str,
    substrings,
    suffix: str,
    force: bool,
    version_tag: str | None = None,
    release: dict | None = None,
):
    if release is None:
        release = github_latest_release(owner, repo)
    assert release is not None
    tag = version_tag or release.get("tag_name")
    if not tag:
        raise RuntimeError("Release tag not found")
    asset = pick_asset(release.get("assets", []), substrings)
    url = asset.get("browser_download_url")
    if not url:
        raise RuntimeError("Asset download URL not found")

    install_dir = Path(INSTALL_BASE) / f"llama.cpp-{tag}-{suffix}"
    ensure_empty_dir(install_dir, force)

    with tempfile.TemporaryDirectory(prefix="llama-prebuilt-") as td:
        archive_path = Path(td) / asset.get("name", "download")
        download_file(url, archive_path)
        extract_archive(archive_path, install_dir)
        flatten_single_top_dir(install_dir)

    print(f"Installed to: {install_dir}")
    return install_dir


def cmd_vulkan_prebuilt(args):
    prebuilt_install(
        PREBUILT_VULKAN_REPO[0],
        PREBUILT_VULKAN_REPO[1],
        PREBUILT_VULKAN_ASSET_MATCHES,
        SUFFIX_PREBUILT_VULKAN,
        args.force,
    )


def cmd_rocm_prebuilt(args):
    release = github_latest_release(PREBUILT_ROCM_REPO[0], PREBUILT_ROCM_REPO[1])
    version_tag = resolve_rocm_release_version(release)
    install_dir = prebuilt_install(
        PREBUILT_ROCM_REPO[0],
        PREBUILT_ROCM_REPO[1],
        PREBUILT_ROCM_ASSET_MATCHES,
        SUFFIX_PREBUILT_ROCM,
        args.force,
        version_tag=version_tag,
        release=release,
    )
    chmod_bins(install_dir)


def cmd_vulkan_build(args):
    src = Path(LLAMA_CPP_SRC)
    if not src.exists():
        raise RuntimeError(f"llama.cpp source not found: {src}")

    ensure_toolbox_or_run()
    generator, c_compiler, cxx_compiler = resolve_build_tools()
    require_vulkan_deps()

    run(["git", "pull"], cwd=src)
    clean_llama_cpp_tree(src)
    build_ver = run(["git", "rev-list", "--count", "HEAD"], cwd=src, capture=True)

    install_dir = Path(INSTALL_BASE) / f"llama.cpp-b{build_ver}-{SUFFIX_BUILD_VULKAN}"
    ensure_empty_dir(install_dir, args.force)

    run(
        [
            "cmake",
            "-S",
            ".",
            "-B",
            "build",
            *generator,
            "-DGGML_VULKAN=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_C_COMPILER={c_compiler}",
            f"-DCMAKE_CXX_COMPILER={cxx_compiler}",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=ON",
            "-DLLAMA_BUILD_SERVER=ON",
        ],
        cwd=src,
    )
    run(["cmake", "--build", "build", "--config", "Release"], cwd=src)
    copy_build_bin_to_install(src / "build", install_dir)
    print(f"Installed to: {install_dir}")


def cmd_vulkan_toolbox(args):
    require_container_runtime()
    dockerfile = resolve_dockerfile(
        TOOLBOX_DOCKERFILE_VULKAN, TOOLBOX_DOCKERFILE_VULKAN_URL
    )
    build_ver = get_remote_llama_cpp_build_ver()
    tag = f"llama.cpp-{build_ver}-{SUFFIX_TOOLBOX_VULKAN}"
    docker_build(dockerfile, tag, Path(TOOLBOX_BUILD_CONTEXT))
    print(f"Built image: {tag}")


def cmd_rocm_toolbox(args):
    require_container_runtime()
    dockerfile = resolve_dockerfile(
        TOOLBOX_DOCKERFILE_ROCM, TOOLBOX_DOCKERFILE_ROCM_URL
    )
    build_ver = get_remote_llama_cpp_build_ver()
    tag = f"llama.cpp-{build_ver}-{SUFFIX_TOOLBOX_ROCM}"
    docker_build(dockerfile, tag, Path(TOOLBOX_BUILD_CONTEXT))
    print(f"Built image: {tag}")


def cmd_update_toolboxes(args):
    repo = Path(TOOLBOXES_REPO)
    if not repo.exists():
        raise RuntimeError(f"Toolboxes repo not found: {repo}")
    refresh_script = repo / "refresh-toolboxes.sh"
    if not refresh_script.exists():
        raise RuntimeError(f"refresh-toolboxes not found: {refresh_script}")
    run(["git", "pull"], cwd=repo)
    run([str(refresh_script), "all"], cwd=repo)


def cmd_all(args):
    steps = [
        ("vulkan-prebuilt", cmd_vulkan_prebuilt),
        ("rocm-prebuilt", cmd_rocm_prebuilt),
        ("vulkan-build", cmd_vulkan_build),
        ("vulkan-toolbox", cmd_vulkan_toolbox),
        ("rocm-toolbox", cmd_rocm_toolbox),
        ("update-toolboxes", cmd_update_toolboxes),
    ]
    failures = []
    for name, func in steps:
        try:
            func(args)
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            eprint(f"Step failed ({name}): {exc}")
    if failures:
        raise RuntimeError("Some steps failed:\n- " + "\n- ".join(failures))


def build_parser():
    p = argparse.ArgumentParser(description="Update/build llama.cpp variants")
    p.add_argument(
        "--force", action="store_true", help="overwrite existing install dir"
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("vulkan-prebuilt", help="Install latest prebuilt Vulkan release")
    sub.add_parser("rocm-prebuilt", help="Install latest prebuilt ROCm release")
    sub.add_parser("vulkan-build", help="Build Vulkan from local llama.cpp source")
    sub.add_parser("vulkan-toolbox", help="Build Vulkan via toolbox Dockerfile")
    sub.add_parser("rocm-toolbox", help="Build ROCm via toolbox Dockerfile")
    sub.add_parser("update-toolboxes", help="Update toolbox repo and refresh toolboxes")
    sub.add_parser("all", help="Run all steps (prebuilt, build, toolbox, update)")

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.cmd == "vulkan-prebuilt":
        cmd_vulkan_prebuilt(args)
    elif args.cmd == "rocm-prebuilt":
        cmd_rocm_prebuilt(args)
    elif args.cmd == "vulkan-build":
        cmd_vulkan_build(args)
    elif args.cmd == "vulkan-toolbox":
        cmd_vulkan_toolbox(args)
    elif args.cmd == "rocm-toolbox":
        cmd_rocm_toolbox(args)
    elif args.cmd == "update-toolboxes":
        cmd_update_toolboxes(args)
    elif args.cmd == "all":
        cmd_all(args)
    else:
        raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
