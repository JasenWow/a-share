import datetime
import tarfile
import hashlib
import os
import requests
import sys
from pathlib import Path

# Logging: prefer loguru if available, otherwise fallback to standard logging
try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("updater")

# Project layout assumptions
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # big-a/
DEFAULT_DATA_DIR = PROJECT_ROOT / 'data' / 'qlib_data' / 'cn_data'


def _data_dir(data_dir: str | None) -> Path:
    return Path(data_dir) if data_dir else DEFAULT_DATA_DIR


def get_last_update_date(data_dir: str | None = None) -> str:
    """Return the last update date (YYYY-MM-DD) from calendars/day.txt.
    Data directory is expected to be big-a/data/qlib_data/cn_data by default."""
    dirp = _data_dir(data_dir)
    day_file = dirp / 'calendars' / 'day.txt'
    if not day_file.exists():
        raise FileNotFoundError(f"Calendar file not found: {day_file}")
    lines = []
    with day_file.open('r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
    if not lines:
        raise ValueError("Calendar day.txt is empty")
    last_line = lines[-1]
    date_token = last_line.split()[0]
    # Basic ISO date validation
    try:
        datetime.datetime.strptime(date_token, "%Y-%m-%d")
    except Exception:
        # If not exactly ISO, return the token as-is to allow downstream handling
        pass
    return date_token


def _latest_release_tarball_url() -> str | None:
    api = "https://api.github.com/repos/chenditc/investment_data/releases/latest"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "big-a-quant-updater",
    }
    try:
        resp = requests.get(api, headers=headers, timeout=60)
    except Exception as e:
        logger.error("Failed to fetch latest release: %s", e)
        return None
    if resp.status_code != 200:
        logger.warning("Failed to fetch latest release: HTTP %s", resp.status_code)
        return None
    data = resp.json()
    assets = data.get("assets", [])
    # Prefer tarballs named qlib_bin.tar.gz if available
    for asset in assets:
        name = asset.get("name", "")
        if name.endswith(".tar.gz") or name.endswith(".tgz"):
            url = asset.get("browser_download_url")
            if url:
                return url
    # Fallback: any tarball-like asset
    for asset in assets:
        url = asset.get("browser_download_url")
        if url and (url.endswith(".tar.gz") or url.endswith(".tgz")):
            return url
    return None


def _download_file(url: str, dest: Path) -> None:
    logger.info("Downloading data tarball from %s", url)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open('wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def _extract_tarball(tar_path: Path, extract_to: Path) -> None:
    logger.info("Extracting tarball %s to %s", tar_path, extract_to)
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=str(extract_to))


def _checksum_files(dirp: Path) -> str:
    # Lightweight integrity check: hash of file names and sizes
    import hashlib

    sha = hashlib.sha256()
    if not dirp.exists():
        return sha.hexdigest()
    for root, _, files in os.walk(dirp):
        for fname in sorted(files):
            path = Path(root) / fname
            try:
                sha.update(fname.encode('utf-8'))
                sha.update(str(path.stat().st_size).encode('utf-8'))
            except Exception:
                continue
    return sha.hexdigest()


def update_incremental(data_dir: str | None = None, start_date: str | None = None) -> None:
    """Incrementally update data using pre-built qlib_bin tarball from chenditc releases.

    This function downloads the latest release tarball and extracts it into the
    configured data directory. It performs a light integrity check after extraction.
    """
    dirp = _data_dir(data_dir)
    if not dirp.exists():
        logger.info("Creating data directory: %s", dirp)
        dirp.mkdir(parents=True, exist_ok=True)

    current = start_date or get_last_update_date(str(dirp))
    logger.info("Starting incremental update. Current data date: %s", current)
    tar_url = _latest_release_tarball_url()
    if not tar_url:
        raise RuntimeError("Could not determine latest qlib_bin tarball URL from chenditc releases")
    tar_path = dirp / "qlib_bin_latest.tar.gz"
    _download_file(tar_url, tar_path)
    _extract_tarball(tar_path, dirp)
    tar_path.unlink(missing_ok=True)
    checksum = _checksum_files(dirp)
    logger.info("Post-update checksum: %s", checksum)
    return


def verify_update(data_dir: str | None = None) -> bool:
    dirp = _data_dir(data_dir)
    calendars = dirp / 'calendars' / 'day.txt'
    if not calendars.exists():
        logger.error("Verification failed: calendars/day.txt missing")
        return False
    try:
        with calendars.open('r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            logger.error("Verification failed: calendars/day.txt empty")
            return False
        last = lines[-1].split()[0]
        datetime.datetime.strptime(last, "%Y-%m-%d")
    except Exception as e:
        logger.error("Verification failed: invalid last date in calendars/day.txt: %s", e)
        return False
    # Basic instrument list sanity check
    instruments_dir = dirp / 'instruments'
    if not instruments_dir.exists():
        logger.warning("Verification warning: instruments directory missing")
    else:
        found = False
        for name in ('csiall.txt', 'all.txt', 'csi300.txt'):
            if (instruments_dir / name).exists():
                found = True
                break
        if not found:
            logger.warning("Verification warning: no instrument list file found in instruments/")
    # Basic feature presence check for a sample stock
    sample_dir = dirp / 'features' / 'SH600000'
    if sample_dir.exists():
        has_bin = any((sample_dir / f).suffix == '.bin' for f in os.listdir(sample_dir))
        if not has_bin:
            logger.warning("Verification warning: no feature bins found for SH600000")
    logger.info("Verification completed (basic checks performed).")
    return True


__all__ = [
    'get_last_update_date', 'update_incremental', 'verify_update'
]
