"""Configuration IDs and descriptions shared by measurement entry points."""
import re
import subprocess

from get_model import ROOT


def git_state():
    """Return source revision and whether anything outside generated results changed."""
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=all', '--', '.',
             ':(exclude)results'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL))
    except (OSError, subprocess.CalledProcessError):
        return 'unknown', 'unknown'
    return commit, dirty


def config_id(value):
    if not re.fullmatch(r'[a-z0-9]+', value):
        raise ValueError('config must match [a-z0-9]+')
    if not (ROOT / f'experiments/{value}.mk').is_file():
        raise ValueError(f'Unknown config: {value}')
    return value


def config_order():
    ids = [config_id(line.strip()) for line in
           (ROOT / 'experiments/order.txt').read_text().splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Config order must be nonempty and contain no duplicates')
    return ids


def config_desc(config):
    return subprocess.check_output(
        ['make', '-s', '--no-print-directory', f'CONFIG={config_id(config)}', 'config-desc'],
        cwd=ROOT, text=True).strip()
