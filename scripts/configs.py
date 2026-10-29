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


def spike_args(config):
    import shlex
    return shlex.split(subprocess.check_output(
        ['make', '-s', '--no-print-directory', f'CONFIG={config_id(config)}', 'config-spike-args'],
        cwd=ROOT, text=True).strip())


def execution_metadata(config):
    import hashlib
    import shlex
    args = spike_args(config)
    return dict(spike_args=shlex.join(args),
                extension_source_sha256=hashlib.sha256(
                    (ROOT / 'src/spike_ext/xqmac.cc').read_bytes()).hexdigest()
                if '--extension=xqmac' in args else '',
                repacked_model_sha256=hashlib.sha256(
                    (ROOT / 'build/model/efficientnet-lite0-int8-aligned.tflite').read_bytes()).hexdigest())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Print one config Spike argument per line')
    parser.add_argument('--spike-args', type=config_id, required=True)
    print('\n'.join(spike_args(parser.parse_args().spike_args)))
