"""Apply the parent repository's reviewed scenario identity fix to pinned AIOpsLab.

No network access, commits, resets or forced updates. Existing unrelated edits are
preserved; incompatible patch context fails with instructions instead of guessing.
"""
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[2]
    dependency = root / 'AIOpsLab'
    patch = root / 'patches/aiopslab-scenario-identity.patch'
    if not (dependency / 'gen_and_telmetry.py').is_file():
        raise SystemExit('Initialize the pinned AIOpsLab submodule with git submodule update --init --recursive first.')
    helper_source = root / 'dataset_generation/scenario_identity.py'
    helper = dependency / 'scenario_identity.py'
    expected = helper_source.read_bytes()
    if helper.exists() and helper.read_bytes() != expected:
        raise SystemExit('AIOpsLab/scenario_identity.py differs from the reviewed helper; preserve and inspect local changes.')
    def install_helper():
        if not helper.exists():
            helper.write_bytes(expected)
    def run(*args):
        return subprocess.run(['git', '-C', str(dependency), 'apply', *args, str(patch)],
                              text=True, capture_output=True, check=False)
    if run('--reverse', '--check').returncode == 0:
        install_helper()
        print('AIOpsLab scenario identity patch is already applied.')
        return
    checked = run('--check')
    if checked.returncode:
        raise SystemExit('AIOpsLab patch conflicts with the current checkout; preserve local work and inspect it.\n' + checked.stderr)
    applied = run()
    if applied.returncode:
        raise SystemExit(applied.stderr)
    install_helper()
    print('Applied AIOpsLab scenario identity patch; pinned submodule commit remains unchanged.')


if __name__ == '__main__':
    main()
