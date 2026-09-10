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
    def run(*args, target=None):
        return subprocess.run(['git', '-C', str(dependency), 'apply', *args, str(target or patch)],
                              text=True, capture_output=True, check=False)
    generator = (dependency / 'gen_and_telmetry.py').read_text()
    if 'attach_scenario_identity(' in generator and 'unique_scenarios(' in generator:
        # Checkouts newer than the pinned commit (this host runs b56eda8 plus
        # local edits) carry the fix as a manual port; the patch context no
        # longer applies, and the submodule must never be reset to satisfy it.
        install_helper()
        print('AIOpsLab generator already carries the scenario identity fix (ported or patched).')
        return
    if run('--reverse', '--check').returncode == 0:
        install_helper()
        print('AIOpsLab scenario identity patch is already applied.')
        return
    # The pinned-commit patch first, then the port recorded for the training
    # host's newer checkout (b56eda8). Neither resets nor commits the submodule.
    errors = []
    for candidate in (patch, root / 'patches/aiopslab-scenario-identity-b56eda8.patch'):
        if not candidate.is_file():
            continue
        checked = run('--check', target=candidate)
        if checked.returncode:
            errors.append(f'{candidate.name}: {checked.stderr.strip()}')
            continue
        applied = run(target=candidate)
        if applied.returncode:
            raise SystemExit(applied.stderr)
        install_helper()
        print(f'Applied {candidate.name}; pinned submodule commit remains unchanged.')
        return
    raise SystemExit('No bundled AIOpsLab patch applies to this checkout; preserve local work and port '
                     'attach_scenario_identity/unique_scenarios by hand (training_pipeline/OPERATIONS.md).\n'
                     + '\n'.join(errors))


if __name__ == '__main__':
    main()
