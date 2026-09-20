from __future__ import annotations

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent

# ============================================================
# FLYISAAC V3.7.1 — SAFE AUTO CLEANUP
# ============================================================
#
# This script DELETES only exact, known legacy diagnostic/test files.
#
# It NEVER deletes:
#   isaac/
#   isaac_v3/
#   brain/
#   learning/
#   datasets/
#   checkpoints/
#   isaac_mods/
#   current V3.7 / V3.7.1 runtime files
#   dopamine_settings.py
#
# No wildcard deletion is used for Python source files.
# ============================================================

SAFE_DELETE_ROOT_FILES = (
    # ---------------- V3.1 / V3.2 ----------------
    "verify_isaac_v31.py",
    "verify_isaac_v32.py",
    "test_temporal_v32.py",

    # ---------------- V3.3 ----------------
    "diagnose_isaac_v33.py",
    "verify_isaac_v33.py",
    "verify_dagger_hotfix.py",
    "verify_focus_pause_v332.py",
    "test_visual_v33.py",
    "test_multimodal_v33.py",

    # ---------------- V3.4 / V3.4.1 ----------------
    "verify_isaac_v34.py",
    "verify_anatomical_monitor_v341.py",
    "test_visual_v34.py",
    "test_multimodal_v34.py",
    "test_neuron_monitor_v34.py",
    "test_anatomical_monitor_v341.py",
    "test_sensory_v34.py",
    "test_decoder_v3.py",

    # ---------------- V3.5 / V3.5.1 ----------------
    "verify_isaac_v35.py",
    "test_grid_v35.py",
    "test_multimodal_v35.py",
    "test_decoder_v35.py",
    "verify_transfer_v351.py",
    "test_transfer_v351.py",
    "audit_legacy_files_v351.py",

    # ---------------- V3.6 / V3.6.x ----------------
    "verify_isaac_v36.py",
    "test_dopamine_v36.py",
    "test_brain_grid_v36.py",
    "verify_bigger_monitor_v361.py",
    "test_bigger_monitor_v361.py",
    "verify_dopamine_settings_v362.py",
    "test_dopamine_settings_v362.py",
    "verify_room_stuck_v363.py",
    "test_room_stuck_v363.py",
)

# These are the current V3.7 / V3.7.1 files that MUST remain.
CURRENT_KEEP_ROOT_FILES = (
    "play_isaac_v3.py",
    "collect_isaac_v3.py",
    "dagger_isaac_v3.py",
    "train_decoder_v3.py",
    "verify_isaac_v37.py",
    "test_tactical_grid_v37.py",
    "test_dopamine_preserved_v37.py",
    "test_anatomy_v37.py",
    "test_sensory_v37.py",
    "test_fast_loop_v37.py",
    "verify_door_memory_v371.py",
    "test_door_memory_v371.py",
    "cleanup_legacy_v371.py",
)

PROTECTED_DIRS = (
    "isaac",
    "isaac_v3",
    "brain",
    "learning",
    "datasets",
    "checkpoints",
    "isaac_mods",
)

# Cache folders contain generated Python bytecode only.
CACHE_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
}


def delete_file(path: Path) -> bool:
    try:
        if path.is_file() or path.is_symlink():
            path.unlink()
            print(f"  [DELETED FILE] {path.relative_to(ROOT)}")
            return True
    except Exception as exc:
        print(f"  [FAILED] {path.name}: {exc}")
    return False


def delete_caches() -> tuple[int, int]:
    deleted_dirs = 0
    failed = 0

    # Do not recurse into persistent data/model directories.
    protected_paths = {(ROOT / name).resolve() for name in PROTECTED_DIRS}

    for path in sorted(ROOT.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if not path.is_dir() or path.name not in CACHE_DIR_NAMES:
            continue

        try:
            resolved = path.resolve()
            inside_protected = any(
                protected == resolved or protected in resolved.parents
                for protected in protected_paths
            )
            # __pycache__ inside source packages is safe too; it is generated
            # bytecode. datasets/checkpoints are skipped simply to be extra safe.
            if inside_protected and any(
                part in {"datasets", "checkpoints"}
                for part in path.parts
            ):
                continue

            shutil.rmtree(path)
            print(f"  [DELETED CACHE] {path.relative_to(ROOT)}")
            deleted_dirs += 1
        except Exception as exc:
            print(f"  [FAILED CACHE] {path}: {exc}")
            failed += 1

    return deleted_dirs, failed


def main() -> None:
    print("=" * 62)
    print(" FLYISAAC V3.7.1 — AUTO LEGACY CLEANUP")
    print("=" * 62)
    print(f"Root: {ROOT}")
    print()
    print("Deleting ONLY known obsolete diagnostics/tests + Python caches.")
    print("Datasets, checkpoints, runtime modules and dopamine settings are protected.")
    print()

    deleted_files = 0
    missing = 0

    for name in SAFE_DELETE_ROOT_FILES:
        path = ROOT / name
        if path.exists():
            deleted_files += int(delete_file(path))
        else:
            missing += 1

    deleted_cache_dirs, cache_failures = delete_caches()

    print()
    print("-" * 62)
    print("RESULT")
    print("-" * 62)
    print(f"Legacy files deleted : {deleted_files}")
    print(f"Already absent       : {missing}")
    print(f"Cache dirs deleted   : {deleted_cache_dirs}")
    print(f"Cache failures       : {cache_failures}")
    print()

    print("PROTECTED / LEFT UNTOUCHED:")
    for name in PROTECTED_DIRS:
        p = ROOT / name
        print(f"  [KEEP] {name}/" + ("  present" if p.exists() else "  not present here"))

    for name in CURRENT_KEEP_ROOT_FILES:
        p = ROOT / name
        if p.exists():
            print(f"  [KEEP] {name}")

    dopamine = ROOT / "isaac_v3" / "dopamine_settings.py"
    if dopamine.exists():
        print("  [KEEP] isaac_v3/dopamine_settings.py")

    print()
    print("Cleanup finished.")
    print("You can run this script again safely; already-deleted files are simply skipped.")


if __name__ == "__main__":
    main()
