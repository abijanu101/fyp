
import argparse
import re
import zipfile
from pathlib import Path, PurePosixPath
from shutil import copyfileobj


def extract_first_frame(archive_path, output_path):
    """Extract the chronologically earliest PNG from a context-camera ZIP."""

    with zipfile.ZipFile(archive_path, "r") as archive:
        frames = [
            name for name in archive.namelist()
            if (
                len(PurePosixPath(name).parts) == 2
                and PurePosixPath(name).parts[0] == "images"
                and name.lower().endswith(".png")
            )
        ]

        if not frames:
            print(f"[SKIP] No frames found: {archive_path}")
            return False

        # Filenames encode timestamps in a lexicographically sortable format.
        first_frame = min(
            frames,
            key=lambda name: PurePosixPath(name).name
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with archive.open(first_frame) as src:
            with open(output_path, "wb") as dst:
                copyfileobj(src, dst)

        print(f"[OK] {archive_path} -> {output_path}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract the first frame from every context camera."
    )

    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Root directory containing group_XX directories."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: dataset_root/first_frames)."
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_root = (
        args.output.resolve()
        if args.output
        else dataset_root / "first_frames"
    )

    if not dataset_root.is_dir():
        parser.error(f"Dataset directory does not exist: {dataset_root}")

    extracted = 0
    skipped = 0

    for archive_path in sorted(dataset_root.glob(
        "group_*/experiment_*/context_cam_*.zip"
    )):
        group_match = re.fullmatch(
            r"group_(\d+)", archive_path.parents[1].name
        )
        experiment_match = re.fullmatch(
            r"experiment_(\d+)", archive_path.parent.name
        )
        camera_match = re.fullmatch(
            r"context_cam_(\d+)", archive_path.stem
        )

        if not all((group_match, experiment_match, camera_match)):
            continue

        group_id = group_match.group(1)
        experiment_id = experiment_match.group(1)
        camera_id = camera_match.group(1)

        # Group subdirectories prevent filename collisions.
        filename = f"{experiment_id}_{camera_id}.png"

        output_path = (
            output_root / f"group_{group_id}" / filename
        )

        print(group_match, experiment_match, camera_match)
        if extract_first_frame(archive_path, output_path):
            extracted += 1
        else:
            skipped += 1

    print("\nExtraction complete.")
    print(f"Extracted: {extracted}")
    print(f"Skipped:   {skipped}")
    print(f"Output:    {output_root}")


if __name__ == "__main__":
    main()
