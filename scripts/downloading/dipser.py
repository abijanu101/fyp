import os
import re
import json
import ftplib
import zipfile

from dotenv import load_dotenv
from pathlib import Path, PurePosixPath


# ============================================================
# Configuration
# ============================================================

load_dotenv()
HOST = "ftp.scidb.cn"
PORT = 2121
USER = os.getenv('FTP_USER')
PASSWORD = os.getenv('FTP_PASSWORD')

REMOTE_ROOT = "/DIPSEER"

LOCAL_ROOT = Path(r"E:\Coding\Projects\fyp\data\DIPSER")
DATASET_ROOT = LOCAL_ROOT / "DIPSEER"
MANIFEST = LOCAL_ROOT / "dataset.json"
IGNORE_GROUPS = [
    "group_01",
    "group_02"
]
IGNORE_EXPERIMENTS = [
    "experiment_01",
    "experiment_02",
    "experiment_03",
    "experiment_04",
    "experiment_05",
    "experiment_06",
    "experiment_07",
]
IGNORE_CONTEXT_CAMS = [
    # "context_cam_04",
]

# ============================================================
# Directory and filename patterns
# ============================================================

GROUP_RE = re.compile(r"group_[0-9]{2}")
EXP_RE = re.compile(r"experiment_[0-9]{2}")
CONTEXT_RE = re.compile(r"context_cam_[0-9]{2}\.zip")
SUBJECT_RE = re.compile(r"subject_[0-9]{2}\.zip")

TIMESTAMP_RE = re.compile(r"^([0-9]{2})_([0-9]{2})_([0-9]{2})_([0-9]+)$")

EMOTION_KEYS = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
)


# ============================================================
# FTP utilities
# ============================================================

def listing(ftp):
    """Return directories and regular files from the current FTP directory."""

    lines = []
    ftp.retrlines("LIST", lines.append)

    entries = []

    for line in lines:
        parts = line.split(maxsplit=8)

        if len(parts) < 9:
            continue

        permissions = parts[0]
        name = parts[8]

        # Ignore symbolic links and unsupported entry types.
        if permissions[0] in ("d", "-"):
            entries.append(name)

    return entries


def remote_metadata(ftp, filename):
    """Retrieve remote file size and modification timestamp."""

    size = None
    modified = None

    try:
        ftp.voidcmd("TYPE I")
        size = ftp.size(filename)
    except (ftplib.error_perm, ftplib.error_reply, ftplib.error_proto):
        pass

    try:
        response = ftp.sendcmd(f"MDTM {filename}")
        timestamp = response[4:].split(".")[0]

        if re.fullmatch(r"[0-9]{14}", timestamp):
            modified = timestamp

    except (ftplib.error_perm, ftplib.error_reply, ftplib.error_proto):
        pass

    return size, modified


# ============================================================
# JSON persistence
# ============================================================

def save_json_atomic(path, data):
    """Write JSON to a temporary file and atomically replace the destination."""

    path.parent.mkdir(parents=True, exist_ok=True)

    temp = path.with_suffix(path.suffix + ".tmp")

    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    os.replace(temp, path)


def save_manifest(data):
    save_json_atomic(MANIFEST, data)


def load_manifest():
    if MANIFEST.exists():
        with open(MANIFEST, encoding="utf-8") as f:
            return json.load(f)

    return {}


# ============================================================
# Context-camera downloads
# ============================================================

def download_file(
    ftp,
    remote_path,
    filename,
    local_path,
    remote_size,
    remote_modified,
    old_metadata
):
    """Download a context-camera archive, with validation and restartability."""

    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists() and remote_size is not None:
        local_size = local_path.stat().st_size

        if local_size == remote_size:

            if not old_metadata:
                print(
                    f"  [ADOPT] {filename} "
                    "(existing file matches remote size)"
                )

                return {
                    "size": remote_size,
                    "modified": remote_modified
                }

            if (
                old_metadata.get("size") == remote_size
                and old_metadata.get("modified") == remote_modified
            ):
                print(f"  [SKIP] {filename} (unchanged)")

                return {
                    "size": remote_size,
                    "modified": remote_modified
                }

    temp_path = local_path.with_suffix(local_path.suffix + ".part")

    print(f"  [DOWNLOAD] {filename}")

    try:
        with open(temp_path, "wb") as f:
            ftp.retrbinary(
                f"RETR {remote_path}/{filename}",
                f.write
            )

        downloaded_size = temp_path.stat().st_size

        if remote_size is not None and downloaded_size != remote_size:
            raise IOError(
                f"Size mismatch: expected {remote_size}, "
                f"received {downloaded_size}"
            )

        os.replace(temp_path, local_path)

        print(f"  [OK] {filename} ({downloaded_size:,} bytes)")

        return {
            "size": downloaded_size,
            "modified": remote_modified
        }

    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


# ============================================================
# Subject archive utilities
# ============================================================

def timestamp_sort_key(filename):
    """
    Sort timestamp-based filenames chronologically.

    Expected format:
        HH_MM_SS_microseconds

    Non-timestamp filenames are placed after timestamp filenames.
    """

    stem = Path(filename).stem
    match = TIMESTAMP_RE.fullmatch(stem)

    if match:
        return (
            0,
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4))
        )

    return (1, stem)


def relative_after_directory(filename, directory):
    """
    Return the path below a particular directory component.

    Example:
        subject_01/labels/subfolder/a.txt
        -> subfolder/a.txt
    """

    path = PurePosixPath(filename)

    if path.is_absolute() or ".." in path.parts or "\\" in filename:
        return None

    try:
        index = path.parts.index(directory)
    except ValueError:
        return None

    relative = path.parts[index + 1:]

    if not relative:
        return None

    return PurePosixPath(*relative).as_posix()


def extract_demographics(metadata):
    """Extract subject-level demographic fields from one metadata record."""

    try:
        face = metadata["person"]["face"]
    except (KeyError, TypeError):
        return {}

    if not isinstance(face, dict):
        return {}

    result = {}

    age = face.get("age")

    if isinstance(age, (int, float)) and not isinstance(age, bool):
        result["age"] = age

    gender = face.get("gender")

    if isinstance(gender, dict):
        gender_name = gender.get("gender_name")

        if isinstance(gender_name, str) and gender_name:
            result["gender"] = gender_name

    race = face.get("race")

    if isinstance(race, dict):
        dominant_race = race.get("dominant_race")

        if isinstance(dominant_race, str) and dominant_race:
            result["race"] = dominant_race

    return result


def extract_emotions(metadata):
    """Extract the seven emotion probabilities from one metadata record."""

    try:
        probabilities = (
            metadata["person"]["face"]["emotion"]["probability_emotion"]
        )
    except (KeyError, TypeError):
        return None

    if not isinstance(probabilities, dict):
        return None

    return {
        emotion: probabilities.get(emotion)
        for emotion in EMOTION_KEYS
    }


def safe_label_path(relative_path):
    """Validate a relative labels path before adding it to the output ZIP."""

    path = PurePosixPath(relative_path)

    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in relative_path
        or not path.parts
    ):
        raise ValueError(f"Unsafe labels path: {relative_path}")

    return path.as_posix()


# ============================================================
# Subject archive transformation
# ============================================================

def build_subject_archive(source_zip, output_zip, subject_id):
    """
    Transform an original subject archive into:

        subject_XX.zip
        ├── labels/
        ├── preview.png
        └── emotion.json

    Returns the subject-level demographics.
    """

    with zipfile.ZipFile(source_zip, "r") as source:

        label_files = []
        metadata_files = []
        image_files = []

        for info in source.infolist():

            if info.is_dir():
                continue

            filename = info.filename

            labels_relative = relative_after_directory(
                filename, "labels"
            )

            metadata_relative = relative_after_directory(
                filename, "metadata"
            )

            images_relative = relative_after_directory(
                filename, "images"
            )

            if labels_relative is not None:
                label_files.append((filename, labels_relative))

            elif (
                metadata_relative is not None
                and metadata_relative.lower().endswith(".json")
            ):
                metadata_files.append(filename)

            elif (
                images_relative is not None
                and images_relative.lower().endswith(".png")
            ):
                image_files.append(filename)

        if not label_files:
            raise ValueError(f"No labels files found in {source_zip}")

        if not metadata_files:
            raise ValueError(f"No metadata JSON files found in {source_zip}")

        if not image_files:
            raise ValueError(f"No PNG images found in {source_zip}")

        # Sort both metadata and images chronologically.
        metadata_files.sort(key=timestamp_sort_key)
        image_files.sort(key=timestamp_sort_key)

        # ----------------------------------------------------
        # Extract demographics and per-frame emotions
        # ----------------------------------------------------

        demographics = {}
        emotion_data = {}

        for filename in metadata_files:

            try:
                metadata = json.loads(source.read(filename))
            except (json.JSONDecodeError, UnicodeDecodeError):
                print(f"  [WARNING] Invalid metadata JSON: {filename}")
                continue

            # First chronologically valid observation for each field.
            extracted = extract_demographics(metadata)

            for key, value in extracted.items():
                if key not in demographics:
                    demographics[key] = value

            emotions = extract_emotions(metadata)

            if emotions is not None:
                timestamp = Path(filename).stem
                emotion_data[timestamp] = emotions

        if not demographics:
            print(f"  [WARNING] No demographic fields found for {subject_id}")

        # ----------------------------------------------------
        # Select earliest image
        # ----------------------------------------------------

        preview_source = image_files[0]
        preview_data = source.read(preview_source)

        # ----------------------------------------------------
        # Construct transformed ZIP
        # ----------------------------------------------------

        with zipfile.ZipFile(
            output_zip,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6
        ) as output:

            # Preserve labels and their relative directory structure.
            for original_path, relative_path in label_files:

                relative_path = safe_label_path(relative_path)
                output_path = f"labels/{relative_path}"

                output.writestr(
                    output_path,
                    source.read(original_path)
                )

            # Store the earliest image.
            output.writestr("preview.png", preview_data)

            # Store per-frame emotion probabilities.
            output.writestr(
                "emotion.json",
                json.dumps(emotion_data, indent=4)
            )

    return demographics


def is_subject_archive_valid(path):
    """Check ZIP integrity and the required output members."""

    try:
        with zipfile.ZipFile(path, "r") as archive:

            if archive.testzip() is not None:
                return False

            names = set(archive.namelist())

            has_labels = any(
                name.startswith("labels/")
                for name in names
            )

            return (
                has_labels
                and "preview.png" in names
                and "emotion.json" in names
            )

    except (zipfile.BadZipFile, OSError):
        return False


# ============================================================
# Subject processing
# ============================================================

def process_subject(
    ftp,
    remote_dir,
    filename,
    subjects_dir,
    remote_size,
    remote_modified,
    old_metadata,
    subject_metadata
):
    """
    Download and transform one subject archive.

    The original source ZIP is retained temporarily and is only
    removed by main() after the manifest has been persisted.
    """

    subjects_dir.mkdir(parents=True, exist_ok=True)

    subject_id = Path(filename).stem

    output_path = subjects_dir / filename
    source_path = subjects_dir / f"{subject_id}.source.zip"
    source_temp = source_path.with_suffix(".zip.part")

    generated_temp = subjects_dir / f"{subject_id}.generated.zip"

    demographics_path = subjects_dir / "metadata.json"

    # --------------------------------------------------------
    # Check whether an existing transformed archive is current
    # --------------------------------------------------------

    if (
        output_path.exists()
        and remote_size is not None
        and old_metadata.get("source_size") == remote_size
        and old_metadata.get("source_modified") == remote_modified
        and old_metadata.get("archive_size") == output_path.stat().st_size
        and subject_id in subject_metadata
        and is_subject_archive_valid(output_path)
    ):

        print(f"  [SKIP] {filename} (already processed)")

        source_path.unlink(missing_ok=True)
        source_temp.unlink(missing_ok=True)
        generated_temp.unlink(missing_ok=True)

        return old_metadata

    # --------------------------------------------------------
    # Download original subject archive
    # --------------------------------------------------------

    print(f"  [DOWNLOAD] {filename}")

    try:
        with open(source_temp, "wb") as f:
            ftp.retrbinary(
                f"RETR {remote_dir}/{filename}",
                f.write
            )

        downloaded_size = source_temp.stat().st_size

        if remote_size is not None and downloaded_size != remote_size:
            raise IOError(
                f"Size mismatch: expected {remote_size}, "
                f"received {downloaded_size}"
            )

        os.replace(source_temp, source_path)

        print(f"  [OK] Source downloaded ({downloaded_size:,} bytes)")

        # ----------------------------------------------------
        # Generate transformed archive
        # ----------------------------------------------------

        print(f"  [PROCESS] {filename}")

        demographics = build_subject_archive(
            source_path,
            generated_temp,
            subject_id
        )

        # Verify the generated archive before replacing anything.
        if not is_subject_archive_valid(generated_temp):
            raise IOError(
                f"Generated archive failed integrity verification: {filename}"
            )

        # ----------------------------------------------------
        # Replace final archive
        # ----------------------------------------------------

        os.replace(generated_temp, output_path)

        archive_size = output_path.stat().st_size

        print(
            f"  [OK] Processed {filename} "
            f"({archive_size:,} bytes)"
        )

        # ----------------------------------------------------
        # Update subject-level demographics
        # ----------------------------------------------------

        subject_metadata[subject_id] = demographics

        save_json_atomic(
            demographics_path,
            subject_metadata
        )

        return {
            "source_size": downloaded_size,
            "source_modified": remote_modified,
            "archive_size": archive_size
        }

    finally:
        # Temporary download/generation files are safe to remove.
        source_temp.unlink(missing_ok=True)
        generated_temp.unlink(missing_ok=True)


# ============================================================
# Main downloader
# ============================================================

def main():

    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    dataset = load_manifest()

    with ftplib.FTP() as ftp:

        ftp.connect(HOST, PORT, timeout=60)
        ftp.login(USER, PASSWORD)
        ftp.cwd(REMOTE_ROOT)

        print("Connection Successful")

        groups = sorted(
            x for x in listing(ftp)
            if GROUP_RE.fullmatch(x)
        )

        print(f"Found {len(groups)} groups.\n")

        for group in groups:
            if group in IGNORE_GROUPS:
                print(f'[IGNORE] {group}')
                continue
            
            dataset.setdefault(group, {})

            group_path = f"{REMOTE_ROOT}/{group}"

            ftp.cwd(group_path)

            experiments = sorted(
                x for x in listing(ftp)
                if EXP_RE.fullmatch(x)
            )

            print(
                f"\n=== {group} "
                f"({len(experiments)} experiments) ==="
            )

            for experiment in experiments:
                if experiment in IGNORE_EXPERIMENTS:
                    print(f"  [IGNORE] {experiment}")
                    continue


                remote_dir = f"{group_path}/{experiment}"

                ftp.cwd(remote_dir)

                files = listing(ftp)

                context_files = sorted(
                    x for x in files
                    if CONTEXT_RE.fullmatch(x)
                )

                subject_files = sorted(
                    x for x in files
                    if SUBJECT_RE.fullmatch(x)
                )

                entry = dataset[group].setdefault(
                    experiment,
                    {
                        "context_cam_files": {},
                        "subjects": {}
                    }
                )

                # Migrate existing manifest entries.
                entry.pop("subject_zip_count", None)
                entry.setdefault("context_cam_files", {})
                entry.setdefault("subjects", {})

                local_dir = DATASET_ROOT / group / experiment
                subjects_dir = local_dir / "subjects"

                print(
                    f"\n[{experiment}] "
                    f"{len(context_files)} context files, "
                    f"{len(subject_files)} subject files"
                )

                # ------------------------------------------------
                # Context-camera archives
                # ------------------------------------------------

                for filename in context_files:
                    if filename[:-4] in IGNORE_CONTEXT_CAMS:
                        print(f'[IGNORE] {filename}')
                        continue

                    size, modified = remote_metadata(
                        ftp, filename
                    )

                    old_metadata = (
                        entry["context_cam_files"].get(filename, {})
                    )

                    local_path = local_dir / filename

                    try:
                        metadata = download_file(
                            ftp,
                            remote_dir,
                            filename,
                            local_path,
                            size,
                            modified,
                            old_metadata
                        )

                        entry["context_cam_files"][filename] = metadata

                        save_manifest(dataset)

                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"  [ERROR] {filename}: {e}")

                # ------------------------------------------------
                # Subject-level metadata
                # ------------------------------------------------

                demographics_path = subjects_dir / "metadata.json"

                if demographics_path.exists():
                    with open(demographics_path, encoding="utf-8") as f:
                        subject_metadata = json.load(f)
                else:
                    subject_metadata = {}

                # ------------------------------------------------
                # Subject-camera archives
                # ------------------------------------------------

                for filename in subject_files:
                    size, modified = remote_metadata(
                        ftp, filename
                    )

                    old_metadata = entry["subjects"].get(
                        filename, {}
                    )

                    try:
                        metadata = process_subject(
                            ftp,
                            remote_dir,
                            filename,
                            subjects_dir,
                            size,
                            modified,
                            old_metadata,
                            subject_metadata
                        )

                        entry["subjects"][filename] = metadata

                        # Persist the manifest before deleting the source ZIP.
                        save_manifest(dataset)

                        # Original source ZIP is no longer needed.
                        subject_id = Path(filename).stem

                        source_path = (
                            subjects_dir / f"{subject_id}.source.zip"
                        )

                        source_path.unlink(missing_ok=True)

                    except Exception as e:
                        import traceback
                        traceback.print_exc()   
                        print(f"  [ERROR] {filename}: {e}")

                # Persist experiment progress even if no files were processed.
                save_manifest(dataset)

            print(f"\n[DONE] {group}")

    print("\n[COMPLETE] Downloader finished.")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()