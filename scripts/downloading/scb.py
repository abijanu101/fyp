
from pathlib import Path
from huggingface_hub import snapshot_download

DEST = Path(r"E:\Coding\Projects\fyp\data\SCB-ST-Dataset4")
REPO_ID = "wintonYF/SCB-Dataset"

def main():
    DEST.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO_ID}")
    print(f"Destination: {DEST}")
    print("This may take a while. Do not close the terminal.")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(DEST),
        max_workers=4,
    )

    print(f"\nDownload complete: {DEST}")

if __name__ == "__main__":
    main()
