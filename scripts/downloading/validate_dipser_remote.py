"""Check the local DIPSER copy against the FTP server without downloading the archives again.

  python scripts/downloading/validate_dipser_remote.py [--groups group_01 ...] [--experiments ...]

The per-student archives on the server hold each student's labels plus their whole video feed and
watch data (~600 GB in total); the downloader keeps only the labels, an emotion summary and a
preview. To compare without re-downloading, the server file is opened with Python's own `zipfile`
through a file-like object that fetches only the byte ranges `zipfile` asks for (FTP resume,
`REST`): the ZIP's directory at the end of the file, and the few small members actually read.
No ZIP parsing here; `zipfile` does all of it, and checks each member's CRC as it reads it.

Per archive on the server:
  camera ZIP   exists locally; same size; same members, each with the same size and CRC-32
  --crc        also re-read every local archive and check each member against its CRC (zipfile's
               testzip): the server comparison proves the local index matches the server, this
               proves the bytes on disk match the index (camera archives: ~1 h in total)
  subject ZIP  every label-looking member (*.json with "label" in its path) is listed; members the
               downloader's rule would drop (no exact `labels` path component, or a backslash) are
               flagged; each label is read from the server and compared byte for byte with the
               local archive; if the local archive is missing, says which of the downloader's
               requirements (labels, metadata JSON, PNG images) the server archive fails

One JSON line per archive in data/DIPSER/validation_remote.jsonl (resumable: done archives are
skipped). Summary at the end. Uses the downloader's server settings and .env credentials.
"""

from __future__ import annotations

import argparse
import ftplib
import io
import json
import time
import zipfile
from pathlib import Path

from scripts.downloading.dipser import (CONTEXT_RE, DATASET_ROOT, EXP_RE, GROUP_RE, HOST, LOCAL_ROOT, PASSWORD, PORT,
                                        REMOTE_ROOT, SUBJECT_RE, USER, listing, relative_after_directory)

OUT = LOCAL_ROOT / "validation_remote.jsonl"


class Server:
    """One FTP connection that can reconnect; fetches byte ranges of a file."""

    def __init__(self):
        self.ftp = None
        self.connect()

    def connect(self):
        if self.ftp is not None:
            try:
                self.ftp.close()
            except Exception:
                pass
        self.ftp = ftplib.FTP()
        self.ftp.connect(HOST, PORT, timeout=120)
        self.ftp.login(USER, PASSWORD)
        self.ftp.voidcmd("TYPE I")

    def call(self, fn, *args):
        """Run fn(ftp, *args); on a protocol or connection error reconnect once and retry."""
        try:
            return fn(self.ftp, *args)
        except (ftplib.all_errors, EOFError):
            self.connect()
            return fn(self.ftp, *args)

    def size(self, path: str) -> int:
        return self.call(lambda ftp: ftp.size(path))

    def list(self, path: str) -> list[str]:
        def go(ftp):
            ftp.cwd(path)
            return listing(ftp)
        return self.call(go)

    def read_range(self, path: str, offset: int, length: int) -> bytes:
        def go(ftp):
            conn = ftp.transfercmd(f"RETR {path}", rest=offset)
            data = bytearray()
            while len(data) < length:
                chunk = conn.recv(min(1 << 16, length - len(data)))
                if not chunk:
                    break
                data += chunk
            conn.close()
            try:
                ftp.voidresp()                      # 226 if the file ended, 426/451 if we stopped early
            except ftplib.all_errors:
                pass
            return bytes(data)
        return self.call(go)


class RemoteFile(io.RawIOBase):
    """Seekable, read-only view of a server file; reads come from cached blocks of `block` bytes."""

    def __init__(self, server: Server, path: str, size: int, block: int = 1 << 18):
        self.server, self.path, self.size, self.block = server, path, size, block
        self.pos, self.cache, self.fetched = 0, {}, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def _block(self, k: int) -> bytes:
        if k not in self.cache:
            start = k * self.block
            self.cache[k] = self.server.read_range(self.path, start, min(self.block, self.size - start))
            self.fetched += len(self.cache[k])
        return self.cache[k]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while len(out) < n:
            k, off = divmod(self.pos + len(out), self.block)
            out += self._block(k)[off: off + n - len(out)]
        self.pos += len(out)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def local_crc(local: Path) -> str | None:
    """Name of the first local member whose bytes fail its CRC, or None."""
    with zipfile.ZipFile(local) as z:
        return z.testzip()


def check_camera(server: Server, remote: str, local: Path, crc: bool = False) -> dict:
    r = {"kind": "camera", "problems": []}
    size = server.size(remote)
    r["server_size"] = size
    if not local.exists():
        r["problems"].append("missing locally")
        return r
    r["local_size"] = local.stat().st_size
    if r["local_size"] != size:
        r["problems"].append(f"size differs: local {r['local_size']}, server {size}")
    rf = RemoteFile(server, remote, size)
    with zipfile.ZipFile(rf) as rz, zipfile.ZipFile(local) as lz:
        rmem = {i.filename: (i.file_size, i.CRC) for i in rz.infolist()}
        lmem = {i.filename: (i.file_size, i.CRC) for i in lz.infolist()}
    only_server, only_local = sorted(set(rmem) - set(lmem)), sorted(set(lmem) - set(rmem))
    differ = sorted(n for n in set(rmem) & set(lmem) if rmem[n] != lmem[n])
    r.update(members=len(rmem), fetched_bytes=rf.fetched)
    if only_server:
        r["problems"].append(f"{len(only_server)} members only on the server, e.g. {only_server[0]}")
    if only_local:
        r["problems"].append(f"{len(only_local)} members only locally, e.g. {only_local[0]}")
    if differ:
        r["problems"].append(f"{len(differ)} members differ in size or CRC, e.g. {differ[0]}")
    if crc and (bad := local_crc(local)):
        r["problems"].append(f"local bytes fail their CRC: {bad}")
    return r


def check_subject(server: Server, remote: str, local: Path, crc: bool = False) -> dict:
    r = {"kind": "subject", "problems": []}
    size = server.size(remote)
    r["server_size"] = size
    rf = RemoteFile(server, remote, size)
    with zipfile.ZipFile(rf) as rz:
        names = [i.filename for i in rz.infolist() if not i.is_dir()]
        labelish = sorted(n for n in names if n.lower().endswith(".json") and "label" in n.lower())
        kept = {n: relative_after_directory(n, "labels") for n in labelish}
        r["server_labels"] = labelish
        dropped = [n for n, rel in kept.items() if rel is None]
        if dropped:
            r["problems"].append(f"label-looking files the downloader's rule drops: {dropped}")
        backslash = [n for n in names if "\\" in n]
        if backslash:
            r["problems"].append(f"{len(backslash)} member paths contain a backslash, e.g. {backslash[0]}")
        # what the downloader requires to build the local archive
        has_meta = any(relative_after_directory(n, "metadata") and n.lower().endswith(".json") for n in names)
        has_png = any(relative_after_directory(n, "images") and n.lower().endswith(".png") for n in names)
        r["builds"] = {"labels": any(kept.values()), "metadata_json": has_meta, "png_images": has_png}
        if not local.exists():
            failing = [k for k, ok in r["builds"].items() if not ok]
            r["problems"].append("missing locally" + (f"; the downloader would fail: no {', '.join(failing)}"
                                                      if failing else "; the downloader's requirements are met"))
        else:
            with zipfile.ZipFile(local) as lz:
                local_names = set(lz.namelist())
                for n, rel in kept.items():
                    if rel is None:
                        continue
                    target = f"labels/{rel}"
                    if target not in local_names:
                        r["problems"].append(f"label missing locally: {target}")
                    elif rz.read(n) != lz.read(target):            # zipfile verifies the CRC while reading
                        r["problems"].append(f"label differs from the server: {target}")
                server_targets = {f"labels/{rel}" for rel in kept.values() if rel}
                extra = sorted(n for n in local_names if n.startswith("labels/") and n not in server_targets)
                if extra:
                    r["problems"].append(f"labels only locally: {extra}")
            if crc and (bad := local_crc(local)):
                r["problems"].append(f"local bytes fail their CRC: {bad}")
    r["fetched_bytes"] = rf.fetched
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="*")
    ap.add_argument("--experiments", nargs="*")
    ap.add_argument("--kind", choices=["all", "camera", "subject"], default="all")
    ap.add_argument("--limit", type=int, help="stop after this many archives (for a quick trial)")
    ap.add_argument("--crc", action="store_true", help="also check every local archive's bytes against its CRCs")
    args = ap.parse_args()
    done = set()
    if OUT.exists():
        done = {json.loads(l)["remote"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
    server, start, n_checked, fetched = Server(), time.monotonic(), 0, 0
    jobs = []
    for g in sorted(x for x in server.list(REMOTE_ROOT) if GROUP_RE.fullmatch(x)):
        if args.groups and g not in args.groups:
            continue
        for e in sorted(x for x in server.list(f"{REMOTE_ROOT}/{g}") if EXP_RE.fullmatch(x)):
            if args.experiments and e not in args.experiments:
                continue
            files = server.list(f"{REMOTE_ROOT}/{g}/{e}")
            local_dir = DATASET_ROOT / g / e
            if args.kind in ("all", "camera"):
                jobs += [("camera", f"{REMOTE_ROOT}/{g}/{e}/{f}", local_dir / f) for f in sorted(files) if CONTEXT_RE.fullmatch(f)]
            if args.kind in ("all", "subject"):
                jobs += [("subject", f"{REMOTE_ROOT}/{g}/{e}/{f}", local_dir / "subjects" / f)
                         for f in sorted(files) if SUBJECT_RE.fullmatch(f)]
    todo = [j for j in jobs if j[1] not in done][: args.limit]
    print(f"{len(jobs)} archives on the server, {len(jobs) - len(todo) if not args.limit else len(done)} already checked, "
          f"{len(todo)} to check", flush=True)
    for kind, remote, local in todo:
        try:
            r = (check_camera if kind == "camera" else check_subject)(server, remote, local, args.crc)
        except Exception as e:                                  # record and move on; nothing is lost
            r = {"kind": kind, "problems": [f"check failed: {type(e).__name__}: {e}"]}
        r["remote"] = remote
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
        n_checked += 1
        fetched += r.get("fetched_bytes", 0)
        status = "ok" if not r["problems"] else "; ".join(r["problems"])
        print(f"[{n_checked}/{len(todo)}] {remote}: {status}  ({r.get('fetched_bytes', 0) / 2**20:.1f} MB fetched, "
              f"{(time.monotonic() - start) / 60:.1f} min)", flush=True)
    results = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()] if OUT.exists() else []
    bad = [r for r in results if r["problems"]]
    print(f"\n{len(results)} archives checked so far, {len(bad)} with problems; "
          f"{fetched / 2**20:.0f} MB fetched this run -> {OUT}")
    for r in bad:
        print(f"  {r['remote']}: {'; '.join(r['problems'])}")


if __name__ == "__main__":
    main()
