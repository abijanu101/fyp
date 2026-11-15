
import re
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from bisect import bisect_right

import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QComboBox,
    QSlider,
    QCheckBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# ---------------------------------------------------------
# Timestamp utilities
# ---------------------------------------------------------

def timestamp_to_us(name):
    """
    Convert HH_MM_SS_microseconds into integer microseconds.

    Returns None if the filename does not follow the
    expected timestamp convention.
    """

    stem = Path(name).stem
    match = re.fullmatch(
        r"(\d+)_(\d{2})_(\d{2})_(\d+)",
        stem
    )

    if not match:
        return None

    hours, minutes, seconds, fraction = match.groups()

    # Normalize fractional seconds to microseconds.
    fraction = (fraction + "000000")[:6]

    return (
        int(hours) * 3_600_000_000
        + int(minutes) * 60_000_000
        + int(seconds) * 1_000_000
        + int(fraction)
    )


def format_timestamp(name):
    """Format a timestamp filename for display."""

    value = timestamp_to_us(name)

    if value is None:
        return name

    hours = value // 3_600_000_000
    minutes = (value // 60_000_000) % 60
    seconds = (value // 1_000_000) % 60
    micros = value % 1_000_000

    return f"{hours:02}:{minutes:02}:{seconds:02}.{micros:06}"


# ---------------------------------------------------------
# Camera sequence loader
# ---------------------------------------------------------

class FrameSource:
    """Read timestamp-indexed frames from a ZIP or directory."""

    def __init__(self, path):
        self.path = Path(path)
        self.archive = None
        self.frames = {}

        if self.path.is_file():
            self.archive = zipfile.ZipFile(self.path, "r")

            for member in self.archive.namelist():
                p = PurePosixPath(member)

                if (
                    len(p.parts) != 2
                    or p.parts[0] != "images"
                    or p.suffix.lower() not in IMAGE_EXTENSIONS
                    or p.name.startswith(".")
                ):
                    continue

                self.frames[p.stem] = member

        elif self.path.is_dir():
            for frame in self.path.rglob("*"):
                if frame.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                self.frames[frame.stem] = frame

        self.timestamps = sorted(
            self.frames,
            key=lambda name: (
                timestamp_to_us(name) is None,
                timestamp_to_us(name)
                if timestamp_to_us(name) is not None
                else name
            )
        )

        self.timestamp_values = [
            timestamp_to_us(name)
            for name in self.timestamps
        ]

        self.cached_timestamp = None
        self.cached_frame = None

    def read(self, timestamp):
        """Return a decoded OpenCV BGR image."""

        if timestamp not in self.frames:
            return None

        location = self.frames[timestamp]

        if self.archive:
            data = self.archive.read(location)
        else:
            data = Path(location).read_bytes()

        buffer = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    def read_latest(self, timestamp):
        """
        Return the latest frame whose timestamp is <= target timestamp.
        Reuse the decoded image if the selected frame has not changed.
        """

        target = timestamp_to_us(timestamp)

        if target is None:
            return None, None

        index = bisect_right(
            self.timestamp_values,
            target
        ) - 1

        if index < 0:
            return None, None

        frame_timestamp = self.timestamps[index]

        if frame_timestamp != self.cached_timestamp:
            self.cached_frame = self.read(frame_timestamp)
            self.cached_timestamp = frame_timestamp

        return frame_timestamp, self.cached_frame

    def close(self):
        if self.archive:
            self.archive.close()
            self.archive = None


# ---------------------------------------------------------
# Individual camera display
# ---------------------------------------------------------

class CameraPane(QGroupBox):

    def __init__(self, camera_id):
        super().__init__(f"Context Cam {camera_id}")

        self.camera_id = camera_id
        self.source = None
        self.last_pixmap = None

        self.last_source_timestamp = None
                
        self.image_label = QLabel("Not loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(320, 180)
        self.image_label.setStyleSheet(
            "background-color: #202020; color: white;"
        )

        self.info_label = QLabel("")
        self.info_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.hold_checkbox = QCheckBox("Hold frame")

        layout = QVBoxLayout()
        layout.addWidget(self.image_label, stretch=1)
        layout.addWidget(self.info_label)
        layout.addWidget(self.hold_checkbox)

        self.setLayout(layout)

    def set_source(self, source):
        self.source = source
        self.last_pixmap = None
        self.last_source_timestamp = None
        self.hold_checkbox.setChecked(False)

        if source is None:
            self.show_message("Camera unavailable")
        else:
            self.info_label.setText(
                f"{len(source.frames):,} frames"
            )

            if not source.frames:
                self.show_message("No images found")

    def show_message(self, message):
        self.last_pixmap = None
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(message)

    def display_frame(self, frame, timestamp, source_timestamp):
        if self.hold_checkbox.isChecked():
            return

        if source_timestamp == self.last_source_timestamp:
            return

        if frame is None:
            self.show_message("No frame at this timestamp")
            return

        height, width, channels = frame.shape
        bytes_per_line = channels * width

        image = QImage(
            frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_BGR888
        ).copy()

        self.last_pixmap = QPixmap.fromImage(image)
        self.last_source_timestamp = source_timestamp

        self.info_label.setText(
            f"{width} × {height} | {format_timestamp(source_timestamp)}"
        )

        self.refresh_image()

    def refresh_image(self):
        if self.last_pixmap is None:
            return

        self.image_label.setText("")
        self.image_label.setPixmap(
            self.last_pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_image()


# ---------------------------------------------------------
# Main application
# ---------------------------------------------------------

class DatasetExplorer(QMainWindow):

    def __init__(self, dataset_root):
        super().__init__()

        self.dataset_root = Path(dataset_root)

        self.sources = {}
        self.timeline = []
        self.current_index = 0
        self.playing = False

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.advance)

        self.setWindowTitle("Classroom Dataset Explorer")
        self.resize(1500, 1000)

        self.playback_wall_start = None
        self.playback_data_start_us = None

        self.build_ui()
        self.discover_groups()

    # -----------------------------------------------------
    # UI construction
    # -----------------------------------------------------

    def build_ui(self):

        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)

        # Dataset navigation
        navigation = QHBoxLayout()

        self.group_combo = QComboBox()
        self.experiment_combo = QComboBox()

        navigation.addWidget(QLabel("Group:"))
        navigation.addWidget(self.group_combo)

        navigation.addWidget(QLabel("Experiment:"))
        navigation.addWidget(self.experiment_combo)

        navigation.addStretch()

        root_layout.addLayout(navigation)

        # Six camera displays
        camera_grid = QGridLayout()

        self.camera_panes = {}

        for camera_id in range(1, 7):
            pane = CameraPane(camera_id)
            self.camera_panes[camera_id] = pane

            row = (camera_id - 1) // 2
            col = (camera_id - 1) % 2

            camera_grid.addWidget(pane, row, col)

        root_layout.addLayout(camera_grid, stretch=1)

        # Playback controls
        controls = QHBoxLayout()

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)

        self.previous_button = QPushButton("Previous")
        self.previous_button.clicked.connect(self.previous_frame)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_frame)

        self.timestamp_label = QLabel("00:00:00.000000")

        self.speed_combo = QComboBox()

        for speed in [0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0, 16.0, 32.0]:
            self.speed_combo.addItem(f"{speed}×", speed)

        self.speed_combo.setCurrentIndex(2)

        controls.addWidget(self.previous_button)
        controls.addWidget(self.play_button)
        controls.addWidget(self.next_button)
        controls.addWidget(self.timestamp_label)
        controls.addStretch()
        controls.addWidget(QLabel("Speed:"))
        controls.addWidget(self.speed_combo)

        root_layout.addLayout(controls)

        # Timeline
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.seek)

        root_layout.addWidget(self.slider)

        # Status
        self.status_label = QLabel("Select an experiment.")
        root_layout.addWidget(self.status_label)

        self.group_combo.currentIndexChanged.connect(
            self.populate_experiments
        )

        self.experiment_combo.currentIndexChanged.connect(
            self.load_experiment
        )

    # -----------------------------------------------------
    # Dataset discovery
    # -----------------------------------------------------

    def discover_groups(self):

        groups = sorted(
            (
                p for p in self.dataset_root.iterdir()
                if p.is_dir()
                and re.fullmatch(r"group_\d+", p.name)
            ),
            key=lambda p: int(p.name.split("_")[1])
        )

        for group in groups:
            self.group_combo.addItem(group.name, group)

        if groups:
            self.populate_experiments()

    def populate_experiments(self):

        self.experiment_combo.clear()

        group = self.group_combo.currentData()

        if group is None:
            return

        experiments = sorted(
            (
                p for p in group.iterdir()
                if p.is_dir()
                and re.fullmatch(r"experiment_\d+", p.name)
            ),
            key=lambda p: int(p.name.split("_")[1])
        )

        for experiment in experiments:
            self.experiment_combo.addItem(
                experiment.name,
                experiment
            )

        if experiments:
            self.load_experiment()

    # -----------------------------------------------------
    # Experiment loading
    # -----------------------------------------------------

    def load_experiment(self):

        self.stop_playback()
        self.close_sources()

        experiment = self.experiment_combo.currentData()

        if experiment is None:
            return

        self.sources = {}
        available_timestamps = set()

        for camera_id in range(1, 7):

            camera_name = f"context_cam_{camera_id:02d}"

            archive_path = experiment / f"{camera_name}.zip"
            directory_path = experiment / camera_name

            source_path = None

            if archive_path.is_file():
                source_path = archive_path
            elif directory_path.is_dir():
                source_path = directory_path

            if source_path is None:
                self.camera_panes[camera_id].set_source(None)
                continue

            try:
                source = FrameSource(source_path)

            except (OSError, zipfile.BadZipFile) as error:
                self.camera_panes[camera_id].set_source(None)
                print(f"[ERROR] {source_path}: {error}")
                continue

            self.sources[camera_id] = source
            self.camera_panes[camera_id].set_source(source)

            available_timestamps.update(source.timestamps)

        # Union timeline: any timestamp available in any camera.
        self.timeline = sorted(
            available_timestamps,
            key=lambda name: (
                timestamp_to_us(name) is None,
                timestamp_to_us(name)
                if timestamp_to_us(name) is not None
                else name
            )
        )
        self.timeline_values = [
            timestamp_to_us(timestamp)
            for timestamp in self.timeline
        ]

        self.current_index = 0

        self.slider.blockSignals(True)
        self.slider.setRange(
            0,
            max(0, len(self.timeline) - 1)
        )
        self.slider.setValue(0)
        self.slider.blockSignals(False)

        available = len(self.sources)

        self.status_label.setText(
            f"{experiment} | "
            f"{available}/6 cameras available | "
            f"{len(self.timeline):,} unique timestamps"
        )

        if self.timeline:
            self.render_current_frame()
        else:
            self.timestamp_label.setText("No frames found")

    # -----------------------------------------------------
    # Frame rendering
    # -----------------------------------------------------

    def render_current_frame(self):

        if not self.timeline:
            return

        timestamp = self.timeline[self.current_index]

        self.timestamp_label.setText(
            format_timestamp(timestamp)
        )

        for camera_id in range(1, 7):

            pane = self.camera_panes[camera_id]
            source = self.sources.get(camera_id)

            if source is None:
                continue

            # A held camera retains its previous image.
            if pane.hold_checkbox.isChecked():
                continue

            frame_timestamp, frame = source.read_latest(timestamp)

            pane.display_frame(
                frame,
                timestamp,
                frame_timestamp
            )

        self.slider.blockSignals(True)
        self.slider.setValue(self.current_index)
        self.slider.blockSignals(False)

    # -----------------------------------------------------
    # Playback
    # -----------------------------------------------------

    def toggle_playback(self):

        if self.playing:
            self.stop_playback()
            return

        if not self.timeline:
            return

        self.playback_wall_start = time.monotonic()
        self.playback_data_start_us = self.timeline_values[self.current_index]

        if self.current_index >= len(self.timeline) - 1:
            self.current_index = 0
            self.render_current_frame()

        self.playing = True
        self.play_button.setText("Pause")

        self.schedule_next_frame()

    def stop_playback(self):

        self.timer.stop()
        self.playing = False
        self.play_button.setText("Play")

    def schedule_next_frame(self):
        self.timer.start(16)  # Approximately 30 renders per second

    def advance(self):
        if not self.playing:
            return

        elapsed_seconds = time.monotonic() - self.playback_wall_start
        speed = self.speed_combo.currentData()

        target_timestamp = (
            self.playback_data_start_us
            + elapsed_seconds * speed * 1_000_000
        )

        self.current_index = min(
            bisect_right(self.timeline_values, target_timestamp) - 1,
            len(self.timeline) - 1
        )

        self.render_current_frame()

        if self.current_index >= len(self.timeline) - 1:
            self.stop_playback()
        else:
            self.schedule_next_frame()

    # -----------------------------------------------------
    # Manual navigation
    # -----------------------------------------------------

    def seek(self, index):

        if not self.timeline:
            return

        self.current_index = index
        self.render_current_frame()

        if self.playing:
            self.schedule_next_frame()

    def previous_frame(self):

        if self.current_index > 0:
            self.current_index -= 1
            self.render_current_frame()

            if self.playing:
                self.schedule_next_frame()

    def next_frame(self):

        if self.current_index < len(self.timeline) - 1:
            self.current_index += 1
            self.render_current_frame()

            if self.playing:
                self.schedule_next_frame()

    # -----------------------------------------------------
    # Cleanup
    # -----------------------------------------------------

    def close_sources(self):

        for source in self.sources.values():
            source.close()

        self.sources.clear()

    def closeEvent(self, event):

        self.stop_playback()
        self.close_sources()

        super().closeEvent(event)


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

def main():

    if len(sys.argv) < 2:
        print(
            "Usage: python dataset_explorer.py "
            '"path/to/dataset"'
        )
        sys.exit(1)

    dataset_root = Path(sys.argv[1]).resolve()

    if not dataset_root.is_dir():
        print(f"Dataset directory not found: {dataset_root}")
        sys.exit(1)

    app = QApplication(sys.argv)

    window = DatasetExplorer(dataset_root)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
