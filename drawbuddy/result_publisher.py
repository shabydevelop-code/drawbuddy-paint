"""Capture the Paint canvas and publish one current result to GitHub.

Publishing is explicit: --publish-result. Requires git push credentials.
"""
import subprocess
from pathlib import Path

import pyautogui as pg


class ResultPublisher:
    def __init__(self, root):
        self.root = Path(root)
        self.output = self.root / "output" / "dog.png"

    def capture(self, paint):
        # The drawing is mapped to the fixed safe rectangle in Paint.
        left, top = paint.point(300, 148)
        right, bottom = paint.point(860, 550)
        screen = pg.screenshot()
        if not (0 <= left < right <= screen.width and
                0 <= top < bottom <= screen.height):
            raise RuntimeError("Paint canvas crop is outside the screen")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        screen.crop((left, top, right, bottom)).save(self.output)
        print(f"Captured canvas: {self.output}")

    def _git(self, *args):
        return subprocess.run(
            ["git", *args], cwd=self.root, check=True, text=True,
            capture_output=True).stdout.strip()

    def publish_file(self, path):
        """Publish only an approved output image, never unrelated files."""
        if not (self.root / ".git").exists():
            raise RuntimeError("Git checkout required to publish the result")
        target = Path(path).resolve()
        output_dir = (self.root / "output").resolve()
        if target.parent != output_dir or target.name not in (
                "dog.png", "fill-diagnostic.png", "gap-diagnostic.png", "closure-diagnostic.png", "closure-preview.png", "closure-color-diagnostic.png", "color-preview.png"):
            raise ValueError("Only approved output images may be published")
        if not target.is_file():
            raise FileNotFoundError(target)
        relative = target.relative_to(self.root.resolve()).as_posix()
        self._git("add", "--", relative)
        if not self._git("diff", "--cached", "--name-only", "--", relative):
            print(f"Result unchanged; no GitHub upload needed: {relative}")
            return
        self._git("commit", "-m", f"Update {target.name}", "--", relative)
        self._git("push", "origin", "HEAD:main")
        print(f"Published current result: {relative}")

    def publish(self):
        self.publish_file(self.output)
