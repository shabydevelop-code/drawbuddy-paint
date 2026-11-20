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

    def publish(self):
        if not (self.root / ".git").exists():
            raise RuntimeError("Git checkout required to publish the result")
        # Never stage other working-tree files.
        self._git("add", "--", "output/dog.png")
        if not self._git("diff", "--cached", "--name-only", "--", "output/dog.png"):
            print("Result unchanged; no GitHub upload needed")
            return
        self._git("commit", "-m", "Update current Paint result", "--", "output/dog.png")
        self._git("push", "origin", "HEAD:main")
        print("Published current result: output/dog.png")
