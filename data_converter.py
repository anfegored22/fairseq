from pathlib import Path
import subprocess
src_root = Path("data/ups_ps/raw_mp3")
dst_root = Path("data/ups_ps/wav16k")
ok = 0
bad = 0
for mp3 in src_root.rglob("*.mp3"):
    rel = mp3.relative_to(src_root).with_suffix(".wav")
    wav = dst_root / rel
    wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", str(mp3),
        "-ac", "1", "-ar", "16000",
        str(wav),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode == 0:
        ok += 1
    else:
        bad += 1
        print(f"[SKIP] {mp3}")
        if p.stderr:
            print(p.stderr.strip().splitlines()[-1])
print(f"done: ok={ok}, bad={bad}")
