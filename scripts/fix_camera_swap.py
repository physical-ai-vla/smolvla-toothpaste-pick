"""
Swap the up/side camera streams for episodes 8-14 only
(the camera cables were swapped while those episodes were recorded).
"""
import os, glob, subprocess, tempfile, shutil
import pyarrow.parquet as pq
import pyarrow as pa
import cv2

DATASET_ROOT = os.path.expanduser("~/.cache/huggingface/lerobot/local/toothpaste_grasp")
SWAP_START = 8
SWAP_END   = 14  # inclusive


def main():
    ep_file = os.path.join(DATASET_ROOT, "meta/episodes/chunk-000/file-000.parquet")
    t = pq.read_table(ep_file)
    d = t.to_pydict()
    n = len(d["episode_index"])

    up_src   = os.path.join(DATASET_ROOT, "videos/observation.images.up/chunk-000/file-000.mp4")
    side_src = os.path.join(DATASET_ROOT, "videos/observation.images.side/chunk-000/file-000.mp4")

    tmp_dir = tempfile.mkdtemp()
    up_clips   = []
    side_clips = []

    for i in range(n):
        ep_idx = d["episode_index"][i]
        uf = d["videos/observation.images.up/from_timestamp"][i]
        ud = d["videos/observation.images.up/to_timestamp"][i] - uf
        sf = d["videos/observation.images.side/from_timestamp"][i]
        sd = d["videos/observation.images.side/to_timestamp"][i] - sf

        print(f"  ep{ep_idx:02d} extracting...", end="\r", flush=True)
        clip_a = os.path.join(tmp_dir, f"a_{i:03d}.mp4")
        clip_b = os.path.join(tmp_dir, f"b_{i:03d}.mp4")

        if SWAP_START <= ep_idx <= SWAP_END:
            # swap: side -> up slot, up -> side slot
            print(f"  ep{ep_idx:02d} SWAP", flush=True)
            _cut(side_src, sf, sd, clip_a)
            _cut(up_src,   uf, ud, clip_b)
        else:
            _cut(up_src,   uf, ud, clip_a)
            _cut(side_src, sf, sd, clip_b)

        up_clips.append(clip_a)
        side_clips.append(clip_b)

    print(f"\nExtraction done, concatenating...")

    for cam, clips in [("observation.images.up", up_clips),
                       ("observation.images.side", side_clips)]:
        out = os.path.join(DATASET_ROOT, f"videos/{cam}/chunk-000/file-000.mp4")
        tmp_out = out + ".new.mp4"
        _concat(clips, tmp_out)
        os.replace(tmp_out, out)
        print(f"  {cam} done")

    # recompute timestamps
    rows = {k: list(v) for k, v in d.items()}
    up_ts = side_ts = 0.0
    for i in range(n):
        cap = cv2.VideoCapture(up_clips[i])
        dur_up = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / (cap.get(cv2.CAP_PROP_FPS) or 30)
        cap.release()
        cap = cv2.VideoCapture(side_clips[i])
        dur_side = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / (cap.get(cv2.CAP_PROP_FPS) or 30)
        cap.release()

        for cam_key, ts, dur in [
            ("observation.images.up",   up_ts,   dur_up),
            ("observation.images.side", side_ts, dur_side),
        ]:
            rows[f"videos/{cam_key}/chunk_index"][i]    = 0
            rows[f"videos/{cam_key}/file_index"][i]     = 0
            rows[f"videos/{cam_key}/from_timestamp"][i] = ts
            rows[f"videos/{cam_key}/to_timestamp"][i]   = ts + dur

        up_ts   += dur_up
        side_ts += dur_side

    pq.write_table(pa.table({k: pa.array(v) for k, v in rows.items()}), ep_file)
    shutil.rmtree(tmp_dir)
    print("\nDone. Check the result with view_episodes.py.")


def _cut(src, t_from, dur, out):
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(t_from), "-i", src,
        "-t", str(dur),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", out
    ], check=True)


def _concat(clips, out):
    lst = out + ".list.txt"
    with open(lst, "w") as f:
        for p in clips:
            f.write(f"file '{p}'\n")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", lst,
        "-c", "copy", out
    ], check=True)
    os.remove(lst)


if __name__ == "__main__":
    main()
