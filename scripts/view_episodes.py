"""
Episode viewer + deletion tool.
Videos play in an OpenCV window; commands are entered in the terminal.
"""
import cv2
import os
import glob
import json
import subprocess
import tempfile
import threading
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

DATASET_ROOT = os.path.expanduser("~/.cache/huggingface/lerobot/local/toothpaste_grasp")


def load_all_episodes():
    ep_files = sorted(glob.glob(
        os.path.join(DATASET_ROOT, "meta/episodes/**/*.parquet"), recursive=True
    ))
    all_eps = []
    for f in ep_files:
        t = pq.read_table(f)
        d = t.to_pydict()
        for i in range(len(d["episode_index"])):
            all_eps.append({
                "ep_idx":      d["episode_index"][i],
                "up_chunk":    d["videos/observation.images.up/chunk_index"][i],
                "up_file":     d["videos/observation.images.up/file_index"][i],
                "up_t_from":   d["videos/observation.images.up/from_timestamp"][i],
                "up_t_to":     d["videos/observation.images.up/to_timestamp"][i],
                "side_chunk":  d["videos/observation.images.side/chunk_index"][i],
                "side_file":   d["videos/observation.images.side/file_index"][i],
                "side_t_from": d["videos/observation.images.side/from_timestamp"][i],
                "side_t_to":   d["videos/observation.images.side/to_timestamp"][i],
            })
    all_eps.sort(key=lambda x: x["ep_idx"])
    return all_eps


def extract_clip(ep_info, cam, tmp_dir):
    tag = cam.split(".")[-1]
    chunk  = ep_info[f"{tag}_chunk"]
    file_i = ep_info[f"{tag}_file"]
    t_from = ep_info[f"{tag}_t_from"]
    t_to   = ep_info[f"{tag}_t_to"]
    src = os.path.join(DATASET_ROOT, f"videos/{cam}/chunk-{chunk:03d}/file-{file_i:03d}.mp4")
    out = os.path.join(tmp_dir, f"ep_{ep_info['ep_idx']:03d}_{tag}.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(t_from), "-i", src,
        "-t", str(t_to - t_from),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        out
    ], check=True)
    return out


def play_episode(clip_up, clip_side, ep_idx, total, to_delete):
    """Play video on the main thread; wait for terminal input on a worker thread."""
    import queue as q_module

    cap_up   = cv2.VideoCapture(clip_up)
    cap_side = cv2.VideoCapture(clip_side)
    fps   = cap_up.get(cv2.CAP_PROP_FPS) or 30
    delay = max(1, int(1000 / fps))

    cmd_queue = q_module.Queue()

    def input_worker():
        print(f"\n  EP {ep_idx:03d}/{total-1:03d}  {'*MARKED FOR DELETION*' if ep_idx in to_delete else ''}")
        print("  [Enter] next  [b] back  [d] mark delete  [r] replay  [q] quit > ", end="", flush=True)
        try:
            cmd = input().strip().lower()
        except EOFError:
            cmd = "q"
        cmd_queue.put(cmd)

    input_thread = threading.Thread(target=input_worker, daemon=True)
    input_thread.start()

    last_frame = None
    video_ended = False
    cmd = None

    # imshow + waitKey on the main thread
    while True:
        # stop immediately once a command arrives
        try:
            cmd = cmd_queue.get_nowait()
            break
        except Exception:
            pass

        if not video_ended:
            ret1, f_up   = cap_up.read()
            ret2, f_side = cap_side.read()
            if ret1 and ret2:
                h = min(f_up.shape[0], f_side.shape[0], 480)
                w = int(f_up.shape[1] * h / f_up.shape[0])
                f_up   = cv2.resize(f_up,   (w, h))
                f_side = cv2.resize(f_side, (w, h))
                cv2.putText(f_up,   "UP",   (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(f_side, "SIDE", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                last_frame = np.hstack([f_up, f_side])
            else:
                video_ended = True

        if last_frame is not None:
            frame = last_frame.copy()
            draw_overlay(frame, ep_idx, total, to_delete, end=video_ended)
            cv2.imshow("Episode Viewer", frame)

        cv2.waitKey(delay)

    cap_up.release()
    cap_side.release()
    return cmd or ""


def draw_overlay(frame, ep_idx, total, to_delete, end=False):
    h, w = frame.shape[:2]
    is_del = ep_idx in to_delete
    color = (0, 0, 255) if is_del else (0, 255, 100)
    status = "  * MARKED FOR DELETION *" if is_del else ""
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"EP {ep_idx:03d} / {total-1:03d}{status}",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    del_list = sorted(to_delete)
    if del_list:
        txt = "delete: " + ", ".join(str(x) for x in del_list[:20])
        cv2.putText(frame, txt, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1)
    if end:
        cv2.rectangle(frame, (0, h-30), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "<<END>> enter a command in the terminal",
                    (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)


def main():
    all_eps = load_all_episodes()
    total = len(all_eps)
    print(f"\n{total} episodes total")
    print("Commands: [Enter] next  [b] back  [d] toggle delete mark  [r] replay  [q] quit\n")

    tmp_dir = tempfile.mkdtemp()
    clip_cache = {}

    def get_clips(ep):
        idx = ep["ep_idx"]
        if idx not in clip_cache:
            print(f"  extracting clip: EP {idx:03d}...", flush=True)
            cup   = extract_clip(ep, "observation.images.up",   tmp_dir)
            cside = extract_clip(ep, "observation.images.side", tmp_dir)
            clip_cache[idx] = (cup, cside)
        return clip_cache[idx]

    to_delete = set()
    i = 0

    # pre-extract the first episode
    get_clips(all_eps[0])

    while 0 <= i < total:
        ep = all_eps[i]
        clip_up, clip_side = get_clips(ep)

        # extract the next episode in the background
        if i + 1 < total:
            next_ep = all_eps[i + 1]
            if next_ep["ep_idx"] not in clip_cache:
                def prefetch(e=next_ep):
                    get_clips(e)
                threading.Thread(target=prefetch, daemon=True).start()

        cmd = play_episode(clip_up, clip_side, ep["ep_idx"], total, to_delete)

        if cmd == "" or cmd == "n":
            i += 1
        elif cmd == "b":
            i = max(0, i - 1)
        elif cmd == "d":
            ep_idx = ep["ep_idx"]
            if ep_idx in to_delete:
                to_delete.discard(ep_idx)
                print(f"  -> unmarked: EP {ep_idx:03d}  marked for deletion: {sorted(to_delete)}")
            else:
                to_delete.add(ep_idx)
                print(f"  -> marked: EP {ep_idx:03d}  marked for deletion: {sorted(to_delete)}")
        elif cmd == "r":
            pass  # replay the same episode
        elif cmd == "q":
            break

    cv2.destroyAllWindows()
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 50)
    if to_delete:
        print(f"marked for deletion: {sorted(to_delete)}")
        confirm = input("Really delete? (y/N): ").strip().lower()
        if confirm == "y":
            delete_episodes(sorted(to_delete))
        else:
            print("Cancelled.")
    else:
        print("No episodes to delete.")


def delete_episodes(ep_indices_to_delete):
    import shutil
    all_ep_files = sorted(glob.glob(
        os.path.join(DATASET_ROOT, "meta/episodes/**/*.parquet"), recursive=True
    ))
    all_eps = load_all_episodes()
    del_set = set(ep_indices_to_delete)
    keep_eps = [ep for ep in all_eps if ep["ep_idx"] not in del_set]
    new_ep_map = {ep["ep_idx"]: new_i for new_i, ep in enumerate(keep_eps)}

    print(f"\ndeleting: {sorted(del_set)}")
    print(f"{len(all_eps)} -> {len(keep_eps)} episodes")

    tmp_dir = tempfile.mkdtemp()

    for cam in ["observation.images.up", "observation.images.side"]:
        tag = cam.split(".")[-1]
        vid_dir = os.path.join(DATASET_ROOT, f"videos/{cam}/chunk-000")
        clips = []
        for ep in keep_eps:
            src = os.path.join(vid_dir, f"file-{ep[tag+'_file']:03d}.mp4")
            clip = os.path.join(tmp_dir, f"{tag}_{ep['ep_idx']:03d}.mp4")
            t_from = ep[f"{tag}_t_from"]
            dur    = ep[f"{tag}_t_to"] - t_from
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", str(t_from), "-i", src,
                "-t", str(dur), "-c:v", "libx264", "-preset", "ultrafast",
                clip
            ], check=True)
            clips.append(clip)
        for f in glob.glob(os.path.join(vid_dir, "*.mp4")):
            os.remove(f)
        list_f = os.path.join(tmp_dir, f"list_{tag}.txt")
        with open(list_f, "w") as f:
            for p in clips:
                f.write(f"file '{p}'\n")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", list_f,
            "-c", "copy", os.path.join(vid_dir, "file-000.mp4")
        ], check=True)
        print(f"  rebuilding video: {cam}")

    # data parquet
    data_files = sorted(glob.glob(os.path.join(DATASET_ROOT, "data/**/*.parquet"), recursive=True))
    all_rows = []
    for df in data_files:
        t = pq.read_table(df)
        d = t.to_pydict()
        for i in range(len(d["episode_index"])):
            if d["episode_index"][i] not in del_set:
                row = {k: d[k][i] for k in d}
                row["episode_index"] = new_ep_map[row["episode_index"]]
                all_rows.append(row)
    for df in data_files:
        os.remove(df)
    if all_rows:
        keys = list(all_rows[0].keys())
        pq.write_table(
            pa.table({k: pa.array([r[k] for r in all_rows]) for k in keys}),
            os.path.join(DATASET_ROOT, "data/chunk-000/file-000.parquet")
        )
    print(f"  data parquet: {len(all_rows)} rows")

    # episodes parquet
    all_ep_rows = []
    cum_ts = {"observation.images.up": 0.0, "observation.images.side": 0.0}
    for ef in all_ep_files:
        t = pq.read_table(ef)
        d = t.to_pydict()
        for i in range(len(d["episode_index"])):
            if d["episode_index"][i] not in del_set:
                row = {k: d[k][i] for k in d}
                new_i = new_ep_map[row["episode_index"]]
                row["episode_index"] = new_i
                for cam_key in ["observation.images.up", "observation.images.side"]:
                    dur = (row[f"videos/{cam_key}/to_timestamp"] -
                           row[f"videos/{cam_key}/from_timestamp"])
                    row[f"videos/{cam_key}/chunk_index"] = 0
                    row[f"videos/{cam_key}/file_index"]  = 0
                    row[f"videos/{cam_key}/from_timestamp"] = cum_ts[cam_key]
                    row[f"videos/{cam_key}/to_timestamp"]   = cum_ts[cam_key] + dur
                    cum_ts[cam_key] += dur
                all_ep_rows.append(row)
    for ef in all_ep_files:
        os.remove(ef)
    keys = list(all_ep_rows[0].keys())
    pq.write_table(
        pa.table({k: pa.array([r[k] for r in all_ep_rows]) for k in keys}),
        os.path.join(DATASET_ROOT, "meta/episodes/chunk-000/file-000.parquet")
    )
    print(f"  episodes parquet: {len(all_ep_rows)} episodes")

    info_path = os.path.join(DATASET_ROOT, "meta/info.json")
    with open(info_path) as f:
        info = json.load(f)
    info["total_episodes"] = len(keep_eps)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    shutil.rmtree(tmp_dir)
    print(f"\nDone. {len(keep_eps)} episodes total")


if __name__ == "__main__":
    main()
