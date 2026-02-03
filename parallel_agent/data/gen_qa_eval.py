import os
import requests
import sys

def download_hotpotqa():
    files = {
        "hotpotqa_dev.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        "hotpotqa_train.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
    }

    chunk_size = 1024 * 1024  # 1 MB

    for filename, url in files.items():
        if os.path.exists(filename):
            print(f"[OK] {filename} 已存在，跳过下载")
            continue

        print(f"[DOWNLOAD] {filename}")

        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("Content-Length", 0))
            downloaded = 0

            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        percent = downloaded * 100 / total_size
                        bar_len = 30
                        filled = int(bar_len * percent / 100)
                        bar = "=" * filled + "-" * (bar_len - filled)
                        sys.stdout.write(
                            f"\r  [{bar}] {percent:6.2f}% ({downloaded/1024/1024:.1f} MB)"
                        )
                        sys.stdout.flush()

        print("\n[DONE]", filename)


if __name__ == "__main__":
    download_hotpotqa()