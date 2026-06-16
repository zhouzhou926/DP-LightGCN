
import requests
import os

proxies = {"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"}
headers = {"User-Agent": "Mozilla/5.0"}
base = "D:/workspace/DP-LightGCN/data"

urls_to_try = [
    # Source: RecGurus (maintained recommendation data)
    ("yelp2018", "https://raw.githubusercontent.com/RecGurus/recsys-datasets/main/Yelp2018/train.txt"),
    ("yelp2018", "https://raw.githubusercontent.com/RecGurus/recsys-datasets/main/Yelp2018/test.txt"),
    # Source: Another backup
    ("yelp2018", "https://raw.githubusercontent.com/xuweijieshuai/GraphRec/master/data/yelp2018/train.txt"),
    ("yelp2018", "https://raw.githubusercontent.com/xuweijieshuai/GraphRec/master/data/yelp2018/test.txt"),
    # Source: From reimplementations
    ("yelp2018", "https://raw.githubusercontent.com/ma765436910/LightGCN-pytorch/master/data/yelp2018/train.txt"),
    ("yelp2018", "https://raw.githubusercontent.com/ma765436910/LightGCN-pytorch/master/data/yelp2018/test.txt"),
]

for dataset, url in urls_to_try:
    fname = url.split("/")[-1]
    save_dir = os.path.join(base, dataset)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, fname)
    
    if os.path.exists(save_path) and os.path.getsize(save_path) > 100000:
        print(f"[SKIP] {dataset}/{fname} already exists")
        continue
    
    try:
        r = requests.get(url, proxies=proxies, headers=headers, timeout=30)
        if r.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(r.content)
            print(f"[OK] {dataset}/{fname} ({os.path.getsize(save_path)/1024/1024:.1f} MB)")
        else:
            print(f"[{r.status_code}] {url[:80]}")
    except Exception as e:
        print(f"[ERR] {url[:80]} -> {str(e)[:40]}")
