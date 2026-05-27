"""
07_download_pretrained.py
==========================
Скачивает pretrained Kronos-Tokenizer-base в локальную папку.
Нужно для варианта B: файнтюна только предиктора с оригинальным токенайзером.

Запуск:
    python 07_download_pretrained.py
"""

import os
from huggingface_hub import snapshot_download

OUTPUT_DIR = "./pretrained/Kronos-Tokenizer-base"

print(f"Скачиваю NeoQuasar/Kronos-Tokenizer-base в {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

path = snapshot_download(
    repo_id="NeoQuasar/Kronos-Tokenizer-base",
    local_dir=OUTPUT_DIR,
)

print(f"\nГотово. Папка: {os.path.abspath(OUTPUT_DIR)}")
print(f"Содержимое:")
for f in os.listdir(OUTPUT_DIR):
    full = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(full):
        size_mb = os.path.getsize(full) / 1024**2
        print(f"  {f}  ({size_mb:.1f} MB)")