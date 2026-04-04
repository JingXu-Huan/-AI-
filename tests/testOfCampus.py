def main():
    from pathlib import Path
    from main import main as entry_main

    folder = Path(r"D:\WechatFiles\xwechat_files\wxid_lsli6i1kinl922_fab6\msg\file\2026-04\地面缺陷图片")
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_files = [str(p) for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in image_exts]

    # main.py 入口一次只接收一个 IMAGE 参数，这里逐张调用
    for image_path in all_files:
        entry_main([image_path])
if __name__ == "__main__":
    main()