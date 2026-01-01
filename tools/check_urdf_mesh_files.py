import os
import re
from pathlib import Path

# URDF 路径
URDF_PATH = Path(r'/home/gotham/EAI-Project-LeRobot/assets/SO101/so101.urdf')
# mesh 文件根目录（与 urdf 文件同级 assets 文件夹）
MESH_ROOT = URDF_PATH.parent / 'assets'

# 支持的 mesh 文件扩展名
MESH_EXTS = {'.stl', '.ply', '.obj'}

# 正则匹配 mesh 路径
MESH_PATTERN = re.compile(r'filename\s*=\s*["\"](.*?)["\"][^>]*')

def find_mesh_files(urdf_path):
    missing = []
    with open(urdf_path, encoding='utf-8') as f:
        for line in f:
            match = MESH_PATTERN.search(line)
            if match:
                mesh_rel_path = match.group(1)
                # 修正：如果 mesh_rel_path 已经包含 assets/ 前缀，则直接拼接 SO101 目录
                mesh_file = (urdf_path.parent / mesh_rel_path).resolve()
                if not mesh_file.exists():
                    missing.append(str(mesh_file))
    return missing

def main():
    missing = find_mesh_files(URDF_PATH)
    if missing:
        print('缺失的 mesh 文件:')
        for f in missing:
            print(f)
    else:
        print('所有 mesh 文件都存在！')

if __name__ == '__main__':
    main()
