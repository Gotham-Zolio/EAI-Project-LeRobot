import re
from pathlib import Path

# URDF 路径
URDF_PATH = Path(r'/home/gotham/EAI-Project-LeRobot/assets/SO101/so101.urdf')

# 匹配 <collision> 或 <visual> 节点中包含 ply mesh 的整段
PLY_BLOCK_PATTERN = re.compile(r'(<(collision|visual)[^>]*>\s*<origin[^>]*>.*?<geometry>\s*<mesh filename="[^"]+\.ply"[^>]*/>\s*</geometry>.*?</\2>)', re.DOTALL)

def comment_ply_blocks(urdf_path):
    with open(urdf_path, encoding='utf-8') as f:
        content = f.read()
    # 用 XML 注释包裹所有 ply mesh 节点
    new_content = PLY_BLOCK_PATTERN.sub(r'<!-- \1 -->', content)
    # 备份原文件
    backup_path = urdf_path.with_suffix('.urdf.plybak')
    urdf_path.rename(backup_path)
    with open(urdf_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f'已注释所有 ply mesh 节点，并备份原文件为: {backup_path}')

def main():
    comment_ply_blocks(URDF_PATH)

if __name__ == '__main__':
    main()
