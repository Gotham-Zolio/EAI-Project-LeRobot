import re
from pathlib import Path

# URDF 路径
URDF_PATH = Path(r'/home/gotham/EAI-Project-LeRobot/assets/SO101/so101.urdf')
# mesh 路径修正为 assets/xxx.stl（相对 SO101 目录）
MESH_PREFIX = 'assets/'

# 匹配 mesh 路径
MESH_PATTERN = re.compile(r'(filename\s*=\s*["\"])(assets/assets/)(.*?)(["\"])')

def fix_mesh_paths(urdf_path):
    with open(urdf_path, encoding='utf-8') as f:
        content = f.read()
    # 替换 assets/assets/ 为 assets/
    new_content = MESH_PATTERN.sub(r'\1' + MESH_PREFIX + r'\3\4', content)
    # 备份原文件
    backup_path = urdf_path.with_suffix('.urdf.bak')
    urdf_path.rename(backup_path)
    with open(urdf_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f'已修正 mesh 路径，并备份原文件为: {backup_path}')

def main():
    fix_mesh_paths(URDF_PATH)

if __name__ == '__main__':
    main()
