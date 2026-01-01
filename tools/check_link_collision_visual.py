import re
from pathlib import Path

# URDF 路径
URDF_PATH = Path(r'd:/75128/Desktop/EAI-Project-LeRobot/assets/SO101/so101.urdf')

# 匹配 link 节点
LINK_PATTERN = re.compile(r'<link name="([^"]+)">(.*?)</link>', re.DOTALL)
# 匹配 collision/visual 节点，且 mesh 为 stl/obj
MESH_PATTERN = re.compile(r'<(collision|visual)[^>]*>.*?<mesh filename="[^"]+\.(stl|obj)"[^>]*/>.*?</\1>', re.DOTALL)

def check_links(urdf_path):
    with open(urdf_path, encoding='utf-8') as f:
        content = f.read()
    missing = []
    for link_match in LINK_PATTERN.finditer(content):
        link_name = link_match.group(1)
        link_body = link_match.group(2)
        mesh_found = MESH_PATTERN.search(link_body)
        if not mesh_found:
            missing.append(link_name)
    return missing

def main():
    missing = check_links(URDF_PATH)
    if missing:
        print('以下 link 节点缺少 stl/obj 格式的 collision/visual：')
        for name in missing:
            print(name)
    else:
        print('所有 link 节点都包含 stl/obj 格式的 collision/visual！')

if __name__ == '__main__':
    main()
