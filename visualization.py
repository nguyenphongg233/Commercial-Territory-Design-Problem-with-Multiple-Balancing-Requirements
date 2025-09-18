import os
import matplotlib.pyplot as plt

def read_graph(filename):
    with open(filename, encoding='utf-8') as f:
        n = int(f.readline())
        nodes = []
        for _ in range(n):
            parts = f.readline().split()
            node_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            nodes.append((node_id, x, y))
        edges = []
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) == 2:
                    u, v = map(int, parts)
                    edges.append((u, v))
    return nodes, edges

def read_districts_custom(filename):
    """
    Đọc file output theo định dạng District X ... Nodes: ...,
    trả về dict: node_id -> district_id
    """
    node_to_district = {}
    with open(filename, encoding='utf-8') as f:
        district_id = None
        for line in f:
            line = line.strip()
            if line.startswith("District"):
                # Lấy số district
                district_id = int(line.split()[1])
            elif line.startswith("Nodes:"):
                # Lấy danh sách node
                node_ids = [int(x) for x in line[6:].split()]
                for node_id in node_ids:
                    node_to_district[node_id] = district_id
    return node_to_district

def plot_graph(nodes, edges):
    xs = [x for _, x, _ in nodes]
    ys = [y for _, _, y in nodes]
    plt.figure(figsize=(8,8))
    # Draw edges
    node_pos = {node_id: (x, y) for node_id, x, y in nodes}
    for u, v in edges:
        x1, y1 = node_pos[u]
        x2, y2 = node_pos[v]
        plt.plot([x1, x2], [y1, y2], color='gray', linewidth=1)
    # Draw nodes
    plt.scatter(xs, ys, color='blue', s=40, zorder=3)
    for node_id, x, y in nodes:
        plt.text(x, y, str(node_id), fontsize=8, ha='center', va='center', color='black')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Graph Visualization')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def plot_graph_colored(nodes, edges, node_to_district, imgname=None, show_popup=True):
    import matplotlib.cm as cm
    import numpy as np
    xs = [x for _, x, _ in nodes]
    ys = [y for _, _, y in nodes]
    plt.figure(figsize=(8,8))
    node_pos = {node_id: (x, y) for node_id, x, y in nodes}
    districts = sorted(set(node_to_district.values()))
    colors = cm.get_cmap('tab20', len(districts))
    district_color = {d: colors(i) for i, d in enumerate(districts)}
    for u, v in edges:
        x1, y1 = node_pos[u]
        x2, y2 = node_pos[v]
        plt.plot([x1, x2], [y1, y2], color='gray', linewidth=1, zorder=1)
    for node_id, x, y in nodes:
        d = node_to_district.get(node_id, -1)
        color = district_color.get(d, 'black')
        plt.scatter(x, y, color=color, s=40, zorder=3)
        plt.text(x, y, str(node_id), fontsize=8, ha='center', va='center', color='black')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Graph Visualization by District')
    plt.axis('equal')
    plt.tight_layout()
    if imgname:
        plt.savefig(imgname, dpi=300)
    if show_popup:
        plt.show()

if __name__ == "__main__":
    s = input()
    filename = "input_edited/" + s + ".dat"
    outname = "output_edited/" + s + ".out"
    # Tạo folder image nếu chưa có
    os.makedirs("image", exist_ok=True)
    imgname = os.path.join("image", s + ".png")
    nodes, edges = read_graph(filename)
    node_to_district = read_districts_custom(outname)
    # Đặt show_popup=True để hiện hình, False để không hiện
    plot_graph_colored(nodes, edges, node_to_district, imgname, show_popup=False)