
import random
import numpy as np
from scipy.spatial import Delaunay
import networkx as nx
import matplotlib.pyplot as plt

def generate_planar_graph(
    n,
    target_m=None,       # nếu None -> dùng tất cả cạnh Delaunay
    coord_range=1000,
    p_ranges=( (4,20), (15,400), (15,100) ),  # p1,p2,p3 ranges
    p_matrix_range=(5,20),  # p cho dòng p p 0.05 0.05 0.05 0.05
    seed=None,
    visualize=True,
    jitter_eps=1e-8
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if n <= 0:
        raise ValueError("n must be positive")

    # 1) Sinh điểm (float) trong [0, coord_range]
    pts = np.random.uniform(0.0, coord_range, size=(n,2))

    # Nếu có điểm hoàn toàn trùng nhau (hiếm), jitter nhẹ để tránh lỗi Delaunay
    # (thường không cần, nhưng an toàn)
    # Check duplicates
    _, idx_counts = np.unique(np.round(pts, 8), axis=0, return_counts=True)
    if np.any(idx_counts > 1):
        pts += np.random.normal(scale=jitter_eps, size=pts.shape)

    # Trường hợp nhỏ n < 3 xử lý riêng
    if n < 3:
        edges = []
        if n == 2:
            edges = [(0,1)]
        # attributes
        params = [(
            random.randint(*p_ranges[0]),
            random.randint(*p_ranges[1]),
            random.randint(*p_ranges[2])
        ) for _ in range(n)]
        # build output
        return _format_and_visualize(pts, params, edges, p_matrix_range, visualize)

    # 2) Delaunay triangulation
    try:
        tri = Delaunay(pts)
    except Exception as e:
        # nếu Delaunay lỗi (ví dụ collinear), jitter nhẹ và retry
        pts += np.random.normal(scale=1e-6, size=pts.shape)
        tri = Delaunay(pts)

    # 3) Lấy unique undirected edges
    edge_set = set()
    for simplex in tri.simplices:
        for i in range(3):
            a = int(simplex[i])
            b = int(simplex[(i+1)%3])
            if a == b:
                continue
            if a > b:
                a,b = b,a
            edge_set.add((a,b))
    edges_tri = list(edge_set)  # all Delaunay edges; planar by construction

    # 4) Nếu không cần giảm cạnh -> dùng tất cả
    if target_m is None or target_m >= len(edges_tri):
        chosen_edges = edges_tri
    else:
        # kiểm tra target_m hợp lệ
        max_possible = len(edges_tri)  # <= 3n - 6
        if target_m < n-1:
            # không thể có đồ thị liên thông với < n-1 cạnh
            raise ValueError(f"target_m too small for connected graph; need at least {n-1}")
        if target_m > max_possible:
            raise ValueError(f"target_m too large: max planar edges from Delaunay = {max_possible}")

        # 4a) Tạo đồ thị chỉ với các cạnh Delaunay và trọng số là khoảng cách
        G = nx.Graph()
        for (u,v) in edges_tri:
            dist = np.linalg.norm(pts[u] - pts[v])
            G.add_edge(u, v, weight=dist)

        # 4b) Tính MST (đảm bảo connected)
        T = nx.minimum_spanning_tree(G, weight='weight')
        chosen = set(tuple(sorted(e)) for e in T.edges())

        # 4c) Thêm dần các cạnh nhỏ nhất từ các cạnh còn lại cho tới khi đạt target_m
        remaining = []
        for (u,v) in edges_tri:
            key = (u,v)
            if key in chosen:
                continue
            dist = np.linalg.norm(pts[u] - pts[v])
            remaining.append((dist,u,v))
        remaining.sort()  # theo khoảng cách tăng dần (có thể random thay vì sort)
        random.shuffle(remaining)  # để tránh bias trong các cạnh cùng khoảng cách
        for dist,u,v in remaining:
            if len(chosen) >= target_m:
                break
            chosen.add((u,v))
        chosen_edges = list(chosen)

    # 5) Sinh tham số p1,p2,p3 cho mỗi node
    params = [(
        random.randint(*p_ranges[0]),
        random.randint(*p_ranges[1]),
        random.randint(*p_ranges[2])
    ) for _ in range(n)]

    # 6) Trả về kết quả đã format và optionally visualize
    return _format_and_visualize(pts, params, chosen_edges, p_matrix_range, visualize)

def _format_and_visualize(pts, params, edges, p_matrix_range, visualize):
    n = len(pts)
    # edges is list of (u,v) sorted pairs
    # 1) choose p
    p = random.randint(*p_matrix_range)
    p = 20  # for consistent testing

    # Build textual output lines (you can save to file)
    lines = []
    lines.append(str(n))
    for i in range(n):
        x,y = pts[i]
        p1,p2,p3 = params[i]
        # format: id x y p1 p2 p3 0
        lines.append(f"{i} {x:.6f} {y:.6f} {p1} {p2} {p3} 0")
    m = len(edges)
    lines.append(str(m))
    for (u,v) in edges:
        lines.append(f"{u} {v}")
    lines.append(f"{p} {p} 0.05 0.05 0.05 0.05")
    # matrix n x p of ones
    for _ in range(p):
        lines.append(" ".join(["1"]*n))

    # Visualization
    if visualize:
        plt.figure(figsize=(8,8))
        # edges
        for u,v in edges:
            x1,y1 = pts[u]
            x2,y2 = pts[v]
            plt.plot([x1,x2],[y1,y2], '-', linewidth=0.8, zorder=1)
        # nodes
        xs = pts[:,0]
        ys = pts[:,1]
        # color/size by first param p1 to visualize attribute
        p1_vals = np.array([pp[0] for pp in params])
        sz = 30 + (p1_vals - p1_vals.min())/(p1_vals.ptp()+1e-9) * 70
        plt.scatter(xs, ys, s=sz, zorder=2)
        for i,(x,y) in enumerate(pts):
            plt.text(x+coord_offset(pts), y+coord_offset(pts), str(i), fontsize=8)
        plt.title(f"Planar graph: n={n}, m={m}")
        plt.axis('equal')
        plt.show()

    # return whole text and parsed items if caller wants
    return {
        "text_lines": lines,
        "points": pts,
        "params": params,
        "edges": edges,
        "p": p
    }

def coord_offset(pts):
    # small offset for labels relative to bounding box size
    rng = np.ptp(pts, axis=0)  # range in x,y
    return max(rng[0], rng[1]) * 0.01 if rng.size>0 else 1.0

# Example usage
if __name__ == "__main__":
    # generate connected planar graph with n=50 and target_m=100
    out = generate_planar_graph(n=100, target_m=150, seed=None, visualize=True)
    # to print to stdout in your requested format:
    print("\n".join(out["text_lines"]))
