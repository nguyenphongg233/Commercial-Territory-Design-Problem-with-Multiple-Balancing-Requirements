"""
Adaptive Large Neighborhood Search (ALNS) for Commercial Territory Design
Problem with multiple balancing requirements (connectivity enforced at all
states, three-level objective lexicographic: balance-violation, affinity-loss,
compactness).

This implementation:
- Reads input from a .txt file formatted as described by the user.
- Builds an initial feasible solution (connected districts) using a seeded
  breadth-first region growing approach (choosing m seeds then growing).
- Implements 4 destroy operators and 4 repair operators. Destroy operators
  remove only nodes that keep the donor district connected (leaf-based removals)
  or perform a district-group destruction that keeps only district centers.
- Uses adaptive operator selection: operator scores are updated based on success.
- Uses a simulated-annealing acceptance criterion on an aggregated weighted-sum
  objective (so worse solutions can be accepted); the comparator used for
  recording best solution is lexicographic (balance_violation, affinity_loss,
  compactness) as required.
- Ensures that each district center chosen is a node i with f_ij == 1 when
  possible. If a district has no node with f_ij==1, it falls back to the
  previous best-distance criterion.
- Adds a repair operator implementing 2-regret (assign nodes by maximizing the
  difference between best insertion cost and second-best insertion cost).

"""

import sys
import math
import random
import time
from collections import deque, defaultdict, Counter

# ------------------------------ Utilities ------------------------------

def read_instance(path):
    with open(path, 'r') as f:
        parts = f.read().strip().split()
    it = iter(parts)
    n = int(next(it))
    nodes = []  # list of dicts: {'id':id, 'x':x,'y':y,'w':[w1,w2,w3]}
    id_to_index = {}
    for idx in range(n):
        nid = next(it)
        x = float(next(it))
        y = float(next(it))
        w1 = float(next(it))
        w2 = float(next(it))
        w3 = float(next(it))
        _z = next(it)  # last zero or unused
        nodes.append({'id': nid, 'x': x, 'y': y, 'w': [w1, w2, w3]})
        id_to_index[nid] = idx
    k = int(next(it))
    edges = []
    adj = [[] for _ in range(n)]
    for _ in range(k):
        a = next(it)
        b = next(it)
        ia = id_to_index[a]
        ib = id_to_index[b]
        edges.append((ia, ib))
        adj[ia].append(ib)
        adj[ib].append(ia)
    # Next line: possibly repeated m twice according to user's spec
    try:
        m1 = int(next(it))
        m2 = int(next(it))
    except StopIteration:
        raise ValueError('Input ended unexpectedly when reading m')
    m = m1
    tau = [float(next(it)), float(next(it)), float(next(it))]
    _z = float(next(it))
    # read m lines each with n affinity values (1/2/3)
    f = [[0]*n for _ in range(m)]
    for j in range(m):
        for i in range(n):
            f[j][i] = int(next(it))
    return {
        'n': n,
        'nodes': nodes,
        'adj': adj,
        'm': m,
        'tau': tau,
        'f': f
    }

def euclidean(a, b):
    return math.hypot(a['x']-b['x'], a['y']-b['y'])
def compute_dist_matrix(nodes):
    n = len(nodes)
    d = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            dist = euclidean(nodes[i], nodes[j])
            d[i][j] = dist
            d[j][i] = dist
    return d

# ------------------------------ Feasibility & metrics ------------------------------

# connectivity on a set of nodes (indices) using given adjacency = O(n+m)

def is_connected_set(node_set, adj):
    if not node_set:
        return True
    start = next(iter(node_set))
    seen = set([start])
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v in node_set and v not in seen:
                seen.add(v)
                q.append(v)
    return len(seen) == len(node_set)

# compute leaf nodes in induced subgraph (degree==1 within induced) = O(m)

def leaf_nodes_of_set(node_set, adj):
    leaves = []
    node_set_lookup = set(node_set)
    for u in node_set:
        deg = 0
        for v in adj[u]:
            if v in node_set_lookup:
                deg += 1
        if deg <= 1:
            leaves.append(u)
    return leaves

# compute district metrics: sums of weights, compactness (sum distances to center), affinity loss = O(n)

def compute_district_metrics(district_nodes, center_idx, nodes, dmat):
    wsum = [0.0, 0.0, 0.0]
    compact = 0.0
    for u in district_nodes:
        for a in range(3):
            wsum[a] += nodes[u]['w'][a]
        if center_idx is not None:
            compact += dmat[u][center_idx]
    return wsum, compact

# compute global objective components = O(n)

def compute_objectives(partition, centers, instance, dmat):
    # partition: list of sets (length m)
    m = instance['m']
    nodes = instance['nodes']
    tau = instance['tau']
    n = instance['n']
    # total weights
    total_w = [0.0, 0.0, 0.0]
    for i in range(n):
        for a in range(3):
            total_w[a] += nodes[i]['w'][a]
    avg_w = [total_w[a]/m for a in range(3)]
    balance_violation = 0.0
    affinity_loss = 0
    compactness = 0.0
    for j in range(m):
        wsum, compact = compute_district_metrics(partition[j], centers[j], nodes, dmat)
        compactness += compact
        for a in range(3):
            lower = (1.0 - tau[a]) * avg_w[a]
            upper = (1.0 + tau[a]) * avg_w[a]
            balance_violation += max(0,lower - wsum[a],wsum[a] - upper)
        # affinity loss uses f matrix: f_ij - 1 summed
        for u in partition[j]:
            affinity_loss += (instance['f'][j][u] - 1)
    return balance_violation, affinity_loss, compactness

# lexicographic comparator: smaller tuple is better

def lex_better(sol1, sol2):
    return sol1 < sol2

# aggregated score for SA acceptance (weighted sum) - weights chosen to reflect priorities

def aggregated_score(objs, weights=(1e10, 1e5, 1.0)):
    # weights large for balance_violation to emulate lexicographic priority
    return weights[0]*objs[0] + weights[1]*objs[1] + weights[2]*objs[2]

# ------------------------------ Initial solution ------------------------------

"""
    Choose a center node for a district preferring nodes with f_{district, node} == 1.
    If no node with f==1 exists in the district, fall back to the distance-minimizing node.
    = O(n * n)
 """

def choose_center_for_district(dnodes, district_idx, instance, dmat):
    f = instance['f']
    nodes = instance['nodes']
    candidates = [u for u in dnodes if f[district_idx][u] == 1]
    if candidates:
        # from candidates choose the one minimizing sum distances to other nodes
        return min(candidates, key=lambda u: sum(dmat[u][v] for v in dnodes))
    # fallback
    return min(dnodes, key=lambda u: sum(dmat[u][v] for v in dnodes))


def initial_solution_seeded_growth(instance, dmat, time_limit=None):
    n = instance['n']
    m = instance['m']
    adj = instance['adj']
    nodes = instance['nodes']
    # pick m seed nodes (spread by farthest-first)
    seeds = []
    remaining = set(range(n))
    first = random.choice(list(remaining))
    seeds.append(first)
    remaining.remove(first)
    while len(seeds) < m:
        # choose node farthest from existing seeds (in terms of min distance)
        best = None
        best_dist = -1
        for u in list(remaining):
            mind = min(dmat[u][s] for s in seeds)
            if mind > best_dist:
                best_dist = mind
                best = u
        seeds.append(best)
        remaining.remove(best)
    # Kmeans++ initialization
    # region growing: each seed expands to neighbors greedily until all nodes assigned
    partition = [set([s]) for s in seeds]
    centers = [None]*m
    for j in range(m):
        centers[j] = choose_center_for_district(partition[j], j, instance, dmat)
    assigned = set(seeds)
    frontier = [set() for _ in range(m)]
    for j in range(m):
        for v in adj[seeds[j]]:
            if v not in assigned:
                frontier[j].add(v)
    
    while len(assigned) < n:
        # pick district with non-empty frontier, choose best candidate by distance to center
        changed = False
        order = list(range(m))
        random.shuffle(order)
        for j in order:
            if not frontier[j]:
                continue
            # evaluate candidates
            cand = min(list(frontier[j]), key=lambda u: dmat[u][centers[j]] if centers[j] is not None else 0.0)
            partition[j].add(cand)
            assigned.add(cand)
            changed = True
            # update frontier
            for v in adj[cand]:
                if v not in assigned:
                    frontier[j].add(v)
            frontier[j].discard(cand)
            # remove cand from other frontiers
            for t in range(m):
                if t != j:
                    frontier[t].discard(cand)
            if len(assigned) >= n:
                break
        if not changed:
            # if no frontier expansions possible (graph disconnected between regions),
            # assign remaining nodes arbitrarily to nearest seed while maintaining connectivity
            for u in range(n):
                if u in assigned:
                    continue
                # find neighbor in assigned set
                neighbors = [v for v in adj[u] if v in assigned]
                if neighbors:
                    # assign to the district of neighbor
                    v = neighbors[0]
                    for j in range(m):
                        if v in partition[j]:
                            partition[j].add(u)
                            assigned.add(u)
                            break
            # safety break to avoid infinite loop
            if not any(u not in assigned for u in range(n)):
                break
    # ensure each partition is connected; if not, repair by BFS splitting components and merging small components
    for j in range(m):
        comps = components_of_set(partition[j], adj)
        if len(comps) > 1:
            # keep largest in j, reassign others to neighboring districts
            comps_sorted = sorted(comps, key=lambda s: -len(s))
            partition[j] = comps_sorted[0]
            for comp in comps_sorted[1:]:
                # find neighboring district to merge into (any adjacent district)
                assigned_flag = False
                for u in comp:
                    for v in adj[u]:
                        for t in range(m):
                            if t != j and v in partition[t]:
                                partition[t].update(comp)
                                assigned_flag = True
                                break
                        if assigned_flag:
                            break
                    if assigned_flag:
                        break
                if not assigned_flag:
                    # assign to smallest partition by size
                    t = min(range(m), key=lambda x: len(partition[x]))
                    partition[t].update(comp)
    # recompute centers: pick one node in each district preferring f==1
    centers = []
    for j in range(m):
        centers.append(choose_center_for_district(partition[j], j, instance, dmat))
    # final check
    for j in range(m):
        if not is_connected_set(partition[j], adj):
            print("Warning: initial district", j, "not connected")
            exit(0)
    return partition, centers

# helper: components of induced subgraph

def components_of_set(node_set, adj):
    node_set = set(node_set)
    comps = []
    while node_set:
        start = next(iter(node_set))
        comp = set([start])
        q = deque([start])
        node_set.remove(start)
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v in node_set:
                    node_set.remove(v)
                    comp.add(v)
                    q.append(v)
        comps.append(comp)
    return comps

# ------------------------------ Destroy operators ------------------------------

# We must ensure destroying doesn't break connectivity of donor districts at any point.
# Strategy: only remove nodes that are leaves in their district induced graph. Removing leaves keeps
# the district connected. For removing multiple nodes, remove leaves iteratively.

# Destroy 1: Random leaf knock-out: remove up to k leaf nodes chosen randomly across districts

def destroy_random_leafs(centers,partition, adj, k):
    removed = []  # list of (node, from_district)
    m = len(partition)
    candidate_pairs = []
    for j in range(m):
        leaves = leaf_nodes_of_set(partition[j], adj)
        for u in leaves:
            if u != centers[j]:  # do not remove center
                candidate_pairs.append((j, u))
    random.shuffle(candidate_pairs)
    for (j, u) in candidate_pairs[:k]:
        partition[j].remove(u)
        removed.append((u, j))
    return removed

# Destroy 2: Path removal: pick a district, pick a short path of leaf-to-leaf nodes and remove them

def destroy_path(centers, partition, adj, k):
    #return []
    m = len(partition)
    j = random.randrange(m)
    # attempt to find a path inside partition[j]
    nodes = list(partition[j])
    if len(nodes) <= 1:
        return []
    # pick a start leaf, do BFS inside partition to find short path
    leaves = leaf_nodes_of_set(partition[j], adj)
    if not leaves:
        return []
    start = random.choice(leaves)
    parent = {start: None}
    q = deque([start])
    target = None
    while q and target is None:
        u = q.popleft()
        for v in adj[u]:
            if v in partition[j] and v not in parent:
                parent[v] = u
                q.append(v)
                if v in leaves and v != start:
                    target = v
                    break
        if target != None:
            break
    if target is None:
        return []
    # reconstruct path
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path = list(reversed(path))
    k = random.randint(1, min(k, len(path)))
    to_remove = path[:k]
    removed = []
    front, back = 0, len(path) - 1
    while front <= back and len(removed) < k:
        leaves_now = leaf_nodes_of_set(partition[j], adj)

        # ưu tiên xóa front
        if (path[front] in leaves_now) and (path[front] != centers[j]):
            u = path[front]
            partition[j].remove(u)
            removed.append((u, j))
            front += 1
            continue

        # nếu không xóa được front thì thử back
        if (path[back] in leaves_now) and (path[back] != centers[j]):
            u = path[back]
            partition[j].remove(u)
            removed.append((u, j))
            back -= 1
            continue

        # nếu cả front lẫn back đều không xóa được thì dừng
        break

    return removed

# Destroy 3: Worst-affinity removal: remove nodes with highest affinity loss (f_ij large) but only if removable

def destroy_worst_affinity(centers,partition, instance, adj, k):
    #return []
    m = len(partition)
    f = instance['f']
    candidate = []
    for j in range(m):
        leaves = leaf_nodes_of_set(partition[j], adj)
        for u in leaves:
            candidate.append((f[j][u], j, u))
    candidate.sort(reverse=True)  # worst affinity first (higher f)
    removed = []
    for item in candidate[:k]:
        _, j, u = item
        if u == centers[j]:
            continue
        partition[j].remove(u)
        removed.append((u, j))
    return removed

# Destroy 4: District-group destruction: remove entire 2- or 3- connected districts, keep only their centers.
# Choose groups dependent on their combined objective contribution (we pick the worst groups to disturb)

def destroy_district_group(partition, centers, instance, adj, dmat, group_size_choices=(2,3)):
    #return []
    m = len(partition)
    # build district adjacency graph
    district_adj = [set() for _ in range(m)]
    for u in range(instance['n']):
        for v in adj[u]:
            # find districts of u and v
            du = None
            dv = None
            for j in range(m):
                if u in partition[j]:
                    du = j
                if v in partition[j]:
                    dv = j
                if du is not None and dv is not None:
                    break
            if du is not None and dv is not None and du != dv:
                district_adj[du].add(dv)
                district_adj[dv].add(du)
    # enumerate connected groups of size 2 or 3 (small m, ok). We'll score each group by combined objective contribution.
    groups = []  # list of (score, [districts]) higher score = worse
    for size in group_size_choices:
        # simple BFS from each district to find connected sets of given size
        for start in range(m):
            q = deque([(start, [start])])
            seen_sets = set()
            while q:
                cur, path = q.popleft()
                if len(path) == size:
                    sset = tuple(sorted(path))
                    if sset not in seen_sets:
                        # compute group's objective contribution (balance violation + affinity + compactness)
                        # We'll use the sub-objective for those districts
                        sub_partition = [partition[j] for j in path]
                        sub_centers = [centers[j] for j in path]
                        score = 0.0
                        # reuse compute_objectives by building tiny instance? We'll approximate: compute balance and affinity and compactness locally
                        # compute weights sums and affinity/compactness
                        for idx, j in enumerate(path):
                            wsum, compact = compute_district_metrics(partition[j], centers[j], instance['nodes'], dmat)
                            # simple score: large balance violation and affinity loss yield high score
                            score += sum(wsum)  # crude proxy: larger weight -> higher impact
                            score += compact * 0.1
                            score += sum((instance['f'][j][u]-1) for u in partition[j]) * 0.5
                        groups.append((score, list(path)))
                        seen_sets.add(sset)
                    continue
                for neigh in district_adj[cur]:
                    if neigh not in path and len(path) < size:
                        q.append((neigh, path + [neigh]))
    if not groups:
        return []
    # choose the worst-scoring group (highest score) randomly among top few
    groups.sort(reverse=True)
    topk = min(5, len(groups))
    chosen = random.choice(groups[:topk])[1]
    removed = []
    # For each district in chosen group: remove all nodes except the center (keep center node only)
    for j in chosen:
        center_node = centers[j]
        to_remove_nodes = [u for u in list(partition[j]) if u != center_node]
        for u in to_remove_nodes:
            partition[j].remove(u)
            removed.append((u, j))
    return removed

# Thay thế hàm destroy_bridge_nodes bằng destroy_articulation_split
def destroy_articulation_split(partition, centers, adj, max_ratio=0.2):
    """
    Chọn các node khớp (articulation points) trong từng district mà khi xóa sẽ làm tăng số thành phần liên thông.
    Xóa ngẫu nhiên không quá max_ratio số node khớp.
    Sau đó chỉ giữ lại các node còn liên thông với center, các thành phần rời bị loại bỏ.
    """
    import networkx as nx
    import random

    removed = []
    m = len(partition)
    for j in range(m):
        G = nx.Graph()
        G.add_nodes_from(partition[j])
        for u in partition[j]:
            for v in adj[u]:
                if v in partition[j]:
                    G.add_edge(u, v)
        # Tìm articulation points
        bridges = list(nx.articulation_points(G))
        # Chỉ chọn những node mà khi xóa sẽ làm tăng số thành phần liên thông
        critical_nodes = []
        comps_before = [c for c in nx.connected_components(G)]
        n_comps_before = len(comps_before)
        for u in bridges:
            G2 = G.copy()
            G2.remove_node(u)
            n_comps_after = nx.number_connected_components(G2)
            if n_comps_after > n_comps_before:
                critical_nodes.append(u)
        # Xóa ngẫu nhiên không quá 20% số critical_nodes
        if critical_nodes:
            k = max(1, int(max_ratio * len(critical_nodes)))
            to_remove = random.sample(critical_nodes, min(k, len(critical_nodes)))
            for u in to_remove:
                partition[j].remove(u)
                removed.append((u, j))
        # Sau khi xóa, chỉ giữ lại các node còn liên thông với center
        if centers[j] in partition[j]:
            G3 = nx.Graph()
            G3.add_nodes_from(partition[j])
            for u in partition[j]:
                for v in adj[u]:
                    if v in partition[j]:
                        G3.add_edge(u, v)
            # Tìm thành phần liên thông chứa center
            for comp in nx.connected_components(G3):
                if centers[j] in comp:
                    keep_nodes = set(comp)
                    break
            remove_nodes = [u for u in partition[j] if u not in keep_nodes]
            for u in remove_nodes:
                partition[j].remove(u)
                removed.append((u, j))
    return removed

# ------------------------------ Repair operators ------------------------------

# Repairs must assign removed nodes while maintaining connectivity. We will only assign a node
# to a district if it has at least one neighbor in that district (so connectivity preserved).

# Repair 1: Greedy by feasibility: assign each unassigned node to the neighbor district that
# produces smallest increase in balance violation, tie-breaker: affinity and compactness

def repair_greedy_feasible(unassigned, partition, centers, instance, dmat):
    adj = instance['adj']
    m = instance['m']
    nodes = instance['nodes']
    tau = instance['tau']
    n = instance['n']
    # compute current sums for districts
    district_w = [ [0.0]*3 for _ in range(m) ]
    for j in range(m):
        for u in partition[j]:
            for a in range(3):
                district_w[j][a] += nodes[u]['w'][a]
    avg_w = [sum(nodes[i]['w'][a] for i in range(n))/m for a in range(3)]
    assigned_nodes = []
    # process nodes in arbitrary order (could be improved)
    for u in list(unassigned):
        # find candidate districts that are adjacent (so connectivity)
        cands = set()
        for v in adj[u]:
            for j in range(m):
                if v in partition[j]:
                    cands.add(j)
        if not cands:
            # isolated node (no neighbor assigned) - assign to smallest partition
            j = min(range(m), key=lambda x: len(partition[x]))
            partition[j].add(u)
            for a in range(3):
                district_w[j][a] += nodes[u]['w'][a]
            assigned_nodes.append((u,j))
            unassigned.remove(u)
            continue
        bestj = None
        best_increase = None
        for j in cands:
            # compute provisional weight sums
            inc = 0.0
            for a in range(3):
                neww = district_w[j][a] + nodes[u]['w'][a]
                lower = (1.0 - tau[a]) * avg_w[a]
                upper = (1.0 + tau[a]) * avg_w[a]
                if neww < lower:
                    inc += (lower - neww)
                elif neww > upper:
                    inc += (neww - upper)
            if best_increase is None or inc < best_increase:
                best_increase = inc
                bestj = j
        # tie-breaker by affinity to district
        partition[bestj].add(u)
        for a in range(3):
            district_w[bestj][a] += nodes[u]['w'][a]
        assigned_nodes.append((u,bestj))
        unassigned.remove(u)
    return assigned_nodes

# Repair 2: Greedy by affinity then compactness: assign nodes to districts where f_ij minimal (closest to 1),
# breaking ties by distance to center

def repair_affinity_compact(unassigned, partition, centers, instance, dmat):
    adj = instance['adj']
    m = instance['m']
    nodes = instance['nodes']
    f = instance['f']
    assigned_nodes = []

    # --- Sort lại unassigned bằng BFS đa nguồn từ các node biên ---
    q = []
    inqueue = set()
    ordered = []

    # khởi tạo queue với các node biên
    for u in list(unassigned):
        for v in adj[u]:
            for j in range(m):
                if v in partition[j]:
                    if u not in inqueue:
                        q.append(u)
                        inqueue.add(u)
                    break
            if u in inqueue:
                break

    #print("Initial queue length: ", len(q))
    #print("Unassigned length: ", len(unassigned))
    # BFS đa nguồn
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        if u in unassigned and u not in ordered:
            ordered.append(u)
            for v in adj[u]:
                if v in unassigned and v not in inqueue:
                    q.append(v)
                    inqueue.add(v)

    # nếu còn node isolated thì thêm cuối
    for u in unassigned:
        if u not in ordered:
           # print("Not legit\n")
            #exit(0)
            ordered.append(u)

    for u in ordered:
        if u not in unassigned:
            continue
        cands = set()
        for v in adj[u]:
            for j in range(m):
                if v in partition[j]:
                    cands.add(j)
        if not cands:
           # print("Isolated node encountered in affinity-compact repair")
            j = min(range(m), key=lambda x: len(partition[x]))
            partition[j].add(u)
            assigned_nodes.append((u,j))
            unassigned.remove(u)
            continue
        bestj = min(
            list(cands),
            key=lambda j: (
                f[j][u],
                dmat[u][centers[j]] if centers[j] is not None else 0.0
            )
        )
        partition[bestj].add(u)
        assigned_nodes.append((u,bestj))
        unassigned.remove(u)

    #print(len(unassigned))
    return assigned_nodes


# Repair 3: Assign to nearest center district (connectivity requires neighbor) - if no neighbor, attach to nearest center's district

def repair_nearest_center(unassigned, partition, centers, instance, dmat):
    adj = instance['adj']
    m = instance['m']
    assigned_nodes = []
    for u in list(unassigned):
        cands = set()
        for v in adj[u]:
            for j in range(m):
                if v in partition[j]:
                    cands.add(j)
        if not cands:
            # assign to nearest center
            bestj = min(range(m), key=lambda j: dmat[u][centers[j]] if centers[j] is not None else float('inf'))
            partition[bestj].add(u)
            assigned_nodes.append((u,bestj))
            unassigned.remove(u)
            continue
        bestj = min(list(cands), key=lambda j: dmat[u][centers[j]] if centers[j] is not None else 0.0)
        partition[bestj].add(u)
        assigned_nodes.append((u,bestj))
        unassigned.remove(u)
    return assigned_nodes

# Repair 4: 2-Regret repair: for each unassigned node compute best and second-best insertion cost (by increase in aggregated score)
# assign node with largest regret (diff between second_best_cost and best_cost), breaking ties by best_cost

def repair_2regret(unassigned, partition, centers, instance, dmat):
    adj = instance['adj']
    m = instance['m']
    nodes = instance['nodes']
    tau = instance['tau']
    n = instance['n']
    assigned_nodes = []
    # precompute district weight sums
    district_w = [ [0.0]*3 for _ in range(m) ]
    for j in range(m):
        for u in partition[j]:
            for a in range(3):
                district_w[j][a] += nodes[u]['w'][a]
    total_w = [0.0,0.0,0.0]
    for i in range(n):
        for a in range(3):
            total_w[a] += nodes[i]['w'][a]
    avg_w = [total_w[a]/m for a in range(3)]

    # helper to compute insertion cost (we use a composite cost: balance violation increase*big + affinity loss + compact increase)
    def insertion_cost(u, j):
        # only allow if u has neighbor in partition[j]
        if not any(v in partition[j] for v in adj[u]):
            return float('inf')
        # balance violation delta
        delta_balance = 0.0
        for a in range(3):
            old = district_w[j][a]
            new = old + nodes[u]['w'][a]
            lower = (1.0 - tau[a]) * avg_w[a]
            upper = (1.0 + tau[a]) * avg_w[a]
            old_violation = 0.0
            new_violation = 0.0
            if old < lower:
                old_violation = (lower - old)
            elif old > upper:
                old_violation = (old - upper)
            if new < lower:
                new_violation = (lower - new)
            elif new > upper:
                new_violation = (new - upper)
            delta_balance += (new_violation - old_violation)
        # affinity increase
        delta_aff = (instance['f'][j][u] - 1)
        # compact increase (distance to center)
        delta_compact = dmat[u][centers[j]] if centers[j] is not None else 0.0
        # large weight to balance
        return 1e6 * max(0.0, delta_balance) + 1e3 * delta_aff + delta_compact

    unassigned_list = list(unassigned)
    # iterative assignment: pick node with max regret
    while unassigned_list:
        best_regret = -float('inf')
        best_node = None
        best_choice = None
        best_cost = None
        for u in unassigned_list:
            costs = []
            for j in range(m):
                c = insertion_cost(u, j)
                costs.append((c, j))
            costs.sort()
            if costs[0][0] == float('inf'):
                # cannot be assigned feasibly based on neighbor rule; allow assignment to smallest partition (penalized)
                costs = [(float('inf'), None)]
            # compute regret = second_best - best
            first_cost = costs[0][0]
            second_cost = costs[1][0] if len(costs) > 1 else float('inf')
            regret = second_cost - first_cost
            # pick node with largest regret
            if regret > best_regret or (regret == best_regret and first_cost < (best_cost or float('inf'))):
                best_regret = regret
                best_node = u
                best_choice = costs[0][1]
                best_cost = first_cost
        # assign best_node
        if best_choice is None:
            # fallback: assign to smallest partition
            j = min(range(m), key=lambda x: len(partition[x]))
        else:
            j = best_choice
        partition[j].add(best_node)
        # update district_w
        for a in range(3):
            district_w[j][a] += nodes[best_node]['w'][a]
        assigned_nodes.append((best_node, j))
        unassigned_list.remove(best_node)
    # remove from original unassigned set
    for u, _ in assigned_nodes:
        if u in unassigned:
            unassigned.remove(u)
    return assigned_nodes

# ------------------------------ ALNS main loop ------------------------------


def alns(instance, iters=50000, time_limit=3000, stable_iters = 500, seed=42):
    #print(iters, time_limit, stable_iters, seed)
    random.seed(seed)
    n = instance['n']
    m = instance['m']
    adj = instance['adj']
    nodes = instance['nodes']
    dmat = compute_dist_matrix(nodes)

    # initial solution
    cur_partition, cur_centers = initial_solution_seeded_growth(instance, dmat)
    cur_objs = compute_objectives(cur_partition, cur_centers, instance, dmat)
    best_partition = [set(s) for s in cur_partition]
    best_centers = list(cur_centers)
    best_objs = cur_objs
    partition = [set(s) for s in cur_partition]
    centers = list(cur_centers)

    #return best_partition, best_centers, best_objs
    # operator pools (added district-group destroy and 2-regret repair)
    destroy_ops = [destroy_random_leafs, destroy_path, destroy_worst_affinity, destroy_district_group]
    repair_ops = [repair_greedy_feasible, repair_affinity_compact, repair_nearest_center, repair_2regret]
    # adaptive weights
    d_weights = [1.0]*len(destroy_ops)
    r_weights = [1.0]*len(repair_ops)
    d_scores = [0.0]*len(destroy_ops)
    r_scores = [0.0]*len(repair_ops)
    d_counts = [1e-6]*len(destroy_ops)
    r_counts = [1e-6]*len(repair_ops)

    T0 = aggregated_score(cur_objs)
    T = T0 if T0>0 else 1.0
    alpha = 0.9995

    start_time = time.time()
    no_improve = 0

    score = [0.1,1,5]  # scores for operator success levels

    for it in range(iters):
        if time_limit and time.time()-start_time > time_limit:
            break
        # select operators probabilistically
        d_idx = roulette_choice(d_weights)
        r_idx = roulette_choice(r_weights)

        # copy current partition
        partition = [set(s) for s in cur_partition]
        centers = list(cur_centers)

        # decide number of nodes to remove (k)
        k = max(1, int(0.02 * n))
        if it % 50 == 0:
            k = max(1, int(0.05 * n))
        # perform destroy (must track removed nodes)
        removed = []
        if destroy_ops[d_idx] is destroy_random_leafs:
            removed = destroy_random_leafs(centers,partition, adj, k)
        elif destroy_ops[d_idx] is destroy_path:
            removed = destroy_path(centers,partition, adj, k)
        elif destroy_ops[d_idx] is destroy_worst_affinity:
            removed = destroy_worst_affinity(centers,partition, instance, adj, k)
        elif destroy_ops[d_idx] is destroy_district_group:
            removed = destroy_district_group(partition, centers, instance, adj, dmat)
        # elif destroy_ops[d_idx] is destroy_articulation_split:
        #     removed = destroy_articulation_split(partition, centers, adj, max_ratio=0.2)
        unassigned = [u for (u,_) in removed]

        # ensure centers still valid; if a center was removed or no f==1 center exists, pick new center with f==1 if possible
        for j in range(m):
            # prefer selecting a center with f==1
            if partition[j]:
                preferred = [u for u in partition[j] if instance['f'][j][u] == 1]
                if preferred:
                    centers[j] = min(preferred, key=lambda u: sum(dmat[u][v] for v in partition[j]))
                else:
                    centers[j] = min(list(partition[j]), key=lambda u: sum(dmat[u][v] for v in partition[j]))
            else:
                centers[j] = None

        # repair using chosen repair operator
        if repair_ops[r_idx] is repair_greedy_feasible:
            assigned = repair_greedy_feasible(unassigned, partition, centers, instance, dmat)
        elif repair_ops[r_idx] is repair_affinity_compact:
            assigned = repair_affinity_compact(unassigned, partition, centers, instance, dmat)
        elif repair_ops[r_idx] is repair_nearest_center:
            assigned = repair_nearest_center(unassigned, partition, centers, instance, dmat)
        elif repair_ops[r_idx] is repair_2regret:
            assigned = repair_2regret(unassigned, partition, centers, instance, dmat)
        # elif repair_ops[r_idx] is repair_kmeans_manual:
        #     assigned = repair_kmeans_manual(unassigned, partition, centers, instance, dmat,m)

        # recompute centers: always prefer f==1 nodes
        for j in range(m):
            if partition[j]:
                preferred = [u for u in partition[j] if instance['f'][j][u] == 1]
                if preferred:
                    centers[j] = min(preferred, key=lambda u: sum(dmat[u][v] for v in partition[j]))
                else:
                    centers[j] = min(list(partition[j]), key=lambda u: sum(dmat[u][v] for v in partition[j]))
            else:
                centers[j] = None

        new_objs = compute_objectives(partition, centers, instance, dmat)
        # acceptance criterion: SA on aggregated score
        cur_score = aggregated_score(cur_objs)
        new_score = aggregated_score(new_objs)
        accept = False
        if new_score <= cur_score:
            accept = True
        else:
            prob = math.exp((cur_score - new_score) / max(1e-9, T))
            if random.random() < prob:
                accept = True

        #print(destroy_ops[d_idx].__name__, "\t", repair_ops[r_idx].__name__,end = "\t")
        for j in range(instance['m']):
            if not is_connected_set(cur_partition[j], instance['adj']):
                accept = False
                break

        if accept:
            #print(" Accepted", end = " ")
            cur_partition = [set(s) for s in partition]
            cur_centers = list(centers)
            cur_objs = new_objs
            # reward operators if improvement in lexicographic sense
            if lex_better(new_objs, best_objs):
                best_partition = [set(s) for s in partition]
                best_centers = list(centers)
                best_objs = new_objs
                d_scores[d_idx] += score[2]
                r_scores[r_idx] += score[2]
                #print(" BEST")
                no_improve = 0
            elif new_score < cur_score:
                d_scores[d_idx] += score[1]
                r_scores[r_idx] += score[1]
                #print(" Better")
            else :
                # small reward for accepted but no improvement
                d_scores[d_idx] += score[0]
                r_scores[r_idx] += score[0]
                #print(" Accepted")
        else:
            # small penalty for non-accepted
            d_scores[d_idx] += score[0]
            r_scores[r_idx] += score[0]
            #print(" BAD")
        

        d_counts[d_idx] += 1
        r_counts[r_idx] += 1
        # periodically update weights
        if it % 50 == 0 and it > 0:
            for i in range(len(d_weights)):
                d_weights[i] = (1 - 0.2) * d_weights[i] + 0.2 * (d_scores[i] / d_counts[i])
            for i in range(len(r_weights)):
                r_weights[i] = (1 - 0.2) * r_weights[i] + 0.2 * (r_scores[i] / r_counts[i])
        # cool down
        T *= alpha
        no_improve += 1
        if no_improve > stable_iters:
            cur_partition, cur_centers = initial_solution_seeded_growth(instance, dmat)
            cur_objs = compute_objectives(cur_partition, cur_centers, instance, dmat)
            no_improve = 0

    
    # for i in range(len(d_weights)):
    #     print(destroy_ops[i],d_weights[i], d_scores[i], d_counts[i])
    # print("-----")
    # for i in range(len(r_weights)):
    #     print(repair_ops[i],r_weights[i], r_scores[i], r_counts[i])
    return best_partition, best_centers, best_objs

# roulette selection helper

def roulette_choice(weights):
    s = sum(weights)
    if s == 0:
        return random.randrange(len(weights))
    r = random.random() * s
    cum = 0.0
    for i, w in enumerate(weights):
        cum += w
        if r <= cum:
            return i
    return len(weights)-1

# ------------------------------ I/O & driver ------------------------------

def write_output(path, partition, centers, objs, instance, dmat):
    with open(path, 'w') as f:
        f.write('BalanceViolation {:.6f}'.format(objs[0]))
        f.write('\nAffinityLoss {:.6f}'.format(objs[1]))
        f.write('\nCompactness {:.6f}'.format(objs[2]))
        for j in range(instance['m']):
            f.writelines('\nDistrict {}'.format(j+1))
            nodes_list = sorted(list(partition[j]))
            # convert to ids
            ids = [instance['nodes'][u]['id'] for u in nodes_list]
            f.writelines('\nNodes: {}'.format(' '.join(ids)))
            wsum, compact = compute_district_metrics(partition[j], centers[j], instance['nodes'], dmat)
            f.writelines('\nCompactness: {:.6f}'.format(compact))
            f.writelines('\nWeights: {:.6f} {:.6f} {:.6f}'.format(wsum[0], wsum[1], wsum[2]))
            f.writelines('\nCenter Node: {}'.format(instance['nodes'][centers[j]]['id'] if centers[j] is not None else 'None'))
            f.writelines('\n')

def main():
    # if len(sys.argv) < 3:
    #     print('Usage: python alns_territory_design.py input.txt output.txt')
    #     return
    s = input()
    t = input()
   # s = "2DU60-05-1-edited"
    input_path  = "E:/Hanoi University of Science and Technology/BKAI/Territory design/ALNS/input/" + s + "/" + t + ".dat"
    output_path = "E:/Hanoi University of Science and Technology/BKAI/Territory design/ALNS/output/" + s + "/" + t + ".out"
    inst = read_instance(input_path)
    start = time.time()
    best_part, best_centers, best_objs = alns(inst, iters=10000, time_limit=300,
                                              stable_iters = 500, seed=None)
    dmat = compute_dist_matrix(inst['nodes'])
    write_output(output_path, best_part, best_centers, best_objs, inst, dmat)
    print('Done. Best objectives (balance_violation, affinity_loss, compactness):', best_objs)
    print('Output written to', output_path)
    print('Elapsed time: {:.2f}s'.format(time.time()-start))

if __name__ == '__main__':
    main()
