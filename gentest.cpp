#include <bits/stdc++.h>
using namespace std;

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

long long Rand(long long l,long long r){
    long long x = rng() % (long long)1e18;
    return x % (r - l + 1) + l;
}
struct Point {
    double x, y;
    int p1, p2, p3;
};

struct Edge {
    int u, v;
    double d2;
    bool operator<(Edge const& o) const { return d2 < o.d2; }
};

// orientation: 0 collinear, 1 clockwise, 2 counterclockwise
int orient(const Point &a, const Point &b, const Point &c) {
    double v = (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y);
    if (fabs(v) < 1e-12) return 0;
    return (v > 0) ? 1 : 2;
}

bool onSegment(const Point &a, const Point &b, const Point &p) {
    return p.x <= max(a.x,b.x) + 1e-12 && p.x + 1e-12 >= min(a.x,b.x)
        && p.y <= max(a.y,b.y) + 1e-12 && p.y + 1e-12 >= min(a.y,b.y);
}

bool segmentsIntersect(const Point &p1, const Point &q1, const Point &p2, const Point &q2) {
    int o1 = orient(p1, q1, p2);
    int o2 = orient(p1, q1, q2);
    int o3 = orient(p2, q2, p1);
    int o4 = orient(p2, q2, q1);

    if (o1 != o2 && o3 != o4) return true;
    if (o1 == 0 && onSegment(p1, q1, p2)) return true;
    if (o2 == 0 && onSegment(p1, q1, q2)) return true;
    if (o3 == 0 && onSegment(p2, q2, p1)) return true;
    if (o4 == 0 && onSegment(p2, q2, q1)) return true;
    return false;
}

signed main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n = 500;
	cin >> n;
	
    uniform_real_distribution<double> distCoord(0.0, 1000.0);
    uniform_int_distribution<int> distP1(1, 40);
    uniform_int_distribution<int> distP2(1, 40);
    uniform_int_distribution<int> distP3(1, 40);
    uniform_int_distribution<int> distP(5, 20);

    vector<Point> pts(n);
    for (int i = 0; i < n; ++i) {
        pts[i].x = distCoord(rng);
        pts[i].y = distCoord(rng);
        pts[i].p1 = distP1(rng);
        pts[i].p2 = distP2(rng);
        pts[i].p3 = distP3(rng);
    }

    // candidate edges
    vector<Edge> cand;
    cand.reserve((long long)n*(n-1)/2);
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            double dx = pts[i].x - pts[j].x;
            double dy = pts[i].y - pts[j].y;
            cand.push_back({i,j,dx*dx+dy*dy});
        }
    }
    sort(cand.begin(), cand.end());

    vector<pair<int,int>> edges;
    for (auto &e : cand) {
        int u = e.u, v = e.v;
        bool ok = true;
        for (auto &ex : edges) {
            int a = ex.first, b = ex.second;
            if (a==u || a==v || b==u || b==v) continue;
            if (segmentsIntersect(pts[u], pts[v], pts[a], pts[b])) {
                ok = false;
                break;
            }
        }
        if (ok) edges.emplace_back(u,v);
    }

    int m = edges.size();
    int p = 50;
	cin >> p;
	
    cout.setf(std::ios::fixed); cout<<setprecision(6);

    // 1. in n
    cout << n << "\n";

    // 2. n dòng node
    for (int i=0;i<n;i++) {
        cout << i << " " << pts[i].x << " " << pts[i].y << " "
             << pts[i].p1 << " " << pts[i].p2 << " " << pts[i].p3 << " 0\n";
    }

    // 3. số cạnh
    cout << m << "\n";

    // 4. m dòng cạnh
    for (auto &e : edges) {
        cout << e.first << " " << e.second << "\n";
    }

    // 5. dòng p p 0.5 0.5 0.5 0.5
    cout << p << " " << p << " 0.05 0.05 0.05 0.05\n";

    // 6. ma trận n x p toàn 1
    for (int i=0;i<p;i++) {
        for (int j=0;j<n;j++) {
            cout << 1 << (j+1==p?'\n':' ');
        }
    }

    return 0;
}
