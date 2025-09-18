// 23 - 12 - 23 

#include<bits/stdc++.h>

using namespace std;

#define read() ios_base::sync_with_stdio(0);cin.tie(0);cout.tie(0)
#define day() time_t now = time(0);char* x = ctime(&now);cerr<<"Right Now Is : "<<x<<"\n"

#define ii pair<int,int>
#define X first
#define Y second 

const long long MAX = (int)1000 + 5;
const long long INF = (int)1e9;
const long long MOD = (int)1e9 + 7;
const int N = 50;
int n,m,p;
struct node{
	int id;
	double x_cord,y_cord;
	int w[3];
}qrx[MAX];
vector<int> adj[MAX];
double t[3];
int f[N][MAX];
vector<int> node[MAX];
int col[MAX];
string sx;

double dist(int i,int j){
	return sqrt((qrx[i].x_cord - qrx[j].x_cord) * (qrx[i].x_cord - qrx[j].x_cord) + (qrx[i].y_cord - qrx[j].y_cord) * (qrx[i].y_cord - qrx[j].y_cord));
}
void count_centroid(){
	int cnt = 0;
	vector<bool> fx(n + 5,0);
	for(int i = 0;i < n;i++){
		if(fx[i])continue;
		cnt++;
		deque<int> h;
		h.push_back(i);
		fx[i] = 1;
		cout << i << " :\n";
		while(!h.empty()){
			int u = h.front();
			h.pop_front();
			cout << u << " ";
			for(auto v : adj[u]){
				if(!fx[v]){
					fx[v] = 1;
					h.push_back(v);
				}
			}
		}
		cout << "\n";
	}
	cout << cnt << "\n";
}
void clear(){
	for(int i = 0;i < MAX;i++){
		adj[i].clear();
		node[i].clear();
		col[i] = 0;
	}
}
int run_code(string sx,bool okx = 0){
	clear();
	if(okx)cerr << ("input_edited//" + sx + "-edited.dat").c_str() << "\n";
	//return 0;
	ifstream file(("input_edited//" + sx + "-edited.dat").c_str());
	file >> n;
	//cout << n << '\n';
	for(int i = 0,x;i < n;i++){
		file >> qrx[i].id >> qrx[i].x_cord >> qrx[i].y_cord >> qrx[i].w[0] >> qrx[i].w[1] >> qrx[i].w[2] >> x;
		//cout << qrx[i].id << " " << qrx[i].x_cord << " " << qrx[i].y_cord << " " << qrx[i].w[0] << " " << qrx[i].w[1] << " " << qrx[i].w[2] << " " << x << "\n";
	}
	file >> m;
	//cout << m << "\n";
	for(int i = 1,u,v;i <= m;i++){
		file >> u >> v;
		//cout << u << " " << v << "\n";
		adj[u].push_back(v);
		adj[v].push_back(u);
	}
	file >> p >> p >> t[0] >> t[1] >> t[2] >> t[2];
	if(okx)cerr << p << " " << t[0] << " " << t[1] << " " << t[2] << "\n";
	for(int i = 1;i <= p;i++){
		for(int j = 1;j <= n;j++){
			file >> f[i][j];
			//cout << f[i][j] << " ";
		}
		//cout << "\n";
	}
	
	
	file.close();
	ifstream file_(("output_edited//" + sx + "-edited.out").c_str());
	if(okx)cerr << ("output_edited//" + sx + "-edited.out").c_str() << "\n";
	string s;
	double balance,affinity,compactness;
	file_ >> s >> balance >> s >> affinity >> s >> compactness;
	if(okx)cerr << balance << " " << affinity << " " << compactness << " " << s << "\n";
	double total_w[] = {0,0,0};
	double total_dist = 0;
	bool ok = 1;
	vector<bool> fx(n + 5,0);
	for(int i = 1,x;i <= p;i++){
		file_ >> s >> x >> s;
		if(okx)cerr << x << " :\n";
		while(file_ >> s){
			if(s[0] > '9' || s[0] < '0')break;
			int r = stoll(s);
			node[x].push_back(r);
			col[r] = x; 
			if(okx)cerr << r << " ";
		}
		if(okx)cerr << "\n";
		double com;
		double w_[] = {0,0,0};
		int center;
		file_ >> com >> s >> w_[0] >> w_[1] >> w_[2] >> s >> s >> center; 
		//if(okx)cerr << com << " " << w_[0] << " " << w_[1] << " " << w_[2] << " " << center << "\n";
		deque<int> h;
		h.push_back(center);
		fx[center] = 1;
		int cnt = 0;
		double distant = 0;
		double ww[] = {0,0,0};
		
		vector<int> nodex;
		while(!h.empty()){
			int u = h.front();
			h.pop_front();
			cnt++;
			distant += dist(u,center);
			ww[0] += qrx[u].w[0];
			ww[1] += qrx[u].w[1];
			ww[2] += qrx[u].w[2];
			if(okx)nodex.push_back(u);
			for(auto v : adj[u]){
				if(col[v] == col[u] && !fx[v]){
					fx[v] = 1;
					h.push_back(v);
				}
			}
		}
		if(okx){
			sort(nodex.begin(),nodex.end());
			for(auto v : nodex)cout << v << " ";
		}
		if(okx)cout << "\n";
		if(cnt != (int)node[x].size())ok = 0;
		if(okx)cerr << distant << " " << ww[0] << " " << ww[1] << " " << ww[2] << " " << center << "\n";
		if(okx)cerr << com << " " << w_[0] << " " << w_[1] << " " << w_[2] << " " << center << "\n";
		for(int i = 0;i < 3;i++){
			total_w[i] += w_[i];
			if(abs(ww[0] - w_[0]) > 1e-2){
				ok = 0;
				//cout << "FUCK\n";
				//break;
			}
		}
		if(abs(distant - com) > 1e-2){
			ok = 0;
			//cout << "FUCK2\n";
			//break;
		}
		total_dist += distant;
	}

	if(okx){
		for(int i = 0;i < n;i++){
			if(!fx[i]){
				ok = 0;
				cerr << i << " NOT COVERED\n";
			}
		}
		count_centroid();
	}
	
	// for(int i = 0;i < 3;i++){
	// 	if(abs(total_w[0] - w[0]) > 1e-7){
	// 		ok = 0;
	// 		break;
	// 	}
	// }
	if(abs(total_dist - compactness) > 1e-2){
		ok = 0;
	}
	
	cerr << sx << "\t: " << (ok ? "LEGIT" : "NOT LEGIT") << "\n";
	//cout << (ok ? "LEGIT" : "NOT LEGIT") << "!!!!!!!\n";
	return ok;
}

signed main(){
	read();

	vector<pair<pair<int,int>,string>> f;	
	f.push_back({{1,20},"2DU60-05-"});
	f.push_back({{1,20},"2DU80-05-"});
	f.push_back({{1,20},"2DU100-05-"});
	f.push_back({{1,20},"2DU120-05-"});
	f.push_back({{1,10},"DU150-05-"});	
	f.push_back({{1,10},"DU200-05-"});
	f.push_back({{1,20},"DU500-20-"});

	for(int j = 0;j < (int)f.size();j++){
		int l = f[j].X.X;
		int r = f[j].X.Y;
		string s = f[j].Y;
		for(int i = l;i <= r;i++){
			sx = s + to_string(i);
			if(!run_code(sx)){
				//cerr << "WRONG ON " << sx << "\n";
			}
			//system(("type " + s + to_string(i) + ".out >>" + "result.txt").c_str());
		}
	}
	//run_code("2DU100-05-2",1);
	return 0;
}