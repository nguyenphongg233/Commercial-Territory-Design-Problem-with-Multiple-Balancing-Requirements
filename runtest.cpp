#include<bits/stdc++.h>

using namespace std;

#define read() ios_base::sync_with_stdio(0);cin.tie(0);cout.tie(0)
#define day() time_t now = time(0);char* x = ctime(&now);cerr<<"Right Now Is : "<<x<<"\n"

#define ii pair<int,int>
#define X first1
#define Y second 

const long long MAX = (int)1e6 + 5;
const long long INF = (int)1e9;
const long long MOD = (int)1e9 + 7;

int l,r;
string s;
struct node{
	int l,r;
	string s;
};
signed main(){
	
	read();

//	string c = "reformat.exe";
	vector<node> f;	
	f.push_back({1,20,"2DU60-05-"});
	f.push_back({1,20,"2DU80-05-"});
	f.push_back({1,20,"2DU100-05-"});
	f.push_back({1,20,"2DU120-05-"});
	f.push_back({1,10,"DU150-05-"});	
	f.push_back({1,10,"DU200-05-"});
	//f.push_back({1,20,"DU500-20-"});
	//string dictionary = "input_edited";
//	system(("md " + dictionary).c_str());
	
	for(int j = 0;j < (int)f.size();j++){
		int l = f[j].l;
		int r = f[j].r;
		string s = f[j].s;
		for(int i = l;i <= r;i++){
			//cerr << ("python planar_graph_ver2.py > input_edited//" + s + to_string(i) + "-edited.dat").c_str() << "\n";
			//system(("python planar_graph_ver2.py > input_edited//" + s + to_string(i) + "-edited.dat").c_str());
			system(("echo " + s + to_string(i) + "-edited> input.txt").c_str());
			cerr << ("echo " + s + to_string(i) + "-edited> input.txt").c_str() << "\n";
			// system("python alns.py < input.txt");
			// cerr << ("python alns.py < input.txt\n");
			system("python visualization.py < input.txt");
			cerr << ("python visualization.py < input.txt\n");
			//system((c + " < input\\" + s + to_string(i) + ".dat > " + dictionary + "\\" + s + to_string(i) + "-edited.dat").c_str());
			//system(("type " + s + to_string(i) + ".out >>" + "result.txt").c_str());
		}
	}
	
}

