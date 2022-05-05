#include<vector>
#include<fstream>
#include<iostream>
#include<limits.h>
#include<float.h>
#include<ctime>
#include<algorithm>
#include<cstring>
#include<unordered_map>
#include<omp.h>


using namespace std;

/***
 *
 * Tensor is the data structure to store data and layout;
 *
 ***/
class Tensor{
public:
    Tensor(){};
    Tensor(vector<int> layout){
       int size = 1;
       for(auto s:layout){
           size *=s;
       }
       data = new int[size];
       for(int i=0;i<size;i++){
           data[i] = INT_MIN;
       }
       sizes = layout;
       length = size;
    }
    Tensor(const Tensor& t){
        data = new int[t.length];
        length = t.length;
        sizes = t.sizes;
        for(int i=0;i<t.length;i++){
           data[i] = t.data[i];
        }
    }
    Tensor& operator=(const Tensor& t){
        if(data != t.data){
            delete data;
            data = new int[t.length];
            length = t.length;
            sizes = t.sizes;
            for(int i=0;i<t.length;i++){
                data[i] = t.data[i];
            }
        }

    }
    Tensor(Tensor&& t){
        length = t.length;
        sizes = t.sizes;
        data = t.data;
    }

    Tensor& operator=(Tensor&& t){
        if(data != t.data){
            delete data;
            data = t.data;
            length = t.length;
            sizes = t.sizes;
            t.data = nullptr;
        }
        return *this;
    }


    int& operator[](int index){
        return data[index];
    }

    vector<int> size(){
        return sizes;
    }

    int len(){
        return length;
    }

    void reset(){
        data = nullptr;
    }

    bool empty(){
        return data==nullptr?false:true;
    }

    int* rawPtr(){
        return data;
    }

    // dump the data to txt;
    void dump(const string& name){
        ofstream writetxt(name);
        if(!writetxt.is_open()) return;
        int size = 1;
        for(auto s:sizes){
            size*=s;
        }
        for(int i=0;i<size;i++){
            writetxt << data[i];
            writetxt << ' ';
        }
        writetxt.close();
    }

    ~Tensor() {
        if(data){
            delete data;
        }
    }
private:
    vector<int> sizes;
    int length;
    int* data = nullptr;;
};

/***
 *
 * Check if the C channls of res and add meet the requirement of broadcasting;
 *
 ***/
vector<int> check_broadcast(vector<int>& res, vector<int>& add){
    auto res_dims = res.size();
    auto add_dims = add.size();
    auto ndim = res_dims > add_dims?res_dims:add_dims;
    vector<int> broadcast_size(ndim);
    for(int i=ndim-1;i>=0;i--){
        auto offset = ndim-1-i;
        auto res_dim = res_dims-1-offset;
        auto add_dim = add_dims-1-offset;
        auto size_res = (res_dim>=0)?res[res_dim]:1;
        auto size_add = (res_dim>=0)?res[res_dim]:1;
        if(!(size_res==size_add || size_res==1 || size_add==1)){
            throw("Can not broadcast!");
        }
        broadcast_size[i] = size_res==1?size_add:size_res;
    }
    return broadcast_size;
}


/***
 *
 * MaxpoolingAdd op class;
 *
 ***/
class MaxpoolingAdd{
public:
    MaxpoolingAdd(vector<int> src_size, vector<int> add_size):src_size(src_size),add_size(add_size){
        res_size = {src_size.at(0), src_size.at(1), (src_size.at(2)-3)/2+2,(src_size.at(3)-3)/2+2};
        broadcast_size = check_broadcast(res_size, add_size);
        res = Tensor(res_size);
        res_broadcast = Tensor(broadcast_size);
    }
    void forward(Tensor &src, Tensor&add);
    Tensor& out(){return res_broadcast;}
private:
    Tensor res;
    Tensor res_broadcast;
    vector<int> src_size;
    vector<int> add_size;
    vector<int> res_size;
    vector<int> broadcast_size;
};

/***
 *
 * Map to store ops;
 *
 ***/
unordered_map<string,MaxpoolingAdd> op_maps;

/***
 *
 * maxpooling_add: fusion kernel of maxpooling and elementwise-add;
 *
 * Input tensor:
 *      src: the input tensor of fusion kernel, layout is NCHW, data type is INT;
 *      add_src: the element-wise add tensor, layout is NH_outW_out, data type is INT;
 *      res: the output of fusion kernel, layout is NCH_outW_out, data type is INT;
 *
 * Output tensor:
 *      index: the index map of output, layout is NCH_outW_out2, data type is INT;
 *
 ***/
void MaxpoolingAdd::forward(Tensor &src, Tensor &add){
    // kernel = 3, pad = 1, stride = 2;
    clock_t start = clock();
    omp_set_num_threads(4);
    double start_1 = omp_get_wtime();

    // retrive the input ouput sizes
    int N = src_size.at(0);
    int C = src_size.at(1);
    int H = src_size.at(2);
    int W = src_size.at(3);
    int H_out = res_size.at(2);
    int W_out = res_size.at(3);

    //vector<int> buffer_size = {9};

    // start calculate
    // FIXME: using parallel causes correct output of max values but false result of index map;
#pragma omp parallel num_threads(4)
    {
    //Tensor buffer(buffer_size);
    //int* buffer_ptr = buffer.rawPtr();
#pragma omp for
    for(int n=0;n<N;n++){
        for(int c=0;c<C;c++){
            for(int h=0;h<H_out;h++){
                for(int w=0;w<W_out;w++){
                    int max_num = INT_MIN;
                    // set buffer to int_min
                    //memset(buffer_ptr, 0x80, k*k*sizeof(int));
                    for(int i=0;i<3;i++){
                        for (int j=0;j<3;j++){
                            int ii = h*2+i;
                            int jj = w*2+j;
                            if(ii==0||ii==H+1||jj==0||jj==W+1) continue;
                            //buffer_ptr[k*i+j] = src[n*C*H*W + c*H*W + (ii-1)*W + (jj-1)];
                            int cur_num = src[n*C*H*W + c*H*W + (ii-1)*W + (jj-1)];
                            if(cur_num >max_num) {
                               max_num = cur_num;
                            }
                        }
                    }
                    res[n*C*H_out*W_out + c*H_out*W_out + h*W_out + w] = max_num;
                    //auto max_iter = max_element(buffer_ptr,buffer_ptr+k*k);
                    //max_num = *max_iter;
                }
            }
        }
    }

    int N_add = add_size.at(0);
    int C_add = add_size.at(1);
    int H_add = add_size.at(2);
    int W_add = add_size.at(3);
    int N_b = broadcast_size.at(0);
    int C_b = broadcast_size.at(1);
    int H_b = broadcast_size.at(2);
    int W_b = broadcast_size.at(3);
#pragma omp for
    for(int n=0;n<N_b;n++){
        int n_res = min(n,N-1);  
        int n_add = min(n,N_add-1);
        for(int c=0;c<C_b;c++){
            int c_res = min(c,C-1);  
            int c_add = min(c,C_add-1);
            for(int h=0;h<H_b;h++){
                int h_res = min(h,H_out-1);  
                int h_add = min(h,H_add-1);
                for(int w=0;w<W_b;w++){
                    int w_res = min(w,W_out-1);  
                    int w_add = min(w,W_add-1);
                    res_broadcast[n*C_b*H_b*W_b+c*H_b*W_b+h*W_b+w] = res[n_res*C*W_out*H_out+c_res*W_out*H_out+h_res*W_out+w_res] + add[n_add*C_add*H_add*W_add+c_add*H_add*W_add+h_add*W_add+w_add];
                }
            }

        } 
    }
    }
    // end calculate
    double end_1 = omp_get_wtime();
    clock_t end = clock();
    std::cout << "Spend: " << (double)(end-start)/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Omp Spend: " << (double)(end_1-start_1) << " seconds" << std::endl;
}

/***
 *
 * readTxt is to read the data from the txt path;
 *
 ***/
Tensor readTxt(const string& path, vector<int>& sizes){
    ifstream input(path);
    Tensor a(sizes);
    int num = 0;
    int index = 0;
    while(input >>  num){
        a[index] = num;
        index++;
    }
    return a;
}

/***
 *
 * compare is the unittest module which compares two tensors;
 *
 ***/
void compare(Tensor& a, Tensor& b){
    try {
        if(!a.empty()||!b.empty()) throw "Tensors must be non-empty!";
        if(a.size()!=b.size()) throw "Sizes of tensors are not equal!";
        int len = a.len();
        for(int i=0;i<len;i++){
            if(a[i]!=b[i]){

            cout << a[i] << " " << b[i] << endl;
            throw "Value of tensors are not equal!";
        }
        }
    } catch(const char* msg) {
        cout << msg << endl;
        return;
    }
    std::cout << "Pass the test! "<< std::endl;
    return;
}

int main(){
    int N = 32;
    int C = 64;
    int H = 112;
    int W = 112;
    int H_add = 56;
    int W_add = 56;
    //int *a= new int[N*C*H*W];
    //for(int i=0;i<N*C*H*W;i++){
    //    a[i]=rand()%(100+100+1)-100;
    //}
    //
    string input_path = "./input.txt";
    string add_path = "./input2.txt";
    string out_path = "./output.txt";
    string gt_path = "./gt.txt";
    vector<int> pooling_sizes = {N,C,H,W};
    vector<int> add_sizes = {N,1,H_add,W_add};

    // read the input data;
    Tensor a = readTxt(input_path, pooling_sizes);
    Tensor b = readTxt(add_path, add_sizes);


    // caculate;
    auto op = MaxpoolingAdd(pooling_sizes,add_sizes);
    op_maps.emplace("maxpoolingadd_1", op);
    op.forward(a, b);

    // dump the output;
    Tensor& out = op.out();
    auto out_size = out.size();
    // read the ground truth output;
    Tensor gt = readTxt(gt_path, out_size);
    out.dump(out_path);

    // unit test;
    compare(gt, out);
}
