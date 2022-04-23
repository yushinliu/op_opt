#include<vector>
#include<fstream>
#include<iostream>
#include<limits.h>
#include<float.h>
#include<ctime>
#include<algorithm>
#include<cstring>


using namespace std;

/***
 *
 * Tensor is the data structure to store data and layout;
 *
 ***/
class Tensor{
public:
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

    int& operator[](int index){
        return data[index];
    }

    vector<int> size(){
        return sizes;
    }

    int len(){
        return length;
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
Tensor maxpooling_add(Tensor &src, Tensor& add_src, Tensor &res){
    // kernel = 3, pad = 1, stride = 2;
    int k = 3;

    // retrive the input ouput sizes
    auto sizes = src.size();
    int N = src.size().at(0);
    int C = src.size().at(1);
    int H = src.size().at(2);
    int W = src.size().at(3);
    int H_out = res.size().at(2);
    int W_out = res.size().at(3);

    vector<int> index_size = res.size();
    vector<int> buffer_size = {k*k};
    Tensor buffer(buffer_size);
    int* buffer_ptr = buffer.rawPtr();
    index_size.push_back(2);
    Tensor index_out(index_size);

    // start calculate
    clock_t start = clock();
#pragma omp for 
    for(int n=0;n<N;n++){
        for(int c=0;c<C;c++){
            for(int h=0;h<H_out;h++){
                for(int w=0;w<W_out;w++){
                    int max_num = INT_MIN;
                    // set buffer to int_min
                    memset(buffer_ptr, 0x80, k*k*sizeof(int));
                    for(int i=0;i<k;i++){
                        for (int j=0;j<k;j++){
                            int ii = h*2+i;
                            int jj = w*2+j;
                            if(ii==0||ii==H+1||jj==0||jj==W+1) continue;
                            buffer_ptr[k*i+j] = src[n*C*H*W + c*H*W + (ii-1)*W + (jj-1)];
                            // int cur_num = src[n*C*H*W + c*H*W + (ii-1)*W + (jj-1)];
                            // if(cur_num >max_num) {
                            //    max_num = cur_num;
                            //    index_out[n*C*H_out*W_out + c*H_out*W_out + h*W_out + w] = ii;
                            //    index_out[n*C*H_out*W_out + c*H_out*W_out + h*W_out + w + 1] = jj;
                            // }
                        }
                    }
                    auto max_iter = max_element(buffer_ptr,buffer_ptr+k*k);
                    max_num = *max_iter;
                    int idx = distance(buffer_ptr, max_iter);
                    int i_idx = 2*h+idx/k;
                    int j_idx = 2*w+idx%k;
                    index_out[n*C*H_out*W_out + c*H_out*W_out + h*W_out + w] = i_idx;
                    index_out[n*C*H_out*W_out + c*H_out*W_out + h*W_out + w + 1] = j_idx;
                    res[n*C*H_out*W_out + c*H_out*W_out + h*W_out + w] = max_num + add_src[n*H_out*W_out + h*W_out + w];
                }
            }
        }
    }
    // end calculate
    clock_t end = clock();
    std::cout << "Spend: " << (double)(end-start)/CLOCKS_PER_SEC << " seconds" << std::endl;
    return index_out;

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
            if(a[i]!=b[i]) throw "Value of tensors are not equal!";
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
    int H_out = 56;
    int W_out = 56;
    //int *a= new int[N*C*H*W];
    //for(int i=0;i<N*C*H*W;i++){
    //    a[i]=rand()%(100+100+1)-100;
    //}
    //
    string input_path = "./input.txt";
    string add_path = "./input2.txt";
    string out_path = "./output.txt";
    string out_idx_path = "./output_idx.txt";
    string gt_path = "./gt.txt";
    string gt_idx_path = "./gt_idx.txt";
    vector<int> pooling_sizes = {N,C,H,W};
    vector<int> add_sizes = {N,1,H_out,W_out};
    vector<int> out_sizes = {N,C,H_out,W_out};
    vector<int> index_sizes = {N,C,H_out,W_out,2};

    // read the input data;
    Tensor a = readTxt(input_path, pooling_sizes);
    Tensor b = readTxt(add_path, add_sizes);

    // read the ground truth output;
    Tensor gt = readTxt(gt_path, out_sizes);
    Tensor gt_idx = readTxt(gt_idx_path, index_sizes);
    Tensor out(out_sizes);

    // caculate;
    Tensor index_map = maxpooling_add(a, b, out);

    // dump the output;
    out.dump(out_path);
    index_map.dump(out_idx_path);

    // unit test;
    compare(gt, out);
    compare(gt_idx, index_map);
}
