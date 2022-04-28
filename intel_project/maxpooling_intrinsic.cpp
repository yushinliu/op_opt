#include<vector>
#include<fstream>
#include<iostream>
#include<limits.h>
#include<float.h>
#include<ctime>
#include<algorithm>
#include<cstring>
#include <x86intrin.h>


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
 * Check if the C channls of res and add meet the requirement of broadcasting;
 *
 ***/
void check_broadcast(Tensor& res, Tensor& add){
    auto res_sizes = res.size();
    auto add_sizes = add.size();
    for(int i=0;i<res_sizes.size();i++){
        int C_add = add_sizes.at(i);
        int C_res = res_sizes.at(i);
        if(i==1){
            if(C_res%C_add) throw "Can not broadcast!";
        } else if(C_add!=C_res) {
            throw "Can not broadcast!";
        }
    }
    return;
}

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
    clock_t start = clock();
    int k = 3;

    // retrive the input ouput sizes
    auto sizes = src.size();
    int N = src.size().at(0);
    int C = src.size().at(1);
    int H = src.size().at(2);
    int W = src.size().at(3);
    int H_out = res.size().at(2);
    int W_out = res.size().at(3);

    check_broadcast(res,add_src);
    int C_add = add_src.size().at(1);

    vector<int> index_size = res.size();
    vector<int> buffer_size = {k*k};
    vector<int> buffer_size1 = {4};
    index_size.push_back(2);
    Tensor index_out(index_size);

    int W_num = W_out/4;
    int W_res = W_out%4;

    // start calculate
    // FIXME: using parallel causes correct output of max values but false result of index map;
#pragma omp parallel num_threads(4)
    {
    Tensor buffer(buffer_size);
    Tensor max_nums(buffer_size1);
    int* buffer_ptr = buffer.rawPtr();
    int* max_ptr = max_nums.rawPtr();
#pragma omp for
    for(int n=0;n<N;n++){
        for(int c=0;c<C;c++){
            for(int h=0;h<H_out;h++){
                for(int w=0;w<W_num;w++){
                    // set buffer to int_min
                    memset(buffer_ptr, 0x80, k*k*sizeof(int));
                    int int_min = buffer_ptr[0];
                    for(int m=0;m<4;m++){
                        int w_idx = 4*w+m;
                        for(int i=0;i<k;i++){
                            for (int j=0;j<k;j++){
                                int ii = h*2+i;
                                int jj = w_idx*2+j;
                                if(ii==0||ii==H+1||jj==0||jj==W+1) continue;
                                buffer_ptr[k*i+j] = src[n*C*H*W + c*H*W + (ii-1)*W + (jj-1)];
                            }
                        }
                        //using intrinsic for calculate max value in the buffer;
                        __m128i max1 = _mm_load_si128((__m128i*)buffer_ptr); 
                        __m128i max2 = _mm_load_si128((__m128i*)(buffer_ptr+4)); 
                        __m128i max_out1 = _mm_max_epi32(max1, max2);
                        int * out_ptr = (int*)(&max_out1);
                        max1 = _mm_setr_epi32(out_ptr[0],out_ptr[1],int_min,int_min);
                        max2 = _mm_setr_epi32(out_ptr[2],out_ptr[3],int_min,int_min);
                        max_out1 = _mm_max_epi32(max1, max2);
                        max1 = _mm_setr_epi32(out_ptr[0],int_min,int_min,int_min);
                        max2 = _mm_setr_epi32(out_ptr[1],int_min,int_min,int_min);
                        max_out1 = _mm_max_epi32(max1, max2);
                        max1 = _mm_setr_epi32(out_ptr[0],int_min,int_min,int_min);
                        max2 = _mm_setr_epi32(buffer_ptr[k*k-1],int_min,int_min,int_min);
                        max_out1 = _mm_max_epi32(max1, max2);
                        max_ptr[m] = out_ptr[0];
                        int idx = 0;
                        for(int i=0;i<k*k;i++){
                            if(buffer_ptr[i]==max_ptr[m]) idx = i;
                        }
                        int i_idx = 2*h+idx/k;
                        int j_idx = 2*w_idx+idx%k;
                        index_out[n*C*H_out*W_out + c*H_out*W_out + h*W_out + w_idx] = i_idx;
                        index_out[n*C*H_out*W_out + c*H_out*W_out + h*W_out + w_idx + 1] = j_idx;
                    }
                    int w_idx = 4*w;
                    //using intrinsic for calculate elementwise_add;
                    __m128i add1 = _mm_load_si128((__m128i*)max_ptr);
                    __m128i add2 = _mm_load_si128((__m128i*)&add_src[n*H_out*W_out+(c%C_add)*H_out*W_out+h*W_out+w_idx]);
                    __m128i out = _mm_add_epi32(add1,add2);
                    _mm_storeu_si128((__m128i*)(&res[n*C*H_out*W_out+c*H_out*W_out+h*W_out+w_idx]),out);
                }
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
            if(a[i]!=b[i]) {
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
    string gt_idx_path = "./gt_idx1.txt";
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
