#include <stdlib.h>
#include <stdio.h>
#include <random>
#include "cnn.h"
#include <assert.h>
#include "iostream"

#include "time.h"
using namespace std;
//项目路径
char * project_path= (char *) "/home/jdwang/ClionProjects/digitRecognition_C/";

void test_model_all(){
//    测试34分类模型
    clock_t ticks1, ticks2;
    //加载cnn模型权重
    char* file_name = combine_strings(project_path, (char *) "model/model_all_weight.mat");
    CNN *cnn = load_cnn_weight(file_name,34,{15,15});
    //加载数据
    ImgArr images_array = load_image(combine_strings(project_path, (char *) "input_data/input_data.mat"), {15,15});
    LabelArr label_array = load_data_label(combine_strings(project_path, (char *) "input_data/input_data_label.mat"));

    int result = 0;
    //    测试
    for(int image_index=0;image_index<label_array->LabelNum;image_index++){
        //        开始测试时间
        ticks1=clock();
        int predict = forward(cnn,images_array->images[image_index]);
        std::cout<<image_index<<","<<intTochar(label_array->labels[image_index])<<" "<<intTochar(predict)<<std::endl;
        std::cout<<"是否正确:"<<(label_array->labels[image_index]==predict)<<"\n";
        result += label_array->labels[image_index]==predict;
//        结束测试时间
        ticks2=clock();

//        std::cout<<"用时："<<(ticks2-ticks1)*1000/(float)(CLOCKS_PER_SEC)<<"ms"<<std::endl;
    }
    std::cout<<std::endl;
    std::cout<<result;
    printf("准确率:%f\n",(result/(float)label_array->LabelNum));
}

void readImage(){
//    读取bmp图片
    char * image_path = (char *) "/home/jdwang/ClionProjects/digitRecognition_C/input_data/201607121911032581.bmp";
    printf("%s\n",image_path);

//    char * image_path =combine_strings(project_path, (char *) "input_data/input_data/201607121911032581.bmp");
    FILE * fp = fopen(image_path,"rb");
    if(fp==NULL){
        printf("open image failed!(%s)\n",image_path);
    }

    fseek(fp, 0L, SEEK_END);
    long size = ftell(fp);
    unsigned char * imgbuf = new  unsigned char[size+ 1];
    fseek(fp,0x0L,SEEK_SET);//图片源
    fread(imgbuf, sizeof(unsigned char), size, fp);
    cout<<size<<endl;
    cout<<imgbuf[0]<<endl;
    cout<<imgbuf[1]<<endl;
    cout<<imgbuf[2]<<endl;
    cout<<imgbuf[3]<<endl;
    cout<<imgbuf[4]<<endl;
    cout<<imgbuf[10]<<endl;
    cout<<imgbuf[11]<<endl;
    cout<<imgbuf[12]<<endl;
    cout<<imgbuf[450]<<endl;
    assert(NULL);
}
void test_classifier(){
//    测试分类器：混合模型
    clock_t ticks1, ticks2;

    //加载cnn模型权重
    CharClassifier *model = load_all_model_weight(project_path);
    //加载数据
    ImgArr images_array = load_image(combine_strings(project_path, (char *) "input_data/input_data.mat"), {15,15});
    LabelArr label_array = load_data_label(combine_strings(project_path, (char *) "input_data/input_data_label_pred.mat"));

    int result = 0;
    //    测试
    for(int image_index=0;image_index<label_array->LabelNum;image_index++){
        //        开始测试时间
        ticks1=clock();
//        image_index = 39125;
//        printmat(images_array->images[image_index],{15,15});
//        预测
        int y_pred = predict(model,images_array->images[image_index]);
        std::cout<<image_index<<","<<intTochar(label_array->labels[image_index])<<" "<<y_pred<<" "<<intTochar(y_pred)<<std::endl;
        std::cout<<"是否正确:"<<(label_array->labels[image_index]==y_pred)<<"\n";
        result += label_array->labels[image_index]==y_pred;
//        assert(NULL);
//        结束测试时间
        ticks2=clock();
//        std::cout<<"用时："<<(ticks2-ticks1)*1000/(float)(CLOCKS_PER_SEC)<<"ms"<<std::endl;
    }
    cout<<endl;
    printf("正确个数：%d,准确率:%f\n",result,(result/(float)label_array->LabelNum));
}

/*主函数*/
int main()
{
//    readImage();

    //    测试34分类模型
//    test_model_all();

    //    测试分类器：混合模型
    test_classifier();
    return 0;
}