#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "minst.h"
#include "iostream"

ImgArr load_image(const char* filename,nSize matSize){
//    从文件中加载图片
    FILE  *fp=NULL;
    fp=fopen(filename,"rb");
//    printf("%s\n",filename);
    if(fp==NULL)
        printf("open file failed\n");

    ImgArr labarr=(ImgArr)malloc(sizeof(MinstImgArr));


//    图片的数量
    int num_image;
    fread(&num_image,sizeof(int),1,fp);

    labarr->ImgNum=num_image;
//    std::cout<<num_image<<std::endl;

    labarr->images= (float***)malloc(num_image*sizeof(float**));

    for(int image_index=0;image_index<num_image;image_index++){
        labarr->images[image_index]= (float**)malloc(matSize.r*sizeof(float*));

        for(int row=0;row<matSize.c;row++){
            labarr->images[image_index][row] = (float*)malloc(matSize.c*sizeof(float));
            for(int col=0;col<matSize.r;col++){
                float* in=(float*)malloc(sizeof(float));
                fread(in,sizeof(float),1,fp);
                labarr->images[image_index][row][col] = in[0];
//            std::cout<<mat[row][col];
                free(in);
            }
        }
    }

    fclose(fp);
    return labarr;
}

LabelArr load_data_label(const char* filename) // 读入图像标记
{
	FILE  *fp=NULL;
	fp=fopen(filename,"rb");
	if(fp==NULL)
		printf("open file failed\n");
	assert(fp);

	int number_of_labels = 0;

	//获取训练或测试image的个数number_of_images
	fread((char*)&number_of_labels,sizeof(number_of_labels),1,fp);  
//    std::cout<<number_of_labels<<std::endl;
	int i,l;

	// 图像标记数组的初始化
	LabelArr labarr=(LabelArr)malloc(sizeof(MinstLabelArr));

	labarr->LabelNum=number_of_labels;
	labarr->labels=(int*)malloc(number_of_labels*sizeof(int));

	for(i = 0; i < number_of_labels; ++i)  
	{
        int temp;
		fread(&temp, sizeof(int),1,fp);
		labarr->labels[i]=temp;
	}

	fclose(fp);
	return labarr;
}

char intTochar(int i)// 将数字转换成字符串
{
	char * character_name= (char *) "0123456789ABCDEFGHIJKLMNPQRSTUWXYZ";
    if (i>=0&&i<=34){
    	return character_name[i];
    }
    return -1;
}

int charToInt(char i){
    char * character_name= (char *) "0123456789ABCDEFGHIJKLMNPQRSTUWXYZ";
    for(int index=0;index<34;index++){
        if(character_name[index]==i){
            return index;
        }
    }
    return -1;
}

float** transform_data(float ** inputData,nSize outputSize,int location){
//    对图片进行转换、裁剪
    int row_start_index,col_start_index;
    if(location==TOP_LEFT){
        row_start_index=0;
        col_start_index=0;
    } else if(location==BOTTOM_LEFT){
        row_start_index=15-outputSize.r;
        col_start_index=0;
    }

    float ** output = (float **) malloc(outputSize.r * sizeof(float*));

    for(int i=0;i<outputSize.r;i++){
        output[i] = (float *) malloc(outputSize.c * sizeof(float));
        for(int j=0;j<outputSize.c;j++){
            output[i][j]=inputData[i+row_start_index][j+col_start_index];
        }
    }

    return output;
}
char * combine_strings(char *a, char *b) // 将两个字符串相连
{
	char *ptr;
	int lena= (int) strlen(a), lenb= (int) strlen(b);
	int i,l=0;
	ptr = (char *)malloc((lena+lenb+1) * sizeof(char));
	for(i=0;i<lena;i++)
		ptr[l++]=a[i];
	for(i=0;i<lenb;i++)
		ptr[l++]=b[i];
	ptr[l]='\0';
	return(ptr);
}

//void save_Img(ImgArr imgarr,char* filedir) // 将图像数据保存成文件
//{
//	int img_number=imgarr->ImgNum;
//
//	int i,r;
//	for(i=0;i<img_number;i++){
//		const char* filename=combine_strings(filedir,combine_strings(intTochar(i), (char *) ".gray"));
//		FILE  *fp=NULL;
//		fp=fopen(filename,"wb");
//		if(fp==NULL)
//		{
//			printf("write file failed\n");
//			assert(fp);
//		}
//
//		assert(fp);
//
//		for(r=0;r<imgarr->ImgPtr[i].r;r++)
//			fwrite(imgarr->ImgPtr[i].ImgData[r],sizeof(float),imgarr->ImgPtr[i].c,fp);
//
//		fclose(fp);
//	}
//}