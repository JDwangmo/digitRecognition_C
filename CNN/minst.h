#ifndef __MINST_
#define __MINST_
/*
图片数据的数据结构和操作
*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "cnn.h"

//上左
#define TOP_LEFT 0
//下左
#define BOTTOM_LEFT 1



typedef struct MinstImgArr{
	int ImgNum;        // 存储图像的数目
	float*** images;  // 存储图像数组指针
}*ImgArr;              // 存储图像数据的数组


typedef struct MinstLabelArr{
	int LabelNum;
	int* labels;
}*LabelArr;              // 存储图像标记的数组

LabelArr load_data_label(const char* filename); // 读入图像标记

ImgArr load_image(const char* filename,nSize matSize);// 读入图像

void save_Img(ImgArr imgarr,char* filedir); // 将图像数据保存成文件
char intTochar(int i);// 将数字转换成字符串
int charToInt(char i);// 将字符串转换成数字

char * combine_strings(char *a, char *b);
float** transform_data(float ** inputData,nSize outputSize,int location);//    对图片进行转换、裁剪

#endif