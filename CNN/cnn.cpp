#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "cnn.h"
#include "assert.h"
#include "iostream"

// 激活函数 input是数据，inputNum说明数据数目，bas表明偏置
float activation_Sigma(float input,float bas) // sigma激活函数
{
	float temp=input+bas;
	return (float)1.0/((float)(1.0+exp(-temp)));
}

// 激活函数 input是数据，inputNum说明数据数目，bas表明偏置
float activation_Tanh(float input,float bas) // sigma激活函数
{
	float temp=input+bas;
    return (float) tanh(temp);
}

void avgPooling(float** output,nSize outputSize,float** input,nSize inputSize,int mapSize) // 求平均值
{
	int outputW=inputSize.c/mapSize;
	int outputH=inputSize.r/mapSize;
	if(outputSize.c!=outputW||outputSize.r!=outputH)
		printf("ERROR: output size is wrong!!");

	int i,j,m,n;
	for(i=0;i<outputH;i++)
		for(j=0;j<outputW;j++)
		{
			float sum=0.0;
			for(m=i*mapSize;m<i*mapSize+mapSize;m++)
				for(n=j*mapSize;n<j*mapSize+mapSize;n++)
					sum=sum+input[m][n];

			output[i][j]=sum/(float)(mapSize*mapSize);
		}
}
void maxPooling(float** output,nSize outputSize,float** input,nSize inputSize,nSize mapSize) // 求最大值
{
	int outputW=inputSize.c/mapSize.c;
	int outputH=inputSize.r/mapSize.r;
	if(outputSize.c!=outputW||outputSize.r!=outputH)
		printf("ERROR: output size is wrong!!");

	int i,j,m,n;
	for(i=0;i<outputH;i++)
		for(j=0;j<outputW;j++)
		{
			float max_value=INT32_MIN;
			for(m=i*mapSize.r;m<i*mapSize.r+mapSize.r;m++)
				for(n=j*mapSize.c;n<j*mapSize.c+mapSize.c;n++)
                    if (input[m][n] > max_value){
                        max_value = input[m][n];
                    }

			output[i][j]=max_value;
		}
}

// 单层全连接神经网络的前向传播
float vecMulti(float* vec1,float* vec2,int vecL)// 两向量相乘
{
	int i;
	float m=0;
	for(i=0;i<vecL;i++)
		m=m+vec1[i]*vec2[i];
	return m;
}
//全连接层矩阵运算
void nnff(float* output,float* input,float** wdata,float* bas,nSize nnSize,int activationType)
{
	int w=nnSize.c;
	int h=nnSize.r;
	
	int i;
	for(i=0;i<h;i++){
        output[i]=vecMulti(input,wdata[i],w);
        if(activationType==Tanh){
            output[i] = activation_Tanh(output[i],bas[i]);
        }
        else if(activationType==Sigmoid){
            output[i] = activation_Sigma(output[i],bas[i]);
        }else{
            output[i] = output[i] +bas[i];
        }

    }
}

float sigma_derivation(float y){ // Logic激活函数的自变量微分
	return y*(1-y); // 这里y是指经过激活函数的输出值，而不是自变量
}


PoolLayer* set_pool_layer(nSize inSize,int num_channels, nSize mapSize,int poolType)
{
	PoolLayer* poolL=(PoolLayer*)malloc(sizeof(PoolLayer));
	poolL->inputSize=inSize;
	poolL->mapSize=mapSize;
	poolL->poolType=poolType;
	poolL->outChannels = num_channels;
	poolL->inChannels = num_channels;

	int output_row = inSize.r/mapSize.r;
	int output_col=inSize.c/mapSize.c;
	poolL->outputSize={output_row,output_col};

	poolL->y=(float***)malloc(num_channels*sizeof(float**));

	for(int j=0;j<num_channels;j++){
		poolL->y[j]=(float**)malloc(output_row*sizeof(float*));
		for(int r=0;r<output_row;r++){
//            分配内存，并初始化为0
			poolL->y[j][r]=(float*)calloc(output_col,sizeof(float));
		}
	}

	return poolL;
}

ConvPool load_conv_weight(FILE * fp, nSize inputSize) {
//    加载卷积层权重：W和b
//    char * file_name = combine_strings(project_path, (char *) "model/model_weights.mat");
//    FILE * fp = fopen(file_name,"rb");
	if(fp==NULL)
		printf("open file failed\n");
	assert(fp);

	CovLayer* covL=(CovLayer*)malloc(sizeof(CovLayer));

//    covL->inputHeight=inputHeight;
//    covL->inputWidth=inputWidth;

	int *conv1_W_shape= (int *) malloc(4 * sizeof(int));
	//从文件中读取sizeof(int) 个到 conv1_W_shape
	fread(conv1_W_shape,sizeof(int),4,fp);

//    std::cout << conv1_W_shape[0] << "," << conv1_W_shape[1] << "," << conv1_W_shape[2] << "," << conv1_W_shape[3] << std::endl;
	covL->inputSize=inputSize;
	covL->outChannels=conv1_W_shape[0];
	covL->inChannels=conv1_W_shape[1];
	covL->mapSize={conv1_W_shape[2],conv1_W_shape[3]};
	covL->isFullConnect=true; // 默认为全连接

	//输出大小
	int output_row = inputSize.r-covL->mapSize.r+1;
	int output_col = inputSize.c-covL->mapSize.c+1;
	covL->outputSize = {output_row,output_col};

	float conv_W_filter[conv1_W_shape[0]][conv1_W_shape[1]][conv1_W_shape[2]][conv1_W_shape[3]];
	covL->mapData = (float ****) malloc(sizeof(float***) * conv1_W_shape[0]);

	for(int num_filter=0;num_filter<conv1_W_shape[0];num_filter++){
		covL->mapData[num_filter] = (float ***) malloc(sizeof(float**) * conv1_W_shape[1]);

		for(int num_channel=0;num_channel<conv1_W_shape[1];num_channel++) {

			covL->mapData[num_filter][num_channel] = (float **) malloc(sizeof(float*) * conv1_W_shape[2]);

			for(int filter_row=0;filter_row<conv1_W_shape[2];filter_row++) {

				covL->mapData[num_filter][num_channel][filter_row] = (float *) malloc(sizeof(float) * conv1_W_shape[3]);

				for(int filter_col=0;filter_col<conv1_W_shape[2];filter_col++) {

					float *value= (float *) malloc(sizeof(float));
					fread(value,sizeof(float),1,fp);
//                    std::cout<<*value<<std::endl;
					conv_W_filter[num_filter][num_channel][filter_row][filter_col] = *value;
					covL->mapData[num_filter][num_channel][filter_row][filter_col] = *value;
					free(value);
				}
			}

		}
	}

//    t( covL->mapData[0][0]);
//    assert(NULL);
//    covL->mapData = (float ****) conv_W_filter;

	// bias
	int *conv1_b_shape= (int *) malloc(1 * sizeof(int));
	fread(conv1_b_shape,sizeof(int),1,fp);
//	std::cout << *conv1_b_shape << std::endl;
	covL->basicData  = (float *) malloc(sizeof(float) * conv1_W_shape[0]);
	for(int num_filter=0;num_filter<conv1_W_shape[0];num_filter++){
		float *value= (float *) malloc(sizeof(float));
		fread(value,sizeof(float),1,fp);
//        std::cout << *value << std::endl;
		covL->basicData[num_filter] = *value;
		free(value);
	}


//为静输入、激发函数输出等分配内存空间
//    covL->d=(float***)malloc(covL->outChannels*sizeof(float**));
	covL->v=(float***)malloc(covL->outChannels*sizeof(float**));
	covL->y=(float***)malloc(covL->outChannels*sizeof(float**));
	for(int j=0;j<covL->outChannels;j++){
//        covL->d[j]=(float**)malloc(output_row*sizeof(float*));
		covL->v[j]=(float**)malloc(output_row*sizeof(float*));
		covL->y[j]=(float**)malloc(output_row*sizeof(float*));
		for(int r=0;r<output_row;r++){
//            covL->d[j][r]=(float*)calloc(output_col,sizeof(float));
			covL->v[j][r]=(float*)calloc(output_col,sizeof(float));
			covL->y[j][r]=(float*)calloc(output_col,sizeof(float));
		}
	}


//    for(int i=0;i<25;i++){
//        std::cout<<covL->basicData[i];
//    }
	ConvPool convPool = {covL,set_pool_layer(covL->outputSize,covL->outChannels,{2,2},MaxPool)};

	return convPool;
}

Dense * load_dense_weight(FILE * fp){
//    加载隐含层权重
	int *dense1_W_shape= (int *) malloc(2 * sizeof(int));
	//从文件中读取sizeof(int) 个到 conv1_W_shape
	fread(dense1_W_shape,sizeof(int),2,fp);

//    std::cout << dense1_W_shape[0] << "," << dense1_W_shape[1] << std::endl;

	Dense * dense1 = (Dense *) malloc(sizeof(Dense));
	dense1->inputNum=dense1_W_shape[0];
	dense1->outputNum=dense1_W_shape[1];

	// 权重的初始化
//     输出行，输入列,
	dense1->wData = (float **) malloc(dense1_W_shape[1] * sizeof(float*));

	for(int i=0;i<dense1_W_shape[1];i++) {
		dense1->wData[i] = (float *) malloc(dense1_W_shape[0] * sizeof(float));
	}

	for(int col=0;col<dense1_W_shape[0];col++){
		for(int row=0;row<dense1_W_shape[1];row++){
			float *value= (float *) malloc(sizeof(float));
			fread(value,sizeof(float),1,fp);
//            std::cout << *value << std::endl;
			dense1->wData[row][col]=*value;
			free(value);
		}
	}

	// bias
	int *dense1_b_shape= (int *) malloc(1 * sizeof(int));
	fread(dense1_b_shape,sizeof(int),1,fp);
//	std::cout << *dense1_b_shape << std::endl;


	dense1->basicData= (float *) malloc(dense1_b_shape[0] * sizeof(float));
	for(int num_filter=0;num_filter<dense1_b_shape[0];num_filter++){

		float *value= (float *) malloc(sizeof(float));
		fread(value,sizeof(float),1,fp);
//        std::cout << *value << std::endl;
		dense1->basicData[num_filter] = *value;
		free(value);
	}

	dense1->v= (float *) calloc(dense1_b_shape[0] , sizeof(float));
	dense1->y= (float *) calloc(dense1_b_shape[0] , sizeof(float));

	dense1->isFullConnect=true;

	return dense1;
}

CNN * load_cnn_weight(char * file_name,int num_labels,nSize inputSize){
//加载cnn模型和权重
	FILE * fp = fopen(file_name,"rb");
	if(fp==NULL)
		printf("open file failed:%s\n",file_name);
	assert(fp);

	// CNN结构的初始化
	CNN* cnn=(CNN*)malloc(sizeof(CNN));
    ConvPool convPool;

    cnn->num_labels = num_labels;
    cnn->inputSize = inputSize;

//第一个卷积核
	convPool = load_conv_weight(fp,inputSize);
	cnn->C11=convPool.C;
	cnn->S12=convPool.S;

//第二个卷积核
	convPool = load_conv_weight(fp,inputSize);
	cnn->C21=convPool.C;
	cnn->S22=convPool.S;

//第三个卷积核
	convPool = load_conv_weight(fp,inputSize);
	cnn->C31=convPool.C;
	cnn->S32=convPool.S;

	cnn->FC1 = load_dense_weight(fp);
	cnn->FC2 = load_dense_weight(fp);
	return cnn;
}

void conv_forward(float** inputData,ConvPool * convPool,nSize inSize){

	nSize mapSize = convPool->C->mapSize;
	nSize mapOutSize = {inSize.r-mapSize.r+1,inSize.c-mapSize.c+1};
	for(int i=0;i<(convPool->C->outChannels);i++){
		for(int j=0;j<(convPool->C->inChannels);j++){

			float** mapout=cov(convPool->C->mapData[i][j],mapSize,inputData,inSize,valid);
//            printmat(mapout,mapOutSize);
			addmat(convPool->C->v[i],convPool->C->v[i],mapOutSize,mapout,mapOutSize);

			for(int r=0;r<mapOutSize.r;r++)
				free(mapout[r]);
			free(mapout);
		}
//        printmat(convPool->C->v[i],convPool->C->outputSize);
		for(int r=0;r<mapOutSize.r;r++)
			for(int c=0;c<mapOutSize.c;c++)
				convPool->C->y[i][r][c]=activation_Tanh(convPool->C->v[i][r][c],convPool->C->basicData[i]);

//        printmat(convPool->C->y[i],mapOutSize);
//        std::cout<<convPool->C->basicData[i]<< std::endl;
//        assert(NULL);
	}

	// 第二层的输出传播S2，采样层
	nSize poolOutSize={mapOutSize.r/2,mapOutSize.c/2};
//    std::cout<<poolOutSize.r<<","<<poolOutSize.c<<std::endl;
	for(int i=0;i<(convPool->C->outChannels);i++){
		if(convPool->S->poolType==MaxPool){
			maxPooling(convPool->S->y[i],poolOutSize,convPool->C->y[i],mapOutSize,convPool->S->mapSize);
//            printmat(convPool->S->y[i],poolOutSize);
		}
	}
}


float * merge(CNN* cnn){
	int s1_number = cnn->S12->outputSize.r*cnn->S12->outputSize.c* cnn->S12->outChannels;
	int s2_number = cnn->S22->outputSize.r*cnn->S22->outputSize.c* cnn->S22->outChannels;
	int s3_number = cnn->S32->outputSize.r*cnn->S32->outputSize.c* cnn->S32->outChannels;
	int total_number = s1_number+s2_number+s3_number;
//    printf("total_number:%d\n",total_number);
	float * flatten_mat = (float *) malloc(total_number * sizeof(float));
	int count =0;
	for(int i=0;i< cnn->S12->outChannels;i++){
		for(int j = 0;j<cnn->S12->outputSize.r;j++){
			for(int k = 0;k<cnn->S12->outputSize.c;k++){
				flatten_mat[count]=cnn->S12->y[i][j][k];
				count++;
			}

		}
	}
//    std::cout<<std::endl;

	for(int i=0;i< cnn->S22->outChannels;i++){
		for(int j = 0;j<cnn->S22->outputSize.r;j++){
			for(int k = 0;k<cnn->S22->outputSize.c;k++){
//                std::cout<<cnn->S22->y[i][j][k]<<" ";
				flatten_mat[count]=cnn->S22->y[i][j][k];
				count++;
			}

		}
	}
//    std::cout<<std::endl;
	for(int i=0;i< cnn->S32->outChannels;i++){
		for(int j = 0;j<cnn->S32->outputSize.r;j++){
			for(int k = 0;k<cnn->S32->outputSize.c;k++){
//                std::cout<<cnn->S32->y[i][j][k]<<" ";
				flatten_mat[count]=cnn->S32->y[i][j][k];
				count++;
			}

		}
	}
	return flatten_mat;
}
void cnnclear(CNN* cnn)
{
    // 将神经元的部分数据清除
    int j,c,r;
    // C1网络
    for(j=0;j<cnn->C11->outChannels;j++){
        for(r=0;r<cnn->C11->outputSize.r;r++){
            for(c=0;c<cnn->C11->outputSize.c;c++){
                cnn->C11->v[j][r][c]=(float)0.0;
            }
        }
    }
    // C2网络
    for(j=0;j<cnn->C21->outChannels;j++){
        for(r=0;r<cnn->C21->outputSize.r;r++){
            for(c=0;c<cnn->C21->outputSize.c;c++){
                cnn->C21->v[j][r][c]=(float)0.0;
            }
        }
    }
    // C3网络
    for(j=0;j<cnn->C31->outChannels;j++){
        for(r=0;r<cnn->C31->outputSize.r;r++){
            for(c=0;c<cnn->C31->outputSize.c;c++){
                cnn->C31->v[j][r][c]=(float)0.0;
            }
        }
    }


}

int get_max_index(float* array,int array_length){
//    返回数组的最大值索引
	float max_value=INT32_MIN;
	int max_value_index=0;
	for(int i=0;i<array_length;i++){
		if( array[i]>max_value){
			max_value=array[i];
			max_value_index=i;
		}
	}
	return max_value_index;
}

int forward(CNN* cnn, float** inputData){

//    cnn前向传播
	ConvPool convPool = {cnn->C11,cnn->S12};
	conv_forward(inputData,&convPool,cnn->inputSize);

//    printmat(cnn->C11->y[0],{4,11});

	convPool = {cnn->C21,cnn->S22};
	conv_forward(inputData,&convPool,cnn->inputSize);
//    printmat(cnn->C21->v[0],{6,13});

//    printmat(cnn->C21->mapData[0][0],{3,3});

	convPool = {cnn->C31,cnn->S32};
	conv_forward(inputData,&convPool,cnn->inputSize);
//    平铺
	float* flatten_mat = merge(cnn);
//    for(int i=0;i<100;i++){
//        printf("%f ",flatten_mat[i]);
//    }
//    printf("\n");
    float *dense1_output = (float *) malloc(cnn->FC1->outputNum * sizeof(float));
	nnff(dense1_output,
		 flatten_mat,
		 cnn->FC1->wData,
		 cnn->FC1->basicData,
		 {cnn->FC1->outputNum,cnn->FC1->inputNum},Tanh
	);

	free(flatten_mat);
	float *dense2_output = (float *) malloc(cnn->FC2->outputNum * sizeof(float));

	nnff(dense2_output,
		 dense1_output,
		 cnn->FC2->wData,
		 cnn->FC2->basicData,
		 {cnn->FC2->outputNum,cnn->FC2->inputNum},Linear
	);
	int result = get_max_index(dense2_output,cnn->num_labels);
// 打印输出层
//    for(int j=0;j<cnn->FC2->outputNum;j++){
//        std::cout<<dense2_output[j]<<" ";
//    }
//    std::cout<<std::endl;
    cnnclear(cnn);
    free(dense1_output);
	free(dense2_output);
	return result;
}

CharClassifier * load_all_model_weight(char * project_path){
//加载所有模型权重
    CharClassifier * model = (CharClassifier *) malloc(sizeof(CharClassifier));
    char * file_name;

    file_name = combine_strings(project_path, (char *) "model/model_all_weight.mat");
    model->model_all = load_cnn_weight(file_name,34,{15,15});

    file_name = combine_strings(project_path, (char *) "model/model_0D_weight.mat");
    model->model_binary_0D = load_cnn_weight(file_name,2,{15,8});

    file_name = combine_strings(project_path, (char *) "model/model_1I_weight.mat");
    model->model_binary_1I = load_cnn_weight(file_name,2,{8,15});

    file_name = combine_strings(project_path, (char *) "model/model_2Z_weight.mat");
    model->model_binary_2Z = load_cnn_weight(file_name,2,{8,15});

    file_name = combine_strings(project_path, (char *) "model/model_4A_weight.mat");
    model->model_binary_4A = load_cnn_weight(file_name,2,{4,5});

    file_name = combine_strings(project_path, (char *) "model/model_8B_weight.mat");
    model->model_binary_8B = load_cnn_weight(file_name,2,{15,8});

    file_name = combine_strings(project_path, (char *) "model/model_56_weight.mat");
    model->model_binary_56 = load_cnn_weight(file_name,2,{8,8});

    return model;
}



int predict(CharClassifier * model, float** inputData){
    int result = forward(model->model_all,inputData);
//    int result = 1;

    if(intTochar(result)=='0' || intTochar(result)=='D'){
        float ** temp = transform_data(inputData,model->model_binary_0D->inputSize,TOP_LEFT);
        result = forward(model->model_binary_0D,temp);
        free(temp);
        char* charset = (char *) "0D";
        result = charToInt(charset[result]);
    }

    if(intTochar(result)=='1' || intTochar(result)=='I'){
//        上边
        float ** temp = transform_data(inputData,model->model_binary_1I->inputSize,TOP_LEFT);
        char* charset = (char *) "1I";
//        assert(NULL);
        result = forward(model->model_binary_1I,temp);
        result = charToInt(charset[result]);
    }
    if(intTochar(result)=='2' || intTochar(result)=='Z'){
        float ** temp = transform_data(inputData,model->model_binary_2Z->inputSize,TOP_LEFT);
        char* charset = (char *) "2Z";
        result = forward(model->model_binary_2Z,temp);
        result = charToInt(charset[result]);
    }

    if(intTochar(result)=='4' || intTochar(result)=='A'){
        float ** temp = transform_data(inputData,model->model_binary_4A->inputSize,BOTTOM_LEFT);
        char* charset = (char *) "4A";
        result = forward(model->model_binary_4A,temp);
        result = charToInt(charset[result]);
    }

    if(intTochar(result)=='8' || intTochar(result)=='B'){
        float ** temp = transform_data(inputData,model->model_binary_8B->inputSize,TOP_LEFT);
        char* charset = (char *) "8B";
        result = forward(model->model_binary_8B,temp);
        result = charToInt(charset[result]);
    }
    if(intTochar(result)=='5' || intTochar(result)=='6'){
        float ** temp = transform_data(inputData,model->model_binary_56->inputSize,BOTTOM_LEFT);
        char* charset = (char *) "56";
        result = forward(model->model_binary_56,temp);
        result = charToInt(charset[result]);
    }
    return result;

}
