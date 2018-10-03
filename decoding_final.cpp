#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#include <opencv2/opencv.hpp>   
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>  

using namespace cv;



#define PI 3.14159265359

typedef struct {
	int r, g, b;
}int_rgb;

struct encodingResult { // 인코딩 결과값
	int avg;
	double alpha;
	int geo;
	int x, y;
	int error;
};
struct OUTPUT {
	int x_best; //최적의 x좌표
	int y_best; //최적의 y좌표
	int error; // 최적일때의 에러
	int step; // 최적일때의 스텝
	int block_mean; // 최적일때의 평균
	int temp;
	double alpha; // 알파값 
	int avg; // 평균값
	int size;
	int flag;

	OUTPUT *sub = NULL;

};

struct INPUT { // 원본이미지의 정보
	int width;
	int height;
	int** img_in;
	int** img_tmp;
	int** img_result;
	int size;
};

void search_Ouput(OUTPUT A)
{

	if (A.flag == 0) {
		printf("%d %d %d %d %lf %d\n", A.x_best, A.y_best, A.step, A.avg, A.alpha, A.flag);
		printf("--------------------------------------------------\n");
		return;
	}
	printf("%d %d %d %d %lf %d\n", A.x_best, A.y_best, A.step, A.avg, A.alpha, A.flag);

	for (int i = 0; i < 4; i++) {
		if (A.sub[i].flag == 0)
			printf("%d %d %d %d %lf %d\n", A.sub[i].x_best, A.sub[i].y_best, A.sub[i].step, A.sub[i].avg, A.sub[i].alpha, A.sub[i].flag);
		else {
			printf("%d %d %d %d %lf %d\n", A.sub[i].x_best, A.sub[i].y_best, A.sub[i].step, A.sub[i].avg, A.sub[i].alpha, A.sub[i].flag);

			for (int j = 0; j < 4; j++) {
				printf("%d %d %d %d %lf %d\n", A.sub[i].sub[j].x_best, A.sub[i].sub[j].y_best, A.sub[i].sub[j].step, A.sub[i].sub[j].avg, A.sub[i].sub[j].alpha, A.sub[i].sub[j].flag);
			}

		}

	}
	printf("--------------------------------------------------\n");


}


int** IntAlloc2(int width, int height)
{
	int** tmp;
	tmp = (int**)calloc(height, sizeof(int*));
	for (int i = 0; i<height; i++)
		tmp[i] = (int*)calloc(width, sizeof(int));
	return(tmp);
}

void IntFree2(int** image, int width, int height)
{
	for (int i = 0; i<height; i++)
		free(image[i]);

	free(image);
}

int_rgb** IntColorAlloc2(int width, int height)
{
	int_rgb** tmp;
	tmp = (int_rgb**)calloc(height, sizeof(int_rgb*));
	for (int i = 0; i<height; i++)
		tmp[i] = (int_rgb*)calloc(width, sizeof(int_rgb));
	return(tmp);
}

void IntColorFree2(int_rgb** image, int width, int height)
{
	for (int i = 0; i<height; i++)
		free(image[i]);

	free(image);
}

int** ReadImage(char* name, int* width, int* height)
{
	Mat img = imread(name, IMREAD_GRAYSCALE);
	int** image = (int**)IntAlloc2(img.cols, img.rows);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i<img.rows; i++)
		for (int j = 0; j<img.cols; j++)
			image[i][j] = img.at<unsigned char>(i, j);

	return(image);
}

void WriteImage(char* name, int** image, int width, int height)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];

	imwrite(name, img);
}


void ImageShow(char* winname, int** image, int width, int height)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];
	imshow(winname, img);
	waitKey(0);
}



int_rgb** ReadColorImage(char* name, int* width, int* height)
{
	Mat img = imread(name, IMREAD_COLOR);
	int_rgb** image = (int_rgb**)IntColorAlloc2(img.cols, img.rows);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i<img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			image[i][j].b = img.at<Vec3b>(i, j)[0];
			image[i][j].g = img.at<Vec3b>(i, j)[1];
			image[i][j].r = img.at<Vec3b>(i, j)[2];
		}

	return(image);
}

void WriteColorImage(char* name, int_rgb** image, int width, int height)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i<height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}

	imwrite(name, img);
}

void ColorImageShow(char* winname, int_rgb** image, int width, int height)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}
	imshow(winname, img);

}

template <typename _TP>
void ConnectedComponentLabeling(_TP** seg, int width, int height, int** label, int* no_label)
{

	//Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
	Mat bw(height, width, CV_8U);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			bw.at<unsigned char>(i, j) = (unsigned char)seg[i][j];
	}
	Mat labelImage(bw.size(), CV_32S);
	*no_label = connectedComponents(bw, labelImage, 8); // 0까지 포함된 갯수임

	(*no_label)--;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			label[i][j] = labelImage.at<int>(i, j);
	}
}


void half_reduction(int** in, int width, int height, int**img_out) // 크기를 반으로 줄임
{
	for (int a = 0; a < height / 2; a++) {
		for (int b = 0; b < width / 2; b++) {
			img_out[a][b] = 0;
		}

	}

	for (int a = 0; a < height; a = a + 2) {
		for (int b = 0; b < width; b = b + 2) {
			img_out[a / 2][b / 2] += (in[a][b] + in[a + 1][b] + in[a][b + 1] + in[a + 1][b + 1]) / 4;
		}

	}
	//ImageShow("출력", img_out, width / 2, height / 2);
}



void isoM_1(int** img_in, int** img_out, int width, int height) { // identity

	for (int a = 0; a < height; a++) {
		for (int b = 0; b < width; b++) {
			img_out[a][b] = img_in[a][b];
		}

	}
}

void isoM_2(int** img_in, int** img_out, int width, int height) { // reflection about mid-vertical

	for (int a = 0; a < height; a++) {
		for (int b = 0; b < width; b++) {
			img_out[a][b] = img_in[height - a - 1][b];
		}

	}

}

void isoM_3(int** img_in, int** img_out, int width, int height) { // reflection about mid-horizontal

	for (int a = 0; a < height; a++) {
		for (int b = 0; b < width; b++) {
			img_out[a][b] = img_in[a][width - 1 - b];
		}

	}

}

void isoM_4(int** in, int** img_out, int width, int height) // reflection about first diagonal
{
	//int** img_out = (int**)IntAlloc2(height, width);
	for (int a = 0; a < height; a++) {
		for (int b = 0; b < width; b++) {
			img_out[b][a] = in[a][b];
		}

	}
	//ImageShow("출력4", img_out, height, width);
}

void isoM_5(int** img_in, int** img_out, int width, int height) // reflecttion about second diagonal
{
	//int** img_out = (int**)IntAlloc2(height, width);
	for (int a = 0; a < height; a++) {
		for (int b = 0; b < width; b++) {
			img_out[b][a] = img_in[height - 1 - a][width - 1 - b];
		}

	}
	//ImageShow("출력5", img_out, height, width);
}

void isoM_6(int** img_in, int** img_out, int width, int height) // Rotation around center of block, through +90
{
	//int** img_out = (int**)IntAlloc2(height, width);
	for (int a = 0; a < height; a++) {
		for (int b = 0; b < width; b++) {
			img_out[b][a] = img_in[a][width - 1 - b];
		}

	}
	//ImageShow("출력6", img_out, height, width);
}
void isoM_7(int** img_in, int** img_out, int width, int height) // Rotation around center of block, through +180
{
	//int** img_out = (int**)IntAlloc2(width, height);
	for (int a = 0; a < height; a++) {
		for (int b = 0; b < width; b++) {
			img_out[a][b] = img_in[height - a - 1][width - b - 1];
		}

	}
	//ImageShow("출력7", img_out, width, height);
}

void isoM_8(int** img_in, int** img_out, int width, int height) // Rotation around center of block, through -90
{
	//int** img_out = (int**)IntAlloc2(width, height);
	for (int a = 0; a < height; a++) {
		for (int b = 0; b < width; b++) {
			img_out[a][b] = img_in[height - a - 1][b];
		}

	}
	//ImageShow("출력8", img_out, width, height);
}
void isometry(int no, int** img_in, int** img_out, int width, int height) { //회전 모음
	switch (no)
	{
	case 1:
		isoM_1(img_in, img_out, width, height);
		break;
	case 2:
		isoM_2(img_in, img_out, width, height);
		break;
	case 3:
		isoM_3(img_in, img_out, width, height);
		break;
	case 4:
		isoM_4(img_in, img_out, width, height);
		break;
	case 5:
		isoM_5(img_in, img_out, width, height);
		break;
	case 6:
		isoM_6(img_in, img_out, width, height);
		break;
	case 7:
		isoM_7(img_in, img_out, width, height);
		break;
	case 8:
		isoM_8(img_in, img_out, width, height);
		break;

	default:
		break;
	}

}
void ReadBlock(int** image, int y, int x, int dx, int dy, int** block) // 블록을 잘라옴
{
	for (int j = 0; j<dy; j++)
		for (int i = 0; i < dx; i++)
		{
			block[j][i] = image[y + j][x + i];
		}


}

void WriteBlock(int** image, int y, int x, int dx, int dy, int** block) // 블록에 씀
{
	for (int i = 0; i<dx; i++)
		for (int j = 0; j < dy; j++)
		{
			image[y + i][x + j] = block[i][j];
		}

}

float AVG(int** image, int width, int height) // 평균구하기
{
	float avg = 0;
	for (int i = 0; i<width; i++)
		for (int j = 0; j < height; j++)
		{
			avg += image[j][i];
		}
	return avg / (width*height);


}

int Compute_Error_image(int** block, int** image, int size_block, int i, int j) // 이미지에서 좌표를 지정하여 블록과의 에러 계산
{
	int error = 0;
	for (int y = 0; y < size_block; y++) {
		for (int x = 0; x < size_block; x++)
		{
			error += abs(block[y][x] - image[y + i][x + j]);

		}
	}
	return error;
}

int Compute_Error_block(int** block, int** block2, int size_block) // 두개의 블록을 비교하여 에러계산
{
	int error = 0;
	for (int y = 0; y < size_block; y++) {
		for (int x = 0; x < size_block; x++)
		{
			error += abs(block[y][x] - block2[y][x]);

		}
	}
	return error;
}






void getACblock(int** block1, int dx, int dy, int** block_AC)
{
	int avg = (int)AVG(block1, dx, dy);
	for (int j = 0; j < dy; j++)
		for (int i = 0; i < dx; i++)
		{
			block_AC[j][i] = block1[j][i] - avg;
		}
}

void plusACblock(int** block1, int dx, int dy, int** block_AC, int avg)
{

	for (int j = 0; j < dy; j++)
		for (int i = 0; i < dx; i++)
		{
			block_AC[j][i] = block1[j][i] + avg;
		}




}



void alpha(int** img_in, int** img_out, int dx, int dy, double a) // 들어온 블럭의 평균을 계산해서 빼준다음 알파를 곱함
{
	int avg = (int)AVG(img_in, dx, dy);
	for (int j = 0; j < dy; j++)
		for (int i = 0; i < dx; i++)
		{
			img_out[j][i] = a*(img_in[j][i] - avg);
		}

}

void alpha2(int** img_in, int** img_out, int dx, int dy, double a, int avg) //받은 평균으로 빼주고 알파를 곱함
{

	for (int j = 0; j < dy; j++)
		for (int i = 0; i < dx; i++)
		{
			img_out[j][i] = a*(img_in[j][i] - avg);
		}

}
void alpha_plus(int** img_in, int** img_out, int dx, int dy, int avg) // 평균을 더해줌
{


	for (int j = 0; j < dy; j++)
		for (int i = 0; i < dx; i++)
		{
			img_out[j][i] = img_in[j][i] + avg;
		}
}

void alpha_ninus(int** img_in, int** img_out, int dx, int dy) // 평균을 빼줌
{
	int avg = (int)AVG(img_in, dx, dy);
	for (int j = 0; j < dy; j++)
		for (int i = 0; i < dx; i++)
		{
			img_out[j][i] = img_in[j][i] - avg;
		}
}



OUTPUT** ER_Alloc2(int width, int height) // 할당
{
	OUTPUT** tmp;
	tmp = (OUTPUT**)calloc(height, sizeof(int*));
	for (int i = 0; i<height; i++)
		tmp[i] = (OUTPUT*)calloc(width, sizeof(OUTPUT));
	return(tmp);
}

void ER_Free2(encodingResult** image, int width, int height) { //메모리 해제
	for (int i = 0; i<height; i++)
		free(image[i]);

	free(image);
}




bool ReadPrameter(char* name, OUTPUT** A, int width, int height, int size) { // 파일을 읽는 함수
	FILE *fp = fopen(name, "r");
	if (fp == NULL) {
		printf("\n Failure in fopen!");
		return(false);
	}
	for (int y = 0; y<height / size; y++)
		for (int x = 0; x < width / size; x++) {


			//fscanf(fp, "%d%d%d%d%lf%d%d", &(A[y][x].x_best), &(A[y][x].y_best), &(A[y][x].step), &(A[y][x].avg), &(A[y][x].alpha), &(A[y][x].size),&(A[y][x].flag));
			fscanf(fp, "%d%d%d%d%lf%d", &(A[y][x].x_best), &(A[y][x].y_best), &(A[y][x].step), &(A[y][x].avg), &(A[y][x].alpha), &(A[y][x].flag));
			if (A[y][x].flag == 0)
				continue;


			// fscanf(fp, "%d %d %d %d %lf %d %d", &(A[y][x].x_best), &(A[y][x].y_best), &(A[y][x].step), &(A[y][x].avg), &(A[y][x].alpha), &(A[y][x].size), &(A[y][x].flag));

			A[y][x].sub = (OUTPUT*)calloc(4, sizeof(OUTPUT));

			for (int i = 0; i < 4; i++) {
				fscanf(fp, "%d%d%d%d%lf%d", &(A[y][x].sub[i].x_best), &(A[y][x].sub[i].y_best), &(A[y][x].sub[i].step), &(A[y][x].sub[i].avg), &(A[y][x].sub[i].alpha), &(A[y][x].sub[i].flag));
				if (A[y][x].sub[i].flag == 1)
				{

					A[y][x].sub[i].sub = (OUTPUT*)calloc(4, sizeof(OUTPUT));
					for (int j = 0; j < 4; j++) {
						fscanf(fp, "%d%d%d%d%lf%d", &(A[y][x].sub[i].sub[j].x_best), &(A[y][x].sub[i].sub[j].y_best), &(A[y][x].sub[i].sub[j].step), &(A[y][x].sub[i].sub[j].avg), &(A[y][x].sub[i].sub[j].alpha), &(A[y][x].sub[i].sub[j].flag));
					}

				}

			}


		}
	fclose(fp);
	return (true);
}
void Write_Gray(int** img, int width, int height) // 회색으로 만드는 함수
{
	for (int y = 0; y<height; y++)
		for (int x = 0; x < width; x++)
		{
			img[y][x] = 128;
		}
}

void Write_white(int** img, int width, int height) // 횐색으로 만드는 함수
{
	for (int y = 0; y<height; y++)
		for (int x = 0; x < width; x++)
		{
			img[y][x] = 255;
		}
}

void Real_Decoding(OUTPUT* t, int width, int height, INPUT *img_dec, int i, int j, int size)
{
	
	if (t->flag == 0) {
		//printf("%d %d %d %d %lf %d\n", t->x_best, t->y_best, t->step, t->avg, t->alpha, t->flag);
		int **block_ori = IntAlloc2(size * 2, size * 2); // 2N 블록을 담을 2차원배열
		int **block_half = IntAlloc2(size, size); // 2N-> N 블록을 담을 2차원배열
		int **block_AC = IntAlloc2(size, size); // 평균빼고 알파 곱한 블록을 담을 2차원배열
		int **block_geo = IntAlloc2(size, size); // 돌린걸 담을 2차원 배열
		int **block_plus = IntAlloc2(size, size); // 평균을 더한블록을 담을 배열



		ReadBlock(img_dec->img_in, t->y_best, t->x_best, 2 * size, 2 * size, block_ori); // 인코딩 정보에 있는 x,y좌표에서 2N블록을 자름
		half_reduction(block_ori, 2 * size, 2 * size, block_half); // 자른걸 다시 N블록으로 축소

		alpha(block_half, block_AC, size, size, t->alpha); // 평균빼고 알파를 곱함
		isometry(t->step, block_AC, block_geo, size, size); // 인코딩정보에 있는 geo값을 토대로 돌림

		plusACblock(block_geo, size, size, block_plus, t->avg); // 인코딩정보에있는 평균값을 더함

		WriteBlock(img_dec->img_tmp, i, j, size, size, block_plus);

	}
	
	else if (t->flag == 1)
	{

		Real_Decoding(&t->sub[0], width, height, img_dec, i, j, size/2);
		Real_Decoding(&t->sub[1], width, height, img_dec, i, j + size/2, size/2);
		Real_Decoding(&t->sub[2], width, height, img_dec, i + size/2, j, size/2);
		Real_Decoding(&t->sub[3], width, height, img_dec, i + size/2, j + size/2, size/2);

	}
	
}

void Decoding(OUTPUT** t, int width, int height, INPUT* img_dec)
{
	for (int i = 0; i < img_dec->height; i += img_dec->size)
		for (int j = 0; j < img_dec->width; j += img_dec->size)
		{


			Real_Decoding(&t[i / img_dec->size][j / img_dec->size], width, height, img_dec, i, j, img_dec->size);


		}
	WriteBlock(img_dec->img_in, 0, 0, img_dec->height, img_dec->width, img_dec->img_tmp); //임시공간에 있던 디코딩 결과를 다시 원래 사진에 옮김으로써 계속 중첩되게함
	
}

void Draw_block(OUTPUT* t, int width, int height, INPUT *img_dec, int y, int x, int size)
{
	
	if (t->flag == 0) {
		for (int i = 0; i < size; i++)
		{
			img_dec->img_in[y][i + x] = 0;
		}

		for (int i = 0; i < size; i++)
		{
			img_dec->img_in[size - 1 + y][i + x] = 0;
		}

		for (int j = 0; j < +size; j++)
		{
			img_dec->img_in[j + y][x + size - 1] = 0;
		}

		for (int j = 0; j < +size; j++)
		{
			img_dec->img_in[j + y][x] = 0;
		}
	}
	else if (t->flag == 1)
	{

		Draw_block(&t->sub[0], width, height, img_dec, y, x, size/2);
		Draw_block(&t->sub[1], width, height, img_dec, y, x + size/2, size/2);
		Draw_block(&t->sub[2], width, height, img_dec, y + size/2, x, size/2);
		Draw_block(&t->sub[3], width, height, img_dec, y + size/2, x + size/2, size/2);

	}

}

void Quad_Draw(OUTPUT** t, int width, int height, INPUT* img_dec)
{
	for (int i = 0; i < img_dec->height; i += img_dec->size)
		for (int j = 0; j < img_dec->width; j += img_dec->size)
		{


			Draw_block(&t[i / img_dec->size][j / img_dec->size], width, height, img_dec, i, j, img_dec->size);


		}
}

double computePSNR(int** A, int** B, int width, int height)
{
	double error = 0.0;
	for(int i=0; i<height; i++)
		for (int j = 0; j < width; j++)
		{
			error += (double)(A[i][j] - B[i][j]) * (A[i][j] - B[i][j]);
		}
	error = error / (width * height);
	double psnr = 10.0*log10(255.*255. / error);
	return psnr;
}





int main(void)
{
	INPUT input; // 처음사진의 정보를 담고있는 구조체
	INPUT psnr;
	input.img_in = ReadImage("lena.bmp", &input.width, &input.height); // 원본사진
	psnr.img_in = ReadImage("lena.bmp", &input.width, &input.height);
	input.size = 16;
	input.img_tmp = IntAlloc2(input.width, input.height);
	double PSNR = 0;
	OUTPUT** Result; // 인코딩 결과를 담을 구조체
	Result = ER_Alloc2(input.width / input.size, input.height / input.size); // 2차원으로 공간 할당
	



	ReadPrameter("encoding5.txt", Result, input.width, input.height, input.size); // 파일 읽기

	for (int i = 0; i < input.height / input.size; i++) //파일이 잘 읽혔는지 출력해봄
		for (int j = 0; j < input.width / input.size; j++) {

			search_Ouput(Result[i][j]);

		}
	
	Write_white(input.img_in, input.width, input.height);
	Quad_Draw(Result, input.width, input.height, &input);
	ImageShow("결과물1", input.img_in, input.width, input.height); //블록 분할 영상 출력
										  
	Write_Gray(input.img_in, input.width, input.height); // 회색 사진 만들기
	for (int i = 0; i < 10; i++) {
	  Decoding(Result, input.width, input.height, &input); // 디코딩 5번 반복
	  PSNR =computePSNR(psnr.img_in, input.img_in, input.width, input.height);
	  printf("PSNR 값 : %f\n", PSNR);
	  ImageShow("복원영상", input.img_in, input.width, input.height); //출력
	}
													  
	return 0;
}
