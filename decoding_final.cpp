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

struct encodingResult { // ���ڵ� �����
	int avg;
	double alpha;
	int geo;
	int x, y;
	int error;
};
struct OUTPUT {
	int x_best; //������ x��ǥ
	int y_best; //������ y��ǥ
	int error; // �����϶��� ����
	int step; // �����϶��� ����
	int block_mean; // �����϶��� ���
	int temp;
	double alpha; // ���İ� 
	int avg; // ��հ�
	int size;
	int flag;

	OUTPUT *sub = NULL;

};

struct INPUT { // �����̹����� ����
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
	*no_label = connectedComponents(bw, labelImage, 8); // 0���� ���Ե� ������

	(*no_label)--;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			label[i][j] = labelImage.at<int>(i, j);
	}
}


void half_reduction(int** in, int width, int height, int**img_out) // ũ�⸦ ������ ����
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
	//ImageShow("���", img_out, width / 2, height / 2);
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
	//ImageShow("���4", img_out, height, width);
}

void isoM_5(int** img_in, int** img_out, int width, int height) // reflecttion about second diagonal
{
	//int** img_out = (int**)IntAlloc2(height, width);
	for (int a = 0; a < height; a++) {
		for (int b = 0; b < width; b++) {
			img_out[b][a] = img_in[height - 1 - a][width - 1 - b];
		}

	}
	//ImageShow("���5", img_out, height, width);
}

void isoM_6(int** img_in, int** img_out, int width, int height) // Rotation around center of block, through +90
{
	//int** img_out = (int**)IntAlloc2(height, width);
	for (int a = 0; a < height; a++) {
		for (int b = 0; b < width; b++) {
			img_out[b][a] = img_in[a][width - 1 - b];
		}

	}
	//ImageShow("���6", img_out, height, width);
}
void isoM_7(int** img_in, int** img_out, int width, int height) // Rotation around center of block, through +180
{
	//int** img_out = (int**)IntAlloc2(width, height);
	for (int a = 0; a < height; a++) {
		for (int b = 0; b < width; b++) {
			img_out[a][b] = img_in[height - a - 1][width - b - 1];
		}

	}
	//ImageShow("���7", img_out, width, height);
}

void isoM_8(int** img_in, int** img_out, int width, int height) // Rotation around center of block, through -90
{
	//int** img_out = (int**)IntAlloc2(width, height);
	for (int a = 0; a < height; a++) {
		for (int b = 0; b < width; b++) {
			img_out[a][b] = img_in[height - a - 1][b];
		}

	}
	//ImageShow("���8", img_out, width, height);
}
void isometry(int no, int** img_in, int** img_out, int width, int height) { //ȸ�� ����
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
void ReadBlock(int** image, int y, int x, int dx, int dy, int** block) // ����� �߶��
{
	for (int j = 0; j<dy; j++)
		for (int i = 0; i < dx; i++)
		{
			block[j][i] = image[y + j][x + i];
		}


}

void WriteBlock(int** image, int y, int x, int dx, int dy, int** block) // ��Ͽ� ��
{
	for (int i = 0; i<dx; i++)
		for (int j = 0; j < dy; j++)
		{
			image[y + i][x + j] = block[i][j];
		}

}

float AVG(int** image, int width, int height) // ��ձ��ϱ�
{
	float avg = 0;
	for (int i = 0; i<width; i++)
		for (int j = 0; j < height; j++)
		{
			avg += image[j][i];
		}
	return avg / (width*height);


}

int Compute_Error_image(int** block, int** image, int size_block, int i, int j) // �̹������� ��ǥ�� �����Ͽ� ��ϰ��� ���� ���
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

int Compute_Error_block(int** block, int** block2, int size_block) // �ΰ��� ����� ���Ͽ� �������
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



void alpha(int** img_in, int** img_out, int dx, int dy, double a) // ���� ���� ����� ����ؼ� ���ش��� ���ĸ� ����
{
	int avg = (int)AVG(img_in, dx, dy);
	for (int j = 0; j < dy; j++)
		for (int i = 0; i < dx; i++)
		{
			img_out[j][i] = a*(img_in[j][i] - avg);
		}

}

void alpha2(int** img_in, int** img_out, int dx, int dy, double a, int avg) //���� ������� ���ְ� ���ĸ� ����
{

	for (int j = 0; j < dy; j++)
		for (int i = 0; i < dx; i++)
		{
			img_out[j][i] = a*(img_in[j][i] - avg);
		}

}
void alpha_plus(int** img_in, int** img_out, int dx, int dy, int avg) // ����� ������
{


	for (int j = 0; j < dy; j++)
		for (int i = 0; i < dx; i++)
		{
			img_out[j][i] = img_in[j][i] + avg;
		}
}

void alpha_ninus(int** img_in, int** img_out, int dx, int dy) // ����� ����
{
	int avg = (int)AVG(img_in, dx, dy);
	for (int j = 0; j < dy; j++)
		for (int i = 0; i < dx; i++)
		{
			img_out[j][i] = img_in[j][i] - avg;
		}
}



OUTPUT** ER_Alloc2(int width, int height) // �Ҵ�
{
	OUTPUT** tmp;
	tmp = (OUTPUT**)calloc(height, sizeof(int*));
	for (int i = 0; i<height; i++)
		tmp[i] = (OUTPUT*)calloc(width, sizeof(OUTPUT));
	return(tmp);
}

void ER_Free2(encodingResult** image, int width, int height) { //�޸� ����
	for (int i = 0; i<height; i++)
		free(image[i]);

	free(image);
}




bool ReadPrameter(char* name, OUTPUT** A, int width, int height, int size) { // ������ �д� �Լ�
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
void Write_Gray(int** img, int width, int height) // ȸ������ ����� �Լ�
{
	for (int y = 0; y<height; y++)
		for (int x = 0; x < width; x++)
		{
			img[y][x] = 128;
		}
}

void Write_white(int** img, int width, int height) // Ⱥ������ ����� �Լ�
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
		int **block_ori = IntAlloc2(size * 2, size * 2); // 2N ����� ���� 2�����迭
		int **block_half = IntAlloc2(size, size); // 2N-> N ����� ���� 2�����迭
		int **block_AC = IntAlloc2(size, size); // ��ջ��� ���� ���� ����� ���� 2�����迭
		int **block_geo = IntAlloc2(size, size); // ������ ���� 2���� �迭
		int **block_plus = IntAlloc2(size, size); // ����� ���Ѻ���� ���� �迭



		ReadBlock(img_dec->img_in, t->y_best, t->x_best, 2 * size, 2 * size, block_ori); // ���ڵ� ������ �ִ� x,y��ǥ���� 2N����� �ڸ�
		half_reduction(block_ori, 2 * size, 2 * size, block_half); // �ڸ��� �ٽ� N������� ���

		alpha(block_half, block_AC, size, size, t->alpha); // ��ջ��� ���ĸ� ����
		isometry(t->step, block_AC, block_geo, size, size); // ���ڵ������� �ִ� geo���� ���� ����

		plusACblock(block_geo, size, size, block_plus, t->avg); // ���ڵ��������ִ� ��հ��� ����

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
	WriteBlock(img_dec->img_in, 0, 0, img_dec->height, img_dec->width, img_dec->img_tmp); //�ӽð����� �ִ� ���ڵ� ����� �ٽ� ���� ������ �ű����ν� ��� ��ø�ǰ���
	
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
	INPUT input; // ó�������� ������ ����ִ� ����ü
	INPUT psnr;
	input.img_in = ReadImage("lena.bmp", &input.width, &input.height); // ��������
	psnr.img_in = ReadImage("lena.bmp", &input.width, &input.height);
	input.size = 16;
	input.img_tmp = IntAlloc2(input.width, input.height);
	double PSNR = 0;
	OUTPUT** Result; // ���ڵ� ����� ���� ����ü
	Result = ER_Alloc2(input.width / input.size, input.height / input.size); // 2�������� ���� �Ҵ�
	



	ReadPrameter("encoding5.txt", Result, input.width, input.height, input.size); // ���� �б�

	for (int i = 0; i < input.height / input.size; i++) //������ �� �������� ����غ�
		for (int j = 0; j < input.width / input.size; j++) {

			search_Ouput(Result[i][j]);

		}
	
	Write_white(input.img_in, input.width, input.height);
	Quad_Draw(Result, input.width, input.height, &input);
	ImageShow("�����1", input.img_in, input.width, input.height); //��� ���� ���� ���
										  
	Write_Gray(input.img_in, input.width, input.height); // ȸ�� ���� �����
	for (int i = 0; i < 10; i++) {
	  Decoding(Result, input.width, input.height, &input); // ���ڵ� 5�� �ݺ�
	  PSNR =computePSNR(psnr.img_in, input.img_in, input.width, input.height);
	  printf("PSNR �� : %f\n", PSNR);
	  ImageShow("��������", input.img_in, input.width, input.height); //���
	}
													  
	return 0;
}
