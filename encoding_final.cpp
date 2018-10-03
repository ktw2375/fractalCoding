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
	int flag;
	int size;
	int threshold;
	OUTPUT *sub;

};

struct INPUT { // �����̹����� ����
	int width;
	int height;
	int** img_in;
	int** img_tmp;
	int** img_result;
	int size;
};

void InitList(OUTPUT *head)
{
  head = (OUTPUT*)calloc(1, sizeof(OUTPUT));
  head->sub = NULL;
}

void PutList(OUTPUT *Target, OUTPUT *Temp)
{
	OUTPUT *New;
	New = (OUTPUT*)calloc(1, sizeof(OUTPUT));
	*New = *Temp;
	//New->sub = NULL;

	New->sub = Target->sub;
	Target->sub = New;

}

void searchList(OUTPUT *out)
{
	if (out->sub == NULL)
	{
		return;
	}
	printf("������ : %d, x��ǥ : %3d, y��ǥ : %3d, ���� : %.1f, ��� : %4d, geo : %2d, error : %3d �÷��� : %d\n", out->size, out->x_best, out->y_best, out->alpha, out->avg, out->step, out->error, out->flag);
	searchList(out->sub);

}

void WritePrameter(char* name, OUTPUT A) { //���Ͽ� ���� �Լ�

	FILE *fp = fopen(name, "a");
	
		fprintf(fp, "%d %d %d %d %lf %d\n", A.x_best, A.y_best, A.step, A.avg, A.alpha, A.flag);
	
	fclose(fp);

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

void WriteBlock(int** image, int x, int y, int dx, int dy, int** block) // ��Ͽ� ��
{
	for (int i = 0; i<dx; i++)
		for (int j = 0; j < dy; j++)
		{
			image[x + i][y + j] = block[i][j];
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

void alpha(int** img_in, int** img_out, int dx, int dy, double a) // ���� ���� ����� ����ؼ� ���ش��� ���ĸ� ����
{
	int avg = (int)AVG(img_in, dx, dy);
	for (int j = 0; j < dy; j++)
		for (int i = 0; i < dx; i++)
		{
			img_out[j][i] = a*(img_in[j][i] - avg);
		}

}


void avg_ninus(int** img_in, int** img_out, int dx, int dy) // ����� ����
{
	int avg = (int)AVG(img_in, dx, dy);
	for (int j = 0; j < dy; j++)
		for (int i = 0; i < dx; i++)
		{
			img_out[j][i] = img_in[j][i] - avg;
		}
}



void TemplateMatchingWithDownSampling_blue_plusShuffle(INPUT *img, int** blockN, int bx, int by, int y, int x)
{
	int MAX_1 = INT_MAX; //�񱳸� ���� �ſ�ū��
	int error = 0;

	int **domain_AC = IntAlloc2(bx, by);
	int **block_AC = IntAlloc2(bx, by);
	int **block2N = IntAlloc2(2 * bx, 2 * by); // 2N���
	int **domain = IntAlloc2(bx, by); // ������ ���� ���
	int **temp = IntAlloc2(bx, by);
	int** block_out = IntAlloc2(bx, by);
	int** blockN2 = IntAlloc2(bx, by);
	int** block_A = IntAlloc2(bx, by);

	OUTPUT out;
	out.flag = 0;
	out.threshold = 3;
	avg_ninus(blockN, blockN2, bx, by); // ����� ����

	for (int i = 0; i < img->height - 2 * by; i++) {
		for (int j = 0; j < img->width - 2 * bx; j++) {
			ReadBlock(img->img_in, i, j, 2 * bx, 2 * by, block2N); // 2N����ڸ�
			half_reduction(block2N, 2 * bx, 2 * by, domain); // ������ ����
			for (int step = 1; step < 9; step++) {
				for (double a = 0.3; a <= 1.0; a = a + 0.1) {
					alpha(domain, temp, bx, by, a); // 0.3 ~1.0 ���� ���ĸ� ���غ�
					isometry(step, temp, block_out, bx, by); // 1 ~ 8 ���� ������
					error = Compute_Error_block(block_out, blockN2, bx); // ���� ��
					if (error < MAX_1) { // ����ü�� ������ ������ ���� (������ ���� ������)
						out.x_best = j;
						out.y_best = i;
						MAX_1 = error;
						out.error = error;
						out.step = step;
						out.alpha = a;
						out.size = bx;
						
					}
				}
			}
		}
	}
	
	out.avg = AVG(blockN, bx, by); //  ó�� ���� blockN�� ���
   if (out.size == 16 && out.threshold < (out.error)/(bx*bx) || (out.size == 8 && out.threshold < out.error/(bx*bx) ))
	{
		out.flag = 1;
	}
	printf("x��ǥ : %d, y��ǥ : %d, ���� : %lf, ��� : %d, ����: %d ���� : %d ������ : %d, �÷���: %d \n", out.x_best, out.y_best, out.alpha, out.avg, out.step, out.error, out.size, out.flag);
	WritePrameter("encoding33.txt", out);

	if (out.size == 4) return; // ����� 4�϶� ��� �ߴ�

	if (out.size == 16 && out.threshold < (out.error) / (bx*bx) || (out.size == 8 && out.threshold < out.error / (bx*bx)))
	{
		
		ReadBlock(img->img_in, y, x, out.size / 2, out.size / 2, block_A); //1������
		TemplateMatchingWithDownSampling_blue_plusShuffle(img, block_A, out.size / 2, out.size / 2, y, x);


		ReadBlock(img->img_in, y, x + (out.size / 2), out.size / 2, out.size / 2, block_A);// 2������
		TemplateMatchingWithDownSampling_blue_plusShuffle(img, block_A, out.size / 2, out.size / 2,y, x + (out.size / 2));


		ReadBlock(img->img_in, y + (out.size / 2), x, out.size / 2, out.size / 2, block_A); //3������
		TemplateMatchingWithDownSampling_blue_plusShuffle(img, block_A, out.size / 2, out.size / 2, y + (out.size / 2), x);


		ReadBlock(img->img_in, y + (out.size / 2), x + (out.size / 2), out.size / 2, out.size / 2, block_A); //4������
		TemplateMatchingWithDownSampling_blue_plusShuffle(img, block_A, out.size / 2, out.size / 2, y + (out.size / 2), x + (out.size / 2));
	}
}









void TemplateMatchingWithDownSampling_Encoding(INPUT *image) //���ڵ��Լ�
{
	int **Block_A = IntAlloc2(image->size, image->size); //�ڸ� ����� ���� ����
	for (int y = 0; y < image->height; y = y + image->size)
		for (int x = 0; x < image->width; x = x + image->size)
		{
			ReadBlock(image->img_in, y, x, image->size, image->size, Block_A); // 16*16��ŭ�ڸ�
		
			TemplateMatchingWithDownSampling_blue_plusShuffle(image, Block_A, image->size, image->size, y, x);	
		}
}




int main(void)
{
	INPUT input; // ó�������� ������ ����ִ� ����ü
	input.size = 16;
	input.img_in = ReadImage("lena.bmp", &input.width, &input.height); // ��������
	input.img_tmp = (int**)IntAlloc2(input.width, input.height); // ��� �Űܳ��� �ӽð���

    TemplateMatchingWithDownSampling_Encoding(&input); //���ڵ�


	return 0;
}

