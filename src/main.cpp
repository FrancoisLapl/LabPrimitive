#include <cv.h> 			//OpenCV lib
#include <highgui.h>		//OpenCV lib
#include <string>

#define NUM_SAMPLES 1000
#define NUM_FEATURES 6

using namespace std;

// Bart Train: 80 items: bart1.bmp - bart80.bmp
// Homer Train 62 items: homer1.bmp - homer62.bmp
// Bart Valid: 54 items: bart116.bmp - bart169.bmp
// Homer Valid: 37 items: homer88.bmp - homer124.bmp

void BuildFileName(int iNum, char *character, char *cFileName, bool training);

float OrangeFeatureExtraction(int h, int w, unsigned char red, unsigned char blue, unsigned char green, float fOrange, const IplImage *processed);

float WhiteFeatureExtraction(unsigned char red, unsigned char blue, unsigned char green, float fWhite);

float BrownFeatureExtraction(unsigned char red, unsigned char blue, unsigned char green, float fBrown);

float BlueFeatureExtraction(unsigned char red, unsigned char blue, unsigned char green, float fBlue);

float GreenFeatureExtraction(unsigned char red, unsigned char blue, unsigned char green, float fGreen);

float RedFeatureExtraction(int h, int w, unsigned char red, unsigned char blue, unsigned char green, float fRed, const IplImage *processed);

void LoopOverAllPixels(const IplImage *img, const IplImage *processed, float &fOrange, float &fWhite, float &fBrown, float &fBlue, float &fGreen, float &fRed);

void
ProcessImageBatch(int firstItemNb, int lastItemNb, char *character, FILE *fp, IplImage *&img, IplImage *&processed,
	bool training);

void InitCharArray(char *cFileName);

int main(int argc, char** argv)
{
	bool training = false;
	char *resultFileName;
	FILE *fp;

	string arg = argv[1];
	if (arg == "train") {
		training = true;
		resultFileName = "apprentissage-homer-bart-lisa.arff";
	}
	else {
		training = false;
		resultFileName = "validation-homer-bart-lisa.arff";
	}

	// Open a text file to store the feature vectors
	fp = fopen(resultFileName, "w");

	if (fp == NULL) {
		perror(resultFileName);
		return EXIT_FAILURE;
	}

	// Setup .arff header
	fprintf(fp, "@relation Homer-Bart\n");
	fprintf(fp, "\n");
	fprintf(fp, "@attribute Orange real\n");
	fprintf(fp, "@attribute White real\n");
	fprintf(fp, "@attribute Brown real\n");
	fprintf(fp, "@attribute Blue real\n");
	fprintf(fp, "@attribute Green real\n");
	fprintf(fp, "@attribute Red real\n");
	fprintf(fp, "\n");
	fprintf(fp, "@attribute classe {homer, bart, lisa}\n");
	fprintf(fp, "@data");
	fprintf(fp, "\n");

	// OpenCV variables related to the image structure.
	// IplImage structure contains several information of the image (See OpenCV manual).
	IplImage *img = NULL;
	IplImage *processed = NULL;
	IplImage *threshold = NULL;

	// OpenCV variable that stores the image width and height
	CvSize tam;

	// OpenCV variable that stores a pixel value
	CvScalar element;

	// *****************************************************************************************************************************************
	// TRAINING SAMPLES 
	// HOMER
	// Homer Train 62 items: homer1.bmp - homer62.bmp
	// Homer Valid 37 items: homer88.bmp - homer124.bmp
	// *****************************************************************************************************************************************

	if (training) {
		ProcessImageBatch(1, 62, "homer", fp, img, processed, true);
	}
	else {
		ProcessImageBatch(88, 124, "homer", fp, img, processed, false);
	}

	// *****************************************************************************************************************************************
	// TRAINING SAMPLES
	// BART
	// Bart Train: 80 items: bart1.bmp - bart80.bmp
	// Bart Valid: 80 items: bart116.bmp - bart169.bmp
	// *****************************************************************************************************************************************

	if (training) {
		ProcessImageBatch(1, 80, "bart", fp, img, processed, true);
	}
	else {
		ProcessImageBatch(116, 169, "bart", fp, img, processed, false);
	}

	// *****************************************************************************************************************************************
	// TRAINING SAMPLES
	// LISA
	// Lisa Train: 33 items: lisa1.bmp - lisa33.bmp
	// Lisa Valid: 33 items: lisa34.bmp - lisa46.bmp
	// *****************************************************************************************************************************************

	if (training) {
		ProcessImageBatch(1, 33, "lisa", fp, img, processed, true);
	}
	else {
		ProcessImageBatch(34, 46, "lisa", fp, img, processed, false);
	}

	// *****************************************************************************************************************************************
	// TRAINING SAMPLES
	// OTHER
	// Other Train: 121 items: other1.bmp - other121.bmp
	// Other Valid: 121 items: other122.bmp - other170.bmp
	// *****************************************************************************************************************************************

	if (training) {
		//ProcessImageBatch(1, 80, "other", fp, img, processed, true);
	}
	else {
		//ProcessImageBatch(122, 170, "other", fp, img, processed, false);
	}

	cvReleaseImage(&img);
	cvDestroyWindow("Original");

	cvReleaseImage(&processed);
	cvDestroyWindow("Processed");

	fclose(fp);

	return 0;
}

void InitCharArray(char *cFileName) {
	// Fill cFileName with zeros
	for (int i = 0; i < 50; i++)
	{
		cFileName[i] = '\0';
	}
}

void
ProcessImageBatch(int firstItemNb, int lastItemNb, char *character, FILE *fp, IplImage *&img, IplImage *&processed, bool training) {

	CvSize tam;
	IplImage *threshold;

	// Feature variables
	float fOrange;
	float fWhite;
	float fBrown;
	float fBlue;
	float fGreen;
	float fRed;

	// In fact it is a "matrix of features"
	float fVector[NUM_SAMPLES][NUM_FEATURES];

	// Fill fVector with zeros
	for (int ii = 0; ii < NUM_SAMPLES; ii++)
	{
		for (int jj = 0; jj < NUM_FEATURES; jj++)
		{
			fVector[ii][jj] = 0.0;
		}
	}

	// Variable filename
	static char cFileName[50] = { '\0' };
	InitCharArray(cFileName);

	// Take all the image files at the range
	for (int iNum = firstItemNb; iNum <= lastItemNb; iNum++) {
		BuildFileName(iNum, character, cFileName, training);

		// Load the image from disk to the structure img.
		// 1  - Load a 3-channel image (color)
		// 0  - Load a 1-channel image (gray level)
		// -1 - Load the image as it is  (depends on the file)
		img = cvLoadImage(cFileName, -1);

		// Gets the image size (width, height) 'img'
		tam = cvGetSize(img);

		// Creates a header and allocates memory (tam) to store a copy of the original image.
		// 1 - gray level image
		// 3 - color image
		// processed = cvCreateImage( tam, IPL_DEPTH_8U, 3);


		// Make a image clone and store it at processed and threshold
		processed = cvCloneImage(img);
		threshold = cvCloneImage(img);

		// Initialize variables with zero
		fOrange = 0.0;
		fWhite = 0.0;
		fBrown = 0.0;
		fBlue = 0.0;
		fGreen = 0.0;
		fRed = 0.0;

		LoopOverAllPixels(img, processed, fOrange, fWhite, fBrown, fBlue, fGreen, fRed);

		// Lets make our counting somewhat independent on the image size...
		// Compute the percentage of pixels of a given colour.
		// Normalize the feature by the image size
		fOrange = fOrange / ((int)img->height * (int)img->width);
		fWhite = fWhite / ((int)img->height * (int)img->width);
		fBrown = fBrown / ((int)img->height * (int)img->width);
		fBlue = fBlue / ((int)img->height * (int)img->width);
		fGreen = fGreen / ((int)img->height * (int)img->width);
		fRed = fRed / ((int)img->height * (int)img->width);


		// Store the feature value in the columns of the feature (matrix) vector
		fVector[iNum][0] = fOrange;
		fVector[iNum][1] = fWhite;
		fVector[iNum][2] = fBrown;
		fVector[iNum][3] = fBlue;
		fVector[iNum][4] = fGreen;
		fVector[iNum][5] = fRed;


		// Here you can add more features to your feature vector by filling the other columns: fVector[iNum][3] = ???; fVector[iNum][4] = ???;

		// Shows the feature vector at the screen
		//printf(" : #%d %f %f\n\n", iNum, fVector[iNum][1], fVector[iNum][2]);
		printf("%d %f %f %f %f %f\n", iNum, fVector[iNum][0], fVector[iNum][1], fVector[iNum][2], fVector[iNum][3], fVector[iNum][4], fVector[iNum][5]);

		// And finally, store your features in a file
		fprintf(fp, "%f,", fVector[iNum][0]);
		fprintf(fp, "%f,", fVector[iNum][1]);
		fprintf(fp, "%f,", fVector[iNum][2]);
		fprintf(fp, "%f,", fVector[iNum][3]);
		fprintf(fp, "%f,", fVector[iNum][4]);
		fprintf(fp, "%f,", fVector[iNum][5]);

		// IMPORTANT
		// Do not forget the label....
		fprintf(fp, "%s\n", character);


		// Finally, give a look at the original image and the image with the pixels of interest in green
		// OpenCV create an output window
		cvShowImage("Original", img);
		cvShowImage("Processed", processed);

		// Wait until a key is pressed to continue...
		cvWaitKey(0);
	}
}

void LoopOverAllPixels(const IplImage *img, const IplImage *processed, float &fOrange, float &fWhite, float &fBrown, float &fBlue, float &fGreen, float &fRed) {

	int h;
	int w;

	// Variables to store the RGB values of a pixel
	unsigned char red;
	unsigned char blue;
	unsigned char green;

	// Loop that reads each image pixel
	for (h = 0; h < img->height; h++) // rows
	{
		for (w = 0; w < img->width; w++) // columns
		{
			// Read each channel and writes it into the blue, green and red variables. Notice that OpenCV considers BGR
			blue = ((uchar *)(img->imageData + h * img->widthStep))[w * img->nChannels + 0];
			green = ((uchar *)(img->imageData + h * img->widthStep))[w * img->nChannels + 1];
			red = ((uchar *)(img->imageData + h * img->widthStep))[w * img->nChannels + 2];

			// Shows the pixel value at the screenl
			//printf( "pixel[%d][%d]= %d %d %d \n", h, w, (int)blue, (int)green, (int)red );

			// Here starts the feature extraction....
			fOrange = OrangeFeatureExtraction(h, w, red, blue, green, fOrange, processed);
			fWhite = WhiteFeatureExtraction(red, blue, green, fWhite);
			fBrown = BrownFeatureExtraction(red, blue, green, fBrown);
			fBlue = BlueFeatureExtraction(red, blue, green, fBlue);
			fGreen = GreenFeatureExtraction(red, blue, green, fGreen);
			fRed = RedFeatureExtraction(h, w, red, blue, green, fRed, processed);

			// Here you can add your own features....... Good luck

		}
	}
}

float WhiteFeatureExtraction(unsigned char red, unsigned char blue, unsigned char green, float fWhite) {

	// Detect and count the number of white pixels (just a dummy feature...)
	// Verify if the pixels have a given value ( White, defined as R[253-255], G[253-255], B[253-255] ). If so, count it...
	if (blue >= 253 && green >= 253 && red >= 253)
	{
		fWhite++;
	}

	return fWhite;
}

float BrownFeatureExtraction(unsigned char red, unsigned char blue, unsigned char green, float fBrown) {

	if (blue >= 102 && blue <= 112 && green >= 168 && green <= 178 && red >= 180 && red <= 210)
	{
		fBrown++;
	}

	return fBrown;
}

float BlueFeatureExtraction(unsigned char red, unsigned char blue, unsigned char green, float fBlue) {

	if (blue >= 100 && green <= 130 && red <= 25)
	{
		fBlue++;
	}

	return fBlue;
}

float GreenFeatureExtraction(unsigned char red, unsigned char blue, unsigned char green, float fGreen) {

	if (blue >= 14 && blue <= 34 && green >= 130 && green <= 150 && red <= 90 && red >= 70)
	{
		fGreen++;
	}

	return fGreen;
}

float RedFeatureExtraction(int h, int w, unsigned char red, unsigned char blue, unsigned char green, float fRed, const IplImage *processed) {

	if (blue <= 50 && green <= 50 && red >= 190)
	{
		fRed++;
		((uchar *)(processed->imageData + h*processed->widthStep))[w*processed->nChannels + 0] = 0;
		((uchar *)(processed->imageData + h*processed->widthStep))[w*processed->nChannels + 1] = 255;
		((uchar *)(processed->imageData + h*processed->widthStep))[w*processed->nChannels + 2] = 0;
	}

	return fRed;
}

float OrangeFeatureExtraction(int h, int w, unsigned char red, unsigned char blue, unsigned char green, float fOrange, const IplImage *processed) {

	// Detect and count the number of orange pixels
	// Verify if the pixels have a given value ( Orange, defined as R[240-255], G[85-105], B[11-22] ). If so, count it...
	if (blue >= 11 && blue <= 22 && green >= 85 && green <= 105 && red >= 240 && red <= 255)
	{
		fOrange++;

		// Just to be sure we are doing the right thing, we change the color of the orange pixels to green [R=0, G=255, B=0] and show them into a cloned image (processed)

	}

	return fOrange;
}

void BuildFileName(int iNum, char *character, char *cFileName, bool training = true) {
#ifdef __linux__ 
	char* trainPathPattern = "../Train/%s%d.bmp";
	char* validPathPattern = "../Valid/%s%d.bmp";
#elif _WIN32
	char* trainPathPattern = "Train/%s%d.bmp";
	char* validPathPattern = "Valid/%s%d.bmp";
#else
	char* trainPathPattern = "Train/%s%d.bmp";
	char* validPathPattern = "Valid/%s%d.bmp";
#endif
	// Build the image filename and path to read from disk
	if (training) {
		sprintf(cFileName, trainPathPattern, character, iNum);
	}
	else {
		sprintf(cFileName, validPathPattern, character, iNum);
	}

	printf("%s", cFileName);
}
