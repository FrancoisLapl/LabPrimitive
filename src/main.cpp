#include <cv.h> 			//OpenCV lib
#include <highgui.h>		//OpenCV lib

#define NUM_SAMPLES 100
#define NUM_FEATURES 5

// Bart Train: 80 items: bart1.bmp - bart80.bmp
// Homer Train 62 items: homer1.bmp - homer62.bmp
// Bart Valid: 54 items: bart116.bmp - bart169.bmp
// Homer Valid: 37 items: homer88.bmp - homer124.bmp

void BuildFileName(int iNum, char *character, char *cFileName, bool training);

float OrangeFeatureExtraction(int h, int w, unsigned char red, unsigned char blue, unsigned char green, float fOrange, const IplImage *processed);

float WhiteFeatureExtraction(unsigned char red, unsigned char blue, unsigned char green, float fWhite);

float BrownFeatureExtraction(unsigned char red, unsigned char blue, unsigned char green, float fBrown);

float DarkBlueFeatureExtraction(unsigned char red, unsigned char blue, unsigned char green, float fDarkBlue);

float LightBlueFeatureExtraction(unsigned char red, unsigned char blue, unsigned char green, float fLightBlue);

void LoopOverAllPixels(const IplImage *img, const IplImage *processed, float &fOrange, float &fWhite, float &fBrown, float &fDarkBlue, float &fLightBlue);

void
ProcessImageBatch(int firstItemNb, int lastItemNb, char *character, FILE *fp, IplImage *&img, IplImage *&processed,
                  bool training);

void InitCharArray(char *cFileName);

int main(int argc, char** argv)
{

	FILE *fp;

	// Open a text file to store the feature vectors
	fp = fopen("apprentissage-homer-bart.arff", "w");
	// fp = fopen ("validation-homer-bart.arff","w");

	if (fp == NULL) {
		perror("failed to open apprentissage-homer-bart.arff");
		// perror("failed to open validation-homer-bart.arff");
		return EXIT_FAILURE;
	}

    // Setup .arff header
    fprintf(fp, "@relation Homer-Bart\n");
    fprintf(fp, "\n");
    fprintf(fp, "@attribute Orange real\n");
    fprintf(fp, "@attribute White real\n");
    fprintf(fp, "@attribute Brown real\n");
    fprintf(fp, "@attribute DarkBlue real\n");
    fprintf(fp, "@attribute LightBlue real\n");
    fprintf(fp, "\n");
    fprintf(fp, "@attribute classe {homer, bart}\n");
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

    ProcessImageBatch(1, 62, "homer", fp, img, processed, true);
    //ProcessImageBatch(88, 124, "homer", fp, img, processed, false);

	// *****************************************************************************************************************************************
	// TRAINING SAMPLES
	// BART
	// Bart Train: 80 items: bart1.bmp - bart80.bmp
	// Bart Valid: 80 items: bart116.bmp - bart169.bmp
	// *****************************************************************************************************************************************

    ProcessImageBatch(1, 80, "bart", fp, img, processed, true);
    //ProcessImageBatch(116, 169, "bart", fp, img, processed, false);

    // *****************************************************************************************************************************************
    // TRAINING SAMPLES
    // LISA
    // Lisa Train: 33 items: lisa1.bmp - lisa33.bmp
    // Lisa Valid: 33 items: lisa34.bmp - lisa46.bmp
    // *****************************************************************************************************************************************

    //ProcessImageBatch(1, 33, "lisa", fp, img, processed, true);
    //ProcessImageBatch(34, 46, "lisa", fp, img, processed, false);

    // *****************************************************************************************************************************************
    // TRAINING SAMPLES
    // OTHER
    // Other Train: 121 items: other1.bmp - other121.bmp
    // Other Valid: 121 items: other122.bmp - other170.bmp
    // *****************************************************************************************************************************************

    //ProcessImageBatch(1, 80, "other", fp, img, processed, true);
    //ProcessImageBatch(122, 170, "other", fp, img, processed, false);

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
	int iNum;
	CvSize tam;
	IplImage *threshold;

	// Feature variables
	float fOrange;
	float fWhite;
	float fBrown;
	float fDarkBlue;
	float fLightBlue;

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
	for (iNum = 1; iNum <= lastItemNb; iNum++) {
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
		fDarkBlue = 0.0;
		fLightBlue = 0.0;

		LoopOverAllPixels(img, processed, fOrange, fWhite, fBrown, fDarkBlue, fLightBlue);

		// Lets make our counting somewhat independent on the image size...
		// Compute the percentage of pixels of a given colour.
		// Normalize the feature by the image size
		fOrange = fOrange / ((int)img->height * (int)img->width);
		fWhite = fWhite / ((int)img->height * (int)img->width);
		fBrown = fBrown / ((int)img->height * (int)img->width);
		fDarkBlue = fDarkBlue / ((int)img->height * (int)img->width);
		fLightBlue = fLightBlue / ((int)img->height * (int)img->width);

		
		// Store the feature value in the columns of the feature (matrix) vector
		fVector[iNum][1] = fOrange;
		fVector[iNum][2] = fWhite;
		fVector[iNum][3] = fBrown;
		fVector[iNum][4] = fDarkBlue;
		fVector[iNum][5] = fLightBlue;
		

		// Here you can add more features to your feature vector by filling the other columns: fVector[iNum][3] = ???; fVector[iNum][4] = ???;

		// Shows the feature vector at the screen
		//printf(" : #%d %f %f\n\n", iNum, fVector[iNum][1], fVector[iNum][2]);
		printf( "%d %f %f %f %f %f\n", iNum, fVector[iNum][1], fVector[iNum][2], fVector[iNum][3], fVector[iNum][4], fVector[iNum][5] );

		// And finally, store your features in a file
		fprintf(fp, "%f,", fVector[iNum][1]);
		fprintf(fp, "%f,", fVector[iNum][2]);
		fprintf( fp, "%f,", fVector[iNum][3]);
		fprintf( fp, "%f,", fVector[iNum][4]);
		fprintf( fp, "%f,", fVector[iNum][5]);

		// IMPORTANT
		// Do not forget the label....
		fprintf(fp, "%s\n", character);


		// Finally, give a look at the original image and the image with the pixels of interest in green
		// OpenCV create an output window
		cvShowImage("Original", img);
		cvShowImage("Processed", processed);

		// Wait until a key is pressed to continue...
		//cvWaitKey(0);
	}
}

void LoopOverAllPixels(const IplImage *img, const IplImage *processed, float &fOrange, float &fWhite, float &fBrown, float &fDarkBlue, float &fLightBlue) {

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
			fDarkBlue = DarkBlueFeatureExtraction(red, blue, green, fDarkBlue);
			fLightBlue = LightBlueFeatureExtraction(red, blue, green, fLightBlue);

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

float DarkBlueFeatureExtraction(unsigned char red, unsigned char blue, unsigned char green, float fDarkBlue) {

	if (blue >= 100 && green <= 75 && red <= 25)
	{
		fDarkBlue++;
	}

	return fDarkBlue;
}

float LightBlueFeatureExtraction(unsigned char red, unsigned char blue, unsigned char green, float fLightBlue) {

	if (blue >= 100 && green > 75 && green <= 130 && red <= 25)
	{
		fLightBlue++;
	}

	return fLightBlue;
}

float OrangeFeatureExtraction(int h, int w, unsigned char red, unsigned char blue, unsigned char green, float fOrange, const IplImage *processed) {

	// Detect and count the number of orange pixels
	// Verify if the pixels have a given value ( Orange, defined as R[240-255], G[85-105], B[11-22] ). If so, count it...
	if (blue >= 11 && blue <= 22 && green >= 85 && green <= 105 && red >= 240 && red <= 255)
	{
		fOrange++;

		// Just to be sure we are doing the right thing, we change the color of the orange pixels to green [R=0, G=255, B=0] and show them into a cloned image (processed)
		((uchar *)(processed->imageData + h*processed->widthStep))[w*processed->nChannels + 0] = 0;
		((uchar *)(processed->imageData + h*processed->widthStep))[w*processed->nChannels + 1] = 255;
		((uchar *)(processed->imageData + h*processed->widthStep))[w*processed->nChannels + 2] = 0;
	}

	return fOrange;
}

void BuildFileName(int iNum, char *character, char *cFileName, bool training = true) {
#ifdef __linux__ 
    	char* trainPathPattern = "../Train/%s%d.bmp";
    	char* trainPathPattern = "../Valid/%s%d.bmp";
#elif _WIN32
    	char* trainPathPattern = "Train/%s%d.bmp";
    	char* trainPathPattern = "Valid/%s%d.bmp";
#else
    	char* trainPathPattern = "Train/%s%d.bmp";
    	char* trainPathPattern = "Valid/%s%d.bmp";
#endif
	// Build the image filename and path to read from disk
	if (training) {
		sprintf(cFileName, trainPathPattern, character, iNum);
	}
	else {
		sprintf(cFileName, trainPathPattern, character, iNum);
	}

	printf("%s", cFileName);
}
