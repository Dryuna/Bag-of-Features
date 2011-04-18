#include <omp.h>
#include "imagefeatures.h"
#include "math.h"

// Destructor
ImageFeatures::~ImageFeatures()
{
	int i;
    for(i = 0; i < size; i++)
        delete [] descriptors[i];
    delete [] descriptors;
}

// Constructor
ImageFeatures::ImageFeatures()
{
    descriptors = NULL;
    size = 0;
    length = 0;
}

ImageFeatures::ImageFeatures(const ImageFeatures &cpy)
{
    int i, j;
    size = cpy.size;
    length = cpy.length;
    descriptors = new double* [size];
    for(i = 0; i < size; i++)
    {
        descriptors[i] = new double [length];
        for(j = 0; j < length; j++)
            descriptors[i][j] = cpy.descriptors[i][j];
    }
}


// Default constructor
ImageFeatures::ImageFeatures(int len)
{
    size = 0;
    length = len;
    descriptors = NULL;
}
ImageFeatures::ImageFeatures(int len, int s)
{
    length = len;
    size = s;
    descriptors = NULL;
}

// Allocating the descriptors
void ImageFeatures::alloc(int len, int s)
{
    int i, j;
    length = len;
    size = s;
    descriptors = new double* [size];
    for(i = 0; i < size; i++)
        descriptors[i] = new double[length];
    for(i = 0; i < size; i++)
        for(j = 0; j < length; j++)
            descriptors[i][j] = 0.0;
}

// Deallocate the descriptors
bool ImageFeatures::dealloc()
{

    int i;
    if(descriptors != NULL && size > 0)
    {
        for(i = 0; i < size; i++)
            delete [] descriptors[i];
        delete [] descriptors;
        //descriptors = NULL;
        return true;
    }
    else
        return false;
}

//Check to see if the descriptor was allocated
bool ImageFeatures::checkAlloc()
{
    if(descriptors == NULL)
        return false;
    else
        return true;
}

void ImageFeatures::extractSIFT_CV(cv::Mat img,
                                   double p1,
                                   double p2,
                                   bool output)
{
    dealloc();
    cv::SIFT siftFeatures(p1, p2);
    vector<cv::KeyPoint> pts;
    //cv::Mat imgGray;
    //cvtColor(img, imgGray, CV_BGR2GRAY);
    cv::Mat descript; //With be 32F type
    cv::Mat mask;

    siftFeatures(img, mask, pts, descript);
    length = descript.cols;
    size = descript.rows;
    alloc(length, size);

    if(output)
    {
        cout << "Found " << size << " SIFT keypoints" << endl;
    }

    for(int i = 0; i < size; ++i)
    {
        const float* ptr = descript.ptr<float>(i);
        for(int j = 0; j < length; ++j)
        {
            descriptors[i][j] = (double)ptr[j];
        }
    }
}

void ImageFeatures::extractSURF_CV(cv::Mat img,
                            double hessThresh,
                            int octaves,
                            int layers,
                            bool extend,
                            bool output)
{
    dealloc();
    cv::SURF surfFeatures(hessThresh, octaves, layers, extend);
    vector<cv::KeyPoint> pts;
    cv::Mat mask;
    vector<float> descript;

    surfFeatures(img, mask, pts, descript);
    if(extend)
        length = 128;
    else
        length = 64;

    size = (descript.size())/length;
    alloc(length, size);

    if(output)
    {
        cout << "Found " << size << " SURF keypoints" << endl;
    }

    int k = 0;
    for(int i = 0; i < size; ++i)
    {
        for(int j = 0; j < length; ++j)
        {
            descriptors[i][j] = descript.at(k);
            k++;
        }
    }
}

// Copy the values in
void ImageFeatures::copyDescriptors(const double** input, int count, int len)
{
    int i, j;
    size = count;
    length = len;
    // Allocate the memory if it hasn't been allocated, for the features
    if(descriptors == NULL)
    {
        descriptors = new double* [size];
        for(i = 0; i < size; i++)
            descriptors[i] = new double[length];
    }
    // Copy all the vectors for this image
    for(i = 0; i < size; i++)
    {
        for(j = 0; j < length; j++)
        {
            descriptors[i][j] = input[i][j];
        }
    }
}

/*
bool ImageFeatures::copyDescriptorAt(const double* vector, int location)
{
    int i;
    // Make sure the memory has been allocated or location
    // is within the bounds
    if(descriptors == NULL || location > size-1 || location < 0)
        return false;
    else
    {
        for(i = 0; i < length; i++)
            descriptors[location][i] = vector[i];
        return true;
    }
}

bool ImageFeatures::copyDescriptorAt(const double* vector, int location)
{
    int i;
    // Make sure the memory has been allocated or location
    // is within the bounds
    if(descriptors == NULL || location > size-1 || location < 0)
        return false;
    else
    {
        for(i = 0; i < length; i++)
            descriptors[location][i] = vector[i];
        return true;
    }
}
*/
