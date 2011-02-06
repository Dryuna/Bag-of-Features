/***********************************************************

	Image Feature class
	object to manage the features found in images
	Histogram class

************************************************************/
#include <fstream>
#include <iostream>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include "parameters.h"

using namespace std;

class ImageFeatures
{
	public:
        // Destructor
        ~ImageFeatures();
        // Constructor
        ImageFeatures();
        // Copy Constructor
        ImageFeatures(const ImageFeatures &cpy);
        // Default constructor
        ImageFeatures(int len);
        ImageFeatures(int len, int s);

        // Allocating the descriptors
        void alloc(int len, int s);
        // Deallocate the descriptors
        bool dealloc();
        //Check to see if the descriptor was allocated
        bool checkAlloc();

        void extractSIFT_CV(const cv::Mat& img, double p1, double p2);

        // Copy the values in
        void copyDescriptors(const float** input, int count, int len);
        bool copyDescriptorAt(const float* vector, int location);
        bool copyDescriptorAt(const double* vector, int location);

        float** descriptors;
        int size;
        int length;

};

class HistogramFeatures
{
	public:

        ~HistogramFeatures();
        HistogramFeatures();
        HistogramFeatures(int n, int l);

        bool alloc(int n, int l);
        bool dealloc();

        float getValAt(int i);
        bool addToBin(int i);
        // Normalize the bins in the histogram from 0 to 1
        void normalizeHist();

        int bins;
        float label;
        float *histogram;
};

class ObjectSet
{
    public:
        ~ObjectSet();
        ObjectSet();
        ObjectSet(const ObjectSet &cpy);
        ObjectSet(int l);

        bool alloc(int l);
        void dealloc();

        ImageFeatures* featureSet;
        HistogramFeatures* histogramSet;
        int setCount;
        int featureCount;
};

