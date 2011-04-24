/***********************************************************

	Image Feature class
	object to manage the features found in images
	Histogram class

************************************************************/
#ifndef IMGFEATURES_H
#define IMGFEATURES_H

#include <fstream>
#include <iostream>
#include <cv.h>
#include <ml.h>
#include <cxcore.h>
#include <highgui.h>
#include "libSVM/svm.h"

using namespace std;

class Dictionary;
class ImageFeatures;
class HistogramFeatures;
class ObjectSet;


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

        void extractSIFT_CV(cv::Mat img,
                            double p1,
                            double p2,
                            bool output);

        void extractSURF_CV(cv::Mat img,
                            double hessThresh,
                            int octaves,
                            int layers,
                            bool extend,
                            bool output);

        // Copy the values in
        void copyDescriptors(const double** input, int count, int len);
        bool copyDescriptorAt(const double* vector, int location);

        double** descriptors;
        int size;
        int length;

};

class HistogramFeatures
{
	public:

        ~HistogramFeatures();
        HistogramFeatures();
        HistogramFeatures(int n, int l);
        HistogramFeatures(const HistogramFeatures& hist);

        void alloc(int n, int l);
        void dealloc();

        float getValAt(int i);
        bool addToBin(int i);

        void buildBoF(const ImageFeatures &img, Dictionary &d, int l);

        float predict(CvSVM& svm);
        double predict(const svm_model* svm);
        // Normalize the bins in the histogram from 0 to 1
        void normalizeHist();

        int bins;
        double label;
        double *histogram;
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

        void buildBoFs(Dictionary &d, int setLabel);

        double predict(CvSVM& svm, int setLabel);
        double predict(const svm_model* svm, int setLabel);

        ImageFeatures* featureSet;
        HistogramFeatures* histogramSet;
        int setCount;
};

class Dictionary
{
    public:
        Dictionary();
        Dictionary(int n, int m);
        Dictionary(const Dictionary &dict);
        ~Dictionary();

        void dealloc();
        void alloc(int n, int m);

        // C-Clustering lib kCluster function
        void buildKClustering(ObjectSet* obj,
                            int numClasses,
                            int numFeatures,
                            int featureLength,
                            int numClusters,
                            int pass,
                            char method,
                            char dist);

        void FLANNClustering(ObjectSet* obj,
                            int numClasses,
                            int numFeatures,
                            int featureLength,
                            int numClusters,
                            int branching,
                            int iterations,
                            cvflann::flann_centers_init_t centerInit,
                            float cbIndex
                            );

        void calcCentroid();
        int matchFeature(const double *feature);

        Dictionary& operator=(const Dictionary &rhs);

        void save(char* name);
        bool load(char* name);

        double** dictionary;
        double* centroid;
        cv::FlannBasedMatcher matcher;

        int size;
        int length;
};

#endif
