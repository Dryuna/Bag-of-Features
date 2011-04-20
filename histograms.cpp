#include "imagefeatures.h"
///////////////////////////////////////////////////////////////////////////
// HISTOGRAMS
///////////////////////////////////////////////////////////////////////////

HistogramFeatures::~HistogramFeatures()
{
    delete [] histogram;
    histogram = NULL;
}

HistogramFeatures::HistogramFeatures()
{
    bins = 0;
    label = NULL;
    histogram = NULL;
}

HistogramFeatures::HistogramFeatures(int n, int l)
{
    int i;
    bins = n;
    label = l;
    histogram = new double [bins];

    for(i = 0; i < bins; i++)
        histogram[i] = 0;
}

HistogramFeatures::HistogramFeatures(const HistogramFeatures& hist)
{
    int i;
    bins = hist.bins;
    label = hist.label;
    histogram = new double [bins];

    for(i = 0; i < bins; i++)
        histogram[i] = hist.histogram[i];
}

void HistogramFeatures::alloc(int n, int l)
{
    dealloc();
    int i;
    bins = n;
    label = l;
    histogram = new double [bins];
    for(i = 0; i < bins; i++)
        histogram[i] = 0;
}

void HistogramFeatures::dealloc()
{
    if(histogram != NULL)
    {
        delete [] histogram;
        histogram = NULL;
    }
}

float HistogramFeatures::getValAt(int i)
{
    if(i > -1 && i < bins)
        return histogram[i];
    else
        return -1;
}

bool HistogramFeatures::addToBin(int i)
{
    if(i > -1 && i < bins)
    {
        histogram[i]++;
        return true;
    }
    else
        return false;
}

void HistogramFeatures::buildBoF(const ImageFeatures &img,
                                 Dictionary &d,
                                 int l)
{
    int i;
    int pos;
    alloc(d.size, l);

    for(i = 0; i < img.size; ++i)
    {
        //cout << "Matching Feature " << i << "\n\t\t\t";
        pos = d.matchFeature(img.descriptors[i]);
        histogram[pos]++;
    }

    normalizeHist();
}

float HistogramFeatures::predict(CvSVM& svm)
{
    cv::Mat hist(1, bins, CV_32FC1);
    float* hPtr = hist.ptr<float>(0);
    for(int i = 0; i < bins; ++i)
        hPtr[i] = (float)histogram[i];

    double prediction = svm.predict(hist, false);

    return prediction;
}

double HistogramFeatures::predict(const svm_model* svm)
{
    //find the number of non-zero elements
    int i, j;
    int nCount = 0;
    for(i = 0; i < bins; ++i)
        if(histogram[i] != 0)
            nCount++;

    struct svm_node* testData = new struct svm_node [nCount+1];
    j = 0;
    for(i = 0; i < bins; ++i)
    {
        if(histogram[i] != 0)
        {
            testData[j].index = i+1;
            testData[j].value = histogram[i];
            ++j;
        }
    }
    testData[nCount].index = -1;

    double classification = svm_predict(svm, testData);
    delete [] testData;

    return classification;
}

// Normalize the bins in the histogram from 0 to 1
void HistogramFeatures::normalizeHist()
{
    double magnitude = 0.0;
    int i;
    for(i = 0; i < bins; i++)
    {
        //magnitude += histogram[i][j]*histogram[i][j];
        magnitude += histogram[i];
    }
    //magnitude = sqrt(magnitude);
    // divide by the magnitude
    if(magnitude > 0)
    {
        for(i = 0; i < bins; i++)
            histogram[i] /= magnitude;
    }
}

