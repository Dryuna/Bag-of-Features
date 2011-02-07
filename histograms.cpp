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

