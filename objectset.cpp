#include "imagefeatures.h"
///////////////////////////////////////////////////
// ObjectSet
//////////////////////////////////////////////////

ObjectSet::ObjectSet()
{
    featureSet = NULL;
    histogramSet = NULL;
    setCount = 0;
}

ObjectSet::~ObjectSet()
{
    delete [] featureSet;
    delete [] histogramSet;
}

ObjectSet::ObjectSet(int l)
{
    featureSet = new ImageFeatures [l];
    histogramSet = new HistogramFeatures [l];
    setCount = l;
}

ObjectSet::ObjectSet(const ObjectSet &cpy)
{
    int i, j, k;
    setCount = cpy.setCount;
    featureSet = new ImageFeatures [setCount];
    histogramSet = new HistogramFeatures [setCount];
    for(i = 0; i < setCount; i++)
    {
        featureSet[i].length = cpy.featureSet[i].length;
        featureSet[i].size = cpy.featureSet[i].size;
        for(j = 0; j < cpy.featureSet[i].size; j++)
            for(k = 0; k < cpy.featureSet[i].length; k++)
                featureSet[i].descriptors[j][k] = cpy.featureSet[i].descriptors[j][k];

        histogramSet[i].alloc(cpy.histogramSet[i].bins, cpy.histogramSet[i].label);
        for(j = 0; j < histogramSet[i].bins; j++)
            histogramSet[i].histogram[j] = cpy.histogramSet[i].histogram[j];
    }
}

bool ObjectSet::alloc(int l)
{
    if(featureSet == NULL)
    {
        featureSet = new ImageFeatures [l];
        histogramSet = new HistogramFeatures [l];
        setCount = l;
        return true;
    }
    else
        return false;
}

void ObjectSet::dealloc()
{
    delete [] featureSet;
    delete [] histogramSet;
}

void ObjectSet::buildBoFs(Dictionary &d, const int setLabel)
{
    int i;
    for(i = 0; i < setCount; ++i)
    {
        //cout << "Histogram" << i << "\n\t\t";
        histogramSet[i].buildBoF(featureSet[i], d, setLabel);
    }
}



