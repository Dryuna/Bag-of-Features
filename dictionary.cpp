extern "C"
{
    #include "libCluster/cluster.h"
}
#include "imagefeatures.h"

Dictionary::Dictionary()
{
    dictionary = NULL;
    centroid = NULL;
    size = 0;
    length = 0;
}

Dictionary::Dictionary(int n, int m)
{
    size = n;
    length = m;

    dictionary = new double* [size];
    for(int i = 0; i < size; ++i)
    {
        dictionary[i] = new double [length];
        for(int j = 0; j < length; ++j)
        {
            dictionary[i][j] = 0;
        }
    }

    centroid = new double [length];
}

Dictionary::~Dictionary()
{
     dealloc();
}

void Dictionary::dealloc()
{
    if(dictionary != NULL)
    {
        for(int i = 0; i < size; ++i)
            delete [] dictionary[i];
        delete [] dictionary;
        delete [] centroid;
    }
}

void Dictionary::alloc(int n, int m)
{
    dealloc();

    size = n;
    length = m;

    dictionary = new double* [size];
    for(int i = 0; i < size; ++i)
    {
        dictionary[i] = new double [length];
        for(int j = 0; j < length; ++j)
        {
            dictionary[i][j] = 0;
        }
    }

    centroid = new double [length];
}

// C-Clustering lib kCluster function
void Dictionary::buildKClustering(ObjectSet* obj,
                    int numClasses,
                    int numFeatures,
                    int featureLength,
                    int numClusters,
                    int pass,
                    char method,
                    char dist)
{
    alloc(numClusters, featureLength);

    cout << "Initializing the data..." << endl;

    int i, j;
    int k = 0, l = 0, m = 0;
    int totalImages = 0;
    int featureCount;
    int ifound;
    double* error = new double [numFeatures];
    int *clusterID = new int [numFeatures];
    double ** featureData = new double* [numFeatures];
    // Allocate mask and set it all to 1 (assume no missing data)
    int ** mask = new int* [numFeatures];
    for(i = 0; i < numFeatures; i++)
    {
        mask[i] = new int [length];
        featureData[i] = new double [length];
        for(j = 0; j < length; j++)
            mask[i][j] = 1;
    }

    // Set the weights equal, all 1
    double* weight = new double [length];
    for(i = 0; i < length; i++)
        weight[i] = 1.0;

    // For each class
    for(m = 0; m < numClasses; m++)
    {
        totalImages = obj[m].setCount;
        // For each image in that class...
        for(l = 0; l < totalImages; l++)
        {
            // for each feature in that image...
            for(i = 0; i < obj[m].featureSet[l].size; i++)
            {
                // Copy the descriptor into the data array
                for(j = 0; j < featureLength; j++)
                {
                    featureData[k][j] = (double)obj[m].featureSet[l].descriptors[i][j];
                    //cout << featureData[k][j] << " ";
                }
                //cout << endl;
                k++;
            }
        }
    }

    cout << "Clustering data..." << endl;

    kcluster(size, numFeatures, length, featureData,
                mask, weight, 0, pass, method, dist,
                clusterID, error, &ifound);

    cout << "Computing cluster centers and building dictionary..." << endl;

    int* indexCount = new int [size];
    int index;
    for(i = 0; i < size; i++)
        indexCount[i] = 0;

    // Figure out how many clusters per index
    for(i = 0; i < numFeatures; i++)
    {
        index = clusterID[i];
        indexCount[index]++;
        for(j = 0; j < length; j++)
        {
            dictionary[index][j] += featureData[i][j];
        }
    }

    for(i = 0; i < size; i++)
    {
        for(j = 0; j < length; j++)
        {
            dictionary[i][j] /= (double)indexCount[i];
        }
    }

    // Release all memory
    for(i = 0; i < numFeatures; i++)
    {
        delete [] mask[i];
        delete [] featureData[i];
    }
    delete [] featureData;
    delete [] mask;
    delete [] weight;
    delete [] indexCount;
    delete [] error;
    delete [] clusterID;

}

void Dictionary::calcCentroid()
{
    for(int i = 0; i < length; ++i)
        centroid[i] = 0;

    for(int i = 0; i < size; ++i)
    {
        for(int j = 0; j < length; ++j)
        {
            centroid[j] += dictionary[i][j];
        }
    }
    for(int i = 0; i < length; ++i)
        centroid[i] /= (double)size;
}

int Dictionary::matchFeature(const double *feature)
{
    int i, j;
    int minIndex;

    double minDistance = DBL_MAX;
    double tempDistance = 0;

    for(i = 0; i < size; ++i)
    {
        tempDistance = 0;
        for(j = 0; j < length; ++j)
        {
            tempDistance += (dictionary[i][j] - feature[j])*
                            (dictionary[i][j] - feature[j]);
        }
        if(tempDistance < minDistance)
        {
            minDistance = tempDistance;
            minIndex = i;
        }
    }

    //cout << "MinIndex: " << minIndex << endl;
    return minIndex;
}
