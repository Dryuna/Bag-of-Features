//#include <vector>

extern "C"
{
    #include "libCluster/cluster.h"
}
#include "imagefeatures.h"

Dictionary::Dictionary()
{
    dictionary = NULL;
    centroid = NULL;
    //this->matcher = NULL;
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
    //this->matcher = NULL;
}

Dictionary::Dictionary(const Dictionary &dict)
{
    size = dict.size;
    length = dict.length;

    dictionary = new double* [size];
    for(int i = 0; i < size; ++i)
    {
        dictionary[i] = new double [length];
        for(int j = 0; j < length; ++j)
        {
            dictionary[i][j] = dict.dictionary[i][j];
        }
    }

    centroid = new double [length];
    for(int i = 0; i < length; ++i)
    {
        centroid[i] = dict.centroid[i];
    }

    matcher = dict.matcher;
    //this->matcher = dict.matcher;
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
    }
    if(centroid != NULL)
        delete [] centroid;
    //if(this->matcher != NULL)
    //    delete this->matcher;
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
    //this->matcher = NULL;
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

void Dictionary::FLANNClustering(ObjectSet* obj,
                                 int numClasses,
                                 int numFeatures,
                                 int featureLength,
                                 int numClusters,
                                 int branching,
                                 int iterations,
                                 cvflann::flann_centers_init_t centerInit,
                                 float cbIndex
                                 )
{
    cv::Mat featureMat(numFeatures,featureLength, CV_32F);
    cv::Mat clusterCenters(numClusters, featureLength, CV_32F);
    cvflann::KMeansIndexParams params(branching, iterations, centerInit, cbIndex);
    int i, j, k, l, m=0;
    int totalImages;
    float* ptr;

    for(i = 0; i < numClasses; i++)
    {
        totalImages = obj[i].setCount;
        // For each image in that class...
        for(j = 0; j < totalImages; j++)
        {
            // for each feature in that image...
            for(k = 0; k < obj[i].featureSet[j].size; k++)
            {
                ptr = featureMat.ptr<float>(m);
                // Copy the descriptor into the data array
                for(l = 0; l < featureLength; l++)
                {
                    ptr[l] = (double)obj[i].featureSet[j].descriptors[k][l];
                    //cout << featureData[k][j] << " ";
                }
                //cout << endl;
                m++;
            }
        }
    }

    int clustersFound = cv::flann::hierarchicalClustering<float,float>(featureMat, clusterCenters, params);

    alloc(clustersFound, featureLength);

    //Build KNN map
    //cvflann::KDTreeIndexParams matchParams(12);
    //cvflann::KMeansIndexParams matchParams;
    //this->matcher = new cv::flann::Index(featureMat, matchParams);

    vector<cv:: Mat> flannMat(1);
    flannMat[0] = clusterCenters;
    matcher.clear();
    matcher.add(flannMat);
    matcher.train();

    for(i = 0; i < size; ++i)
    {
        ptr = clusterCenters.ptr<float>(i);
        for(j = 0; j < length; ++j)
        {
            dictionary[i][j] = ptr[j];
        }
    }
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
    {
        centroid[i] /= (double)size;
        //cout << centroid[i] << " ";
    }
    //cout << endl;
}

/*
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
*/

int Dictionary::matchFeature(const double *feature)
{
    //std::vector<float> query(length);
    vector<cv::DMatch> indexes;
    cv::Mat query(1, length, CV_32FC1);
    float* ptr = query.ptr<float>(0);
    for(int i = 0; i < length; ++i)
        ptr[i] = feature[i];

    matcher.match(query, indexes);

    //cout << indexes[0].trainIdx << endl;

    return indexes[0].trainIdx;
}


Dictionary& Dictionary::operator=(const Dictionary &rhs)
{
    if(this == &rhs)
        return *this;

    dealloc();

    this->size = rhs.size;
    this->length = rhs.length;

    this->dictionary = new double* [this->size];
    for(int i = 0; i < this->size; ++i)
    {
        this->dictionary[i] = new double [this->length];
        for(int j = 0; j < this->length; ++j)
        {
            this->dictionary[i][j] = rhs.dictionary[i][j];
        }
    }

    centroid = new double [length];
    for(int i = 0; i < length; ++i)
    {
        centroid[i] = rhs.centroid[i];
    }

    matcher = rhs.matcher;

    return *this;
}

void Dictionary::save(char* name)
{
    char temp[64];
    strcpy(temp, name);
    strcat(temp, "_codex");

    ofstream fout;
    fout.open(temp);
    int i, j;

    fout << size << " " << length << endl;
    for(i = 0; i < size; ++i)
    {
        for(j = 0; j < length; ++j)
        {
            fout << dictionary[i][j] << " ";
        }
        fout << endl;
    }
    fout.close();

/*
    strcpy(temp, name);
    strcat(temp, "_flannMatcher");
    cv::FileStorage storage;
    storage << temp;

    matcher.write(storage);
*/
}

bool Dictionary::load(char* name)
{
    char temp[64];
    strcpy(temp, name);
    strcat(temp, "_codex");

    ifstream fin;
    fin.open(temp);
    if(!fin.good())
        return false;
    int i, j;

    fin >> size;
    fin >> length;

    alloc(size, length);

    vector<cv:: Mat> flannMat(1);
    flannMat[0].create(size, length, CV_32FC1);
    float* ptr;

    for(i = 0; i < size; ++i)
    {
        ptr = flannMat[0].ptr<float>(i);
        for(j = 0; j < length; ++j)
        {
            fin >> dictionary[i][j];
            ptr[j] = dictionary[i][j];
        }
    }
    fin.close();
/*
    strcpy(temp, name);
    strcat(temp, "_flannMatcher");
    cv::FileNode storage(temp, cv::FileStorage::READ);

    matcher.read(storage);
*/

    matcher.clear();
    matcher.add(flannMat);
    matcher.train();

    return true;
}
