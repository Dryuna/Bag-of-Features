#include <stdlib.h>
#include "bagoffeatures.h"

extern "C"
{
    #include "Sift/include/sift.h"
    #include "Sift/include/imgfeatures.h"
    #include "Sift/include/kdtree.h"
    #include "Sift/include/minpq.h"
    #include "Sift/include/utils.h"
    #include "Sift/include/xform.h"
}

#include "Surf/surflib.h"

BagOfFeatures::BagOfFeatures()
{
    numClasses = 0;
    numFeatures = 0;
    descrSize = 0;
    testObject = NULL;
    validObject = NULL;
    trainObject = NULL;
    data = NULL;
    dictionary = NULL;
    SVMModel = NULL;
    classifierType = -1;
    //SVMModel_CV = NULL;
    //NBModel_CV = NULL;
}

BagOfFeatures::BagOfFeatures(const int n, DataSet* val)
{
    int i;

    numClasses = n;
    numFeatures = 0;
    descrSize = 0;
    dictionary = NULL;
    SVMModel = NULL;
    //SVMModel_CV = NULL;
    //NBModel_CV = NULL;
    testObject = new ObjectSet [n];
    validObject = new ObjectSet [n];
    trainObject = new ObjectSet [n];
    data = new DataSet [n];
    classifierType = -1;
    int train, valid, test, label;

    for(i = 0; i < numClasses; i++)
    {
        data[i] = val[i];
        data[i].getDataInfo(train, valid, test, label);
        if(test > 0)
            testObject[i].alloc(test);
        if(valid > 0)
            validObject[i].alloc(valid);
        if(train > 0)
            trainObject[i].alloc(train);
    }
}

BagOfFeatures::~BagOfFeatures()
{
    numClasses = 0;
    numFeatures = 0;
    descrSize = 0;
    delete [] testObject;
    delete [] validObject;
    delete [] trainObject;
    delete [] data;
    if(SVMModel)
        svm_destroy_model(SVMModel);
    //SVMModel_CV.clear();
    //NBModel_CV.clear();
}

void BagOfFeatures::allocBoF(const int n, DataSet* val)
{
    int i;
    if(data != NULL)
    {
        numClasses = 0;
        delete [] testObject;
        delete [] validObject;
        delete [] trainObject;
        delete [] data;
    }

    numClasses = n;
    numFeatures = 0;
    descrSize = 0;
    classifierType = -1;
    testObject = new ObjectSet [n];
    validObject = new ObjectSet [n];
    trainObject = new ObjectSet [n];
    data = new DataSet [n];

    int train, valid, test, label;

    for(i = 0; i < numClasses; i++)
    {
        data[i] = val[i];
        data[i].getDataInfo(train, valid, test, label);
        if(test > 0)
            testObject[i].alloc(test);
        if(valid > 0)
            validObject[i].alloc(valid);
        if(train > 0)
            trainObject[i].alloc(train);
    }
}


bool BagOfFeatures::extractSIFTFeatures(int lvls,
                                        double sigma,
                                        double thresh1,
                                        int thresh2,
                                        int dbl,
                                        int width,
                                        int bins,
                                        int sizeMin,
                                        int sizeMax)
{
    if(numFeatures)
        return false;

	int i, j;
    int train, valid, test, label;
    char fileName[256];
    descrSize = width*width*bins;

    // For each object class
	for(i = 0; i < numClasses; i++)
	{
	    // Get the distribution of data
        data[i].getDataInfo(train, valid, test, label);

        // Extract the features of the training set
        // For each training image
        for(j = 0; j < train; j++)
        {
            // Get the image from the data list
            strcpy(fileName, data[i].getDataList(j));
            numFeatures += getSIFT(fileName, trainObject[i].featureSet[j],
                                lvls, sigma, thresh1, thresh2, dbl, width, bins,
                                sizeMin, sizeMax);
        }

        // Extract the features of the validation set
        // For each validation image
        for(j = 0; j < valid; j++)
        {
            // Get the image from the data list
            strcpy(fileName, data[i].getDataList(j+train));
            getSIFT(fileName, validObject[i].featureSet[j],
                    lvls, sigma, thresh1, thresh2, dbl, width, bins,
                    sizeMin, sizeMax);
        }

        // Extract the features of the test set
        // For each test image
        for(j = 0; j < test; j++)
        {
            // Get the image from the data list
            strcpy(fileName, data[i].getDataList(j+train+valid));
            getSIFT(fileName, testObject[i].featureSet[j],
                    lvls, sigma, thresh1, thresh2, dbl, width, bins,
                    sizeMin, sizeMax);
        }
	}

	return true;
}

bool BagOfFeatures::extractSURFFeatures(bool invariant,
                                        int octaves,
                                        int intervals,
                                        int step,
                                        float thresh,
                                        int sizeMin,
                                        int sizeMax)
{
    if(numFeatures > 0)
        return false;

	int i, j;
    int train, valid, test, label;

    char fileName[256];
    descrSize = 64;

	for(i = 0; i < numClasses; i++)
	{
	    // Get the distribution of data
        data[i].getDataInfo(train, valid, test, label);
	    // Extrain the features of the training set
        // For each training image
        for(j = 0; j < train; j++)
        {
            strcpy(fileName, data[i].getDataList(j));
            numFeatures += getSURF(fileName, trainObject[i].featureSet[j],
                            invariant, octaves, intervals, step, thresh,
                            sizeMin, sizeMax);
        }

        // Extrain the features of the validation set
        // For each validation image
        for(j = 0; j < valid; j++)
        {
            strcpy(fileName, data[i].getDataList(j+train));
            getSURF(fileName, validObject[i].featureSet[j],
                    invariant, octaves, intervals, step, thresh,
                    sizeMin, sizeMax);
        }

        // Extrain the features of the test set
        // For each test image
        for(j = 0; j < test; j++)
        {
            strcpy(fileName, data[i].getDataList(j+train+valid));
            getSURF(fileName, testObject[i].featureSet[j],
                    invariant, octaves, intervals, step, thresh,
                    sizeMin, sizeMax);
        }
	}

    return true;
}

bool BagOfFeatures::buildHierarchicalTree(int transpose,
                                          char dist,
                                          char method,
                                          double** distmatrix)
{
//double** buildMatrixHierarchical(const ImageFeatures *featurePts, const int feature_count,
//			const int image_count, Node*& tree )
    // Check to makes sure that features were found
    if(trainObject ==  NULL || numFeatures == 0 || descrSize == 0)
        return false;

	int i, j;
	int k = 0, l = 0, m;
	int size;
	int totalImages = 0;

    cout << "Initializing the data..." << endl;

	// Initialize the data
	hClusterData = new double* [numFeatures];
	for(i = 0; i < numFeatures; i++)
		hClusterData[i] = new double [descrSize];

    // Allocate mask and set it all to 1 (assume no missing data)
	int ** mask = new int* [numFeatures];
	for(i = 0; i < numFeatures; i++)
	{
	    mask[i] = new int [descrSize];
		for(j = 0; j < descrSize; j++)
			mask[i][j] = 1;
	}

	// Set the weights equal, all 1
	double* weight = new double [descrSize];
	for(i = 0; i < descrSize; i++)
		weight[i] = 1.0;


	// For each class
    for(m = 0; m < numClasses; m++)
    {
        totalImages = data[m].getTrainSize();
        // For each image in that class...
        for(l = 0; l < totalImages; l++)
        {
            size = trainObject[m].featureSet[l].size;
            // for each feature in that image...
            for(i = 0; i < size; i++)
            {
                // Copy the descriptor into the data array
                for(j = 0; j < descrSize; j++)
                {
                    hClusterData[k][j] = (double)trainObject[m].featureSet[l].descriptors[i][j];
                    //cout << hClusterData[k][j] << " ";
                }
                //cout << endl;
                k++;
            }
        }
    }

    cout << "Calculating the distance matrix..." << endl;

    distmatrix = distancematrix (descrSize, numFeatures,  hClusterData, mask, weight, dist, transpose);

    cout << "Building Hierarchical Tree" << endl;
	// Centroid Hierarchical Clustering
	// feature_count X DESCRIPTOR_SIZE
	// The feature vectors
	// mask (all 1s)
	// weights (all 1s)
	hTree = treecluster(descrSize, numFeatures, hClusterData, mask, weight, transpose, dist, method, distmatrix);

	// Release the mask
	for(i = 0; i < numFeatures; i++)
		delete [] mask[i];
	delete [] mask;
	// Release the weight
	delete [] weight;

    // Make sure that the tree was allocated
	if(!hTree)
	{
		cout << "Could not allocate the tree: Insufficient memory..." << endl;
		for(i = 0; i < numFeatures; i++)
            delete [] hClusterData[i];
        delete [] hClusterData;
		return false;
	}

    return true;
}

bool BagOfFeatures::cutHierarchicalTree(int numClusters)
{
    if(hClusterData == NULL || hTree == NULL)
        return false;

    if(dictionary != NULL)
        cvReleaseMat(&dictionary);
    int i, j, index;
    float *ptrCenter;

    int *clusterID = new int [numFeatures];
	int *indexCount = new int [numClusters];
	// initialize the count to zero
	for(i = 0; i < numClusters; i++)
		indexCount[i] = 0;

    dictionary = cvCreateMat(numClusters, descrSize, CV_32FC1);

    cvSetZero(dictionary);

	// Cluster the features based on the cluster_count
	cuttree(numFeatures, hTree, numClusters, clusterID);

    // Find the number of features in each cluster
    for(i = 0; i < numFeatures; i++)
    {
        index = clusterID[i];
        indexCount[index]++;
    }

	// Figure out how many clusters per index
	for(i = 0; i < numFeatures; i++)
	{
        index = clusterID[i];
		ptrCenter = (float *)(dictionary->data.ptr + index * dictionary->step);
		//cout << i << "\t";
		for(j = 0; j < descrSize; j++)
        {
            ptrCenter[j] += (float)hClusterData[i][j];
            cout << hClusterData[i][j] << " ";
        }
        cout << endl;
	}

	for(i = 0; i < numClusters; i++)
	{
        ptrCenter = (float *)(dictionary->data.ptr + i * dictionary->step);
        //cout << i << " \t\t\t" << indexCount[i] << endl << endl;
        float t = indexCount[i];
        for(j = 0; j < descrSize; j++)
        {
            ptrCenter[j] /= (float)indexCount[i];
        }
    }


/*
    int k;
    float *checkData = new float [descrSize];
    float minDist;
    float dist;
    int temp;
    int minIndex;

    for(i = 0; i < numFeatures; i++)
    {
        minDist = 999999.;
        for(j = 0; j < numClusters; j++)
        {
            ptrCenter = (float*)(dictionary->data.ptr + j*dictionary->step);
            for(k = 0; k < descrSize; k++)
            {
                checkData[k] = ptrCenter[k];
            }
            dist = 0;
            for(k = 0; k < descrSize; k++)
            {
                dist += (checkData[k] - hClusterData[i][k])*(checkData[k] - hClusterData[i][k]);
            }
            dist /= descrSize;//sqrt(dist);
            if(dist < minDist)
            {
                minDist = dist;
                minIndex = j;
            }
        }
        temp = clusterID[i];
        if(minIndex != clusterID[i])
            cout << "PROBLEM DURING CLUSTERING" << endl;
    }
    delete [] checkData;
*/

    delete [] clusterID;
	delete [] indexCount;
    return true;
}

bool BagOfFeatures::buildKMeans(int numClusters,
                                CvTermCriteria criteria = cvTermCriteria(
                                        CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,
                                        5,
                                        1.0),
                                int repeat=5)
{
    if(numFeatures == 0 || trainObject == NULL)
        return false;

    if(dictionary != NULL)
        cvReleaseMat(&dictionary);
	int i, j, k = 0, l = 0, m = 0;
	int size, index;
	int totalImages;
	int emptyClusters = 0;
	float* ptrRaw = NULL, *ptrCenter = NULL;
	int* ptrIndex = NULL;
	int* indexCount = NULL;

	// create a matrix that will  contain all the features
	CvMat* feature_mat = cvCreateMat(numFeatures, descrSize, CV_32FC1);
	CvMat* descriptor_clusters = cvCreateMat(numFeatures, 1, CV_32SC1);

	//keep track of how many descriptors there are in each cluster
	indexCount = new int [numClusters];
	// initialize the count to zero
	for(i = 0; i < numClusters; i++)
		indexCount[i] = 0;

    // For each class
    for(m = 0; m < numClasses; m++)
    {
        totalImages = data[m].getTrainSize();
        // For each image in that class...
        for(l = 0; l < totalImages; l++)
        {
            size = trainObject[m].featureSet[l].size;
            // for each feature in that image...
            for(i = 0; i < size; i++)
            {
                ptrRaw = (float *)(feature_mat->data.ptr + k * feature_mat->step);
                // put them in the raw descriptors matrix
                for(j = 0; j < descrSize; j++)
                {
                    ptrRaw[j] = trainObject[m].featureSet[l].descriptors[i][j];
                }
                k++;
            }
        }
    }

	// Cluster the raw matrix with a number of cluster found previously
	cvKMeans2( feature_mat, numClusters, descriptor_clusters, criteria,	repeat);
	// Repeat the clustering by CLUSTER_REPEAT times to get best results
    cout << "Done clustering... \nRegecting empty clusters..." << endl;
	// Figure out how many clusters per index
	for(i = 0; i < numFeatures; i++)
	{
		ptrIndex = (int *)(descriptor_clusters->data.ptr + i * descriptor_clusters->step);
		index = *ptrIndex;
		// increment the number of vectors found in that cluster
		indexCount[index]++;
	}

	// Find how many empty clusters there are
	for(i = 0; i < numClusters; i++)
	{
		if(indexCount[i] == 0)
		{
			emptyClusters++;
		}
	}

	// Descriptor cluster centers: This will look at all the clusters, even the empty
	CvMat* raw_cluster_centers = cvCreateMat(numClusters, descrSize, CV_32FC1);

	for(i = 0; i < numClusters; i++)
	{
		ptrCenter = (float *)(raw_cluster_centers->data.ptr + i * raw_cluster_centers->step);
		for(j = 0; j < descrSize; j++)
		{
			ptrCenter[j] = 0;
		}
	}

	cout << "Total Empty clusters found: " << emptyClusters
		<< " out of " << numClusters << " total clusters" << endl;
	// Calculate the cluster center for the descriptors
	for(i = 0; i < numFeatures; i++)
	{
		ptrRaw = (float *)(feature_mat->data.ptr + i * feature_mat->step);
		// This will give the cluster index number for each descriptor
		ptrIndex = (int *)(descriptor_clusters->data.ptr + i * descriptor_clusters->step);
		index = *ptrIndex;
		ptrCenter = (float *)(raw_cluster_centers->data.ptr + index * raw_cluster_centers->step);
		// Sum up the vectors for each cluster
		for(j = 0; j < descrSize; j++)
		{
			ptrCenter[j] += ptrRaw[j];
		}
	}

	dictionary = cvCreateMat(numClusters - emptyClusters, descrSize, CV_32FC1);
	cvSetZero(dictionary);
	k = 0;
	// Copy all the non-empty clusters to the cluster_center matrix
	// And output the clusters to the file
	for(i = 0; i < numClusters; i++)
	{
		ptrRaw = (float *)(raw_cluster_centers->data.ptr + i * raw_cluster_centers->step);
		if(indexCount[i] > 0)
		{
			ptrCenter = (float *)(dictionary->data.ptr + k * dictionary->step);
			//cout << i << " \t\t\t" << indexCount[i] << endl << endl;
			for(j = 0; j < descrSize; j++)
			{
				// Calulate the average by dividing by how many in that cluster
				ptrCenter[j] = (ptrRaw[j] / indexCount[i]);
			}
			k++;
		}
	}

	// Release all the matrices allocated
	cvReleaseMat(&feature_mat);
	cvReleaseMat(&descriptor_clusters);
	cvReleaseMat(&raw_cluster_centers);
	// Release the index count
	delete [] indexCount;

	return true;
}

bool BagOfFeatures::buildKClustering(int numClusters, int pass, char method, char dist)
{
    if(dictionary != NULL)
        cvReleaseMat(&dictionary);

    cout << "Initializing the data..." << endl;

    int i, j;
	int k = 0, l = 0, m = 0;
	int size;
	int totalImages = 0;
    double* error = new double [numFeatures];
    int ifound;
    int *clusterID = new int [numFeatures];
    double ** featureData = new double* [numFeatures];
    // Allocate mask and set it all to 1 (assume no missing data)
	int ** mask = new int* [numFeatures];
	for(i = 0; i < numFeatures; i++)
	{
	    mask[i] = new int [descrSize];
	    featureData[i] = new double [descrSize];
		for(j = 0; j < descrSize; j++)
			mask[i][j] = 1;
	}

	// Set the weights equal, all 1
	double* weight = new double [descrSize];
	for(i = 0; i < descrSize; i++)
		weight[i] = 1.0;

	// For each class
    for(m = 0; m < numClasses; m++)
    {
        totalImages = data[m].getTrainSize();
        // For each image in that class...
        for(l = 0; l < totalImages; l++)
        {
            size = trainObject[m].featureSet[l].size;
            // for each feature in that image...
            for(i = 0; i < size; i++)
            {
                // Copy the descriptor into the data array
                for(j = 0; j < descrSize; j++)
                {
                    featureData[k][j] = (double)trainObject[m].featureSet[l].descriptors[i][j];
                    //cout << featureData[k][j] << " ";
                }
                //cout << endl;
                k++;
            }
        }
    }

    cout << "Clustering data..." << endl;

    kcluster(numClusters, numFeatures, descrSize, featureData,
                mask, weight, 0, pass, method, dist,
                clusterID, error, &ifound);

    cout << "Computing cluster centers and building dictionary..." << endl;

    int* indexCount = new int [numClusters];
    int index;
    float *ptrCenter;

    dictionary = cvCreateMat(numClusters, descrSize, CV_32FC1);
    cvSetZero(dictionary);

    for(i = 0; i < numClusters; i++)
        indexCount[i] = 0;

	// Figure out how many clusters per index
	for(i = 0; i < numFeatures; i++)
	{
        index = clusterID[i];
        indexCount[index]++;
		ptrCenter = (float *)(dictionary->data.ptr + index * dictionary->step);
		for(j = 0; j < descrSize; j++)
        {
            ptrCenter[j] += (float)featureData[i][j];
        }
	}

	for(i = 0; i < numClusters; i++)
	{
        ptrCenter = (float *)(dictionary->data.ptr + i * dictionary->step);
        //cout << i << " \t\t\t" << indexCount[i] << endl << endl;
        float t = indexCount[i];
        for(j = 0; j < descrSize; j++)
        {
            ptrCenter[j] /= (float)indexCount[i];
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

    // Make sure that the tree was allocated
    return true;
}

bool BagOfFeatures::buildBofHistograms(bool normalize)
{
    if(dictionary == NULL)
        return false;

	int l, m;
	int count = dictionary->rows;
	int train, valid, test, label;

	// If the identity matrix is used for cvMahalanobis, then
	// the distance is equal to Euclidean distance
	//cvSetIdentity(identMat);

     // For each class
    for(m = 0; m < numClasses; m++)
    {
        // Get the information
        data[m].getDataInfo(train, valid, test, label);

        //Training Histograms
        // allocate the histogram of size "count", with label, and number of histograms "train"
        // Make sure it hasn't been allocated before
        for(l = 0; l < train; l++)
        {
            if(!trainObject[m].histogramSet[l].alloc(count, label))
            {
                trainObject[m].histogramSet[l].dealloc();
                trainObject[m].histogramSet[l].alloc(count, label);
            }
        }

        // For each training image in that class...
        for(l = 0; l < train; l++)
        {
            getHistogram(trainObject[m].featureSet[l], trainObject[m].histogramSet[l],
                        dictionary, descrSize, normalize);
        }
        // Validation Histograms:
        // allocate the histogram of size "count", with label, and number of histograms "train"
        // Make sure it hasn't been allocated before
        for(l = 0; l < valid; l++)
        {
            if(!validObject[m].histogramSet[l].alloc(count, label))
            {
                validObject[m].histogramSet[l].dealloc();
                validObject[m].histogramSet[l].alloc(count, label);
            }
        }
        // For each training image in that class...
        for(l = 0; l < valid; l++)
        {
            getHistogram(validObject[m].featureSet[l], validObject[m].histogramSet[l],
                        dictionary, descrSize, normalize);
        }
        // test Histograms:
        // allocate the histogram of size "count", with label, and number of histograms "train"
        // Make sure it hasn't been allocated before
        for(l = 0; l < test; l++)
        {
            if(!testObject[m].histogramSet[l].alloc(count, label))
            {
                testObject[m].histogramSet[l].dealloc();
                testObject[m].histogramSet[l].alloc(count, label);
            }
        }
        // For each training image in that class...
        for(l = 0; l < test; l++)
        {
            getHistogram(testObject[m].featureSet[l], testObject[m].histogramSet[l],
                        dictionary, descrSize, normalize);
        }
    }

    return true;
}

void BagOfFeatures::filterDictionary(double R)
{
    int i, j, k;
    int trainSize;
    //Initialize the system
    double* meanHist = new double [dictionary->rows];
    double* stdevHist = new double [dictionary->rows];
    for(i = 0; i < dictionary->rows; j++)
    {
        meanHist[i] = 0;
        stdevHist[j] = 0;
    }
    double** totalHist = new double* [numClasses];
    for(i = 0; i < numClasses; i++)
    {
        totalHist[i] = new double [dictionary->rows];
        for(j = 0; j < dictionary->rows; j++)
        {
            totalHist[i][j] = 0;
        }
    }

    // Compute the total histograms for each class
    for(i = 0; i < numClasses; i++)
    {
        trainSize = data[i].getTrainSize();
        for(j = 0; j < trainSize; j++)
            for(k = 0; k < dictionary->rows; k++)
                totalHist[i][k] += trainObject[i].histogramSet[j].histogram[k];
        //Compute the average
        for(k = 0; k < dictionary->rows; k++)
            meanHist[k] += totalHist[i][k]/(double)numClasses;
    }

    //Compute the standard deviation
    for(i = 0; i < numClasses; i++)
    {
        for(j = 0; j < dictionary->rows; j++)
            stdevHist[j] += pow(totalHist[i][j] - meanHist[j], 2.)/(double)(numClasses);
    }

    ofstream statFile;
    statFile.open("DictionaryStats");

    for(i = 0; i < dictionary->rows; i++)
    {
        statFile << meanHist[i] << "\t" << stdevHist[i] << endl;
    }

    delete [] meanHist;
    delete [] stdevHist;
    for(i = 0; i < numClasses; i++)
    {
        delete [] totalHist[i];
    }
    delete [] totalHist;
}


void BagOfFeatures::trainSVM(int type = NU_SVC,
                            int kernel = RBF,
                            double degree = 0.05,
                            double gamma = 0.25,
                            double coef0 = 0.5,
                            double C = .05,
                            double cache = 300,
                            double eps = 0.000001,
                            double nu = 0.5,
                            int shrinking = 0,
                            int probability = 0,
                            int weight = 0)
{
    if(SVMModel != NULL)
    {
        svm_destroy_model(SVMModel);
        //svm_destroy_param(&SVMParam);
    }

    int i, j, k, l = -1;
    int totalData = 0;
    int size, length = dictionary->rows;
    int count;
    //Get the total number of training data
    for(i = 0; i < numClasses; i++)
        totalData += data[i].getTrainSize();

    // Set up the data
    struct svm_problem SVMProblem;
    SVMProblem.l = totalData;
    SVMProblem.y = new double [totalData];
    SVMProblem.x = new struct svm_node* [totalData];
    // Allocate memory
    //for(i = 0; i < totalData; i++)
    //{
    //    SVMProblem.x[i] = new struct svm_node [length+1];
    //}

    // For each class
    for(i = 0; i < numClasses; i++)
    {
        // Get the number of images
        size = data[i].getTrainSize();
        for(j = 0; j < size; j++)
        {
            l++;
            count = 0;
            for(k = 0; k < length; k++)
            {
                if(trainObject[i].histogramSet[j].histogram[k] != 0)
                    count++;
            }
            SVMProblem.x[l] = new struct svm_node [count+1];
            count = 0;
            for(k = 0; k < length; k++)
            {
                if(trainObject[i].histogramSet[j].histogram[k] != 0)
                {
                    SVMProblem.x[l][count].index = k+1;
                    SVMProblem.x[l][count].value = trainObject[i].histogramSet[j].histogram[k];
                    //cout << "(" << SVMProblem.x[l][count].index
                    //    << ", " << SVMProblem.x[l][count].value << ")" << endl;
                    count++;
                }
            }
            SVMProblem.x[l][count].index = -1;
            //cout << endl;
            //SVMProblem.x[l][count].value = -1;
            // Copy the histograms
            //for(k = 0; k < length; k++)
            //{
            //    SVMProblem.x[l][k].index = k;
            //    SVMProblem.x[l][k].value = trainObject[i].histogramSet.histogram[j][k];
            //}
            // End of the data
            //SVMProblem.x[l][length].index = -1;
            //SVMProblem.x[l][length].value = -1;
            //Attach the labels

            SVMProblem.y[l] = data[i].getLabel();
            //cout << "Label: " << SVMProblem.y[l] << endl;
        }
    }

    // Types
    SVMParam.svm_type = type;
    SVMParam.kernel_type = kernel;
    // Parameters
    SVMParam.degree = degree;
    SVMParam.gamma = gamma;
    SVMParam.coef0 = coef0;
    SVMParam.C = C;
    // For training only
    SVMParam.cache_size = cache;
    SVMParam.eps = eps;
    SVMParam.nu = nu;
    SVMParam.shrinking = shrinking;
    SVMParam.probability = probability;
    // Don't change the weights
    SVMParam.nr_weight = weight;


    double* target = new double [totalData];
    svm_check_parameter(&SVMProblem, &SVMParam);
    svm_cross_validation(&SVMProblem, &SVMParam, 10, target);
    SVMModel = svm_train(&SVMProblem, &SVMParam);
    delete [] target;
    svm_save_model("svmSURF800",SVMModel);
    classifierType = LIBSVM_CLASSIFIER;

}

CvSVM* BagOfFeatures::trainSVM_CV(int type, int kernel, double degree, double gamma, double coef0,
                        double C, double nu, double p, int termType, int iterations, double eps,
                        char* fileName)
{
    int i, j, k, l = -1;
    int totalData = 0;
    int size, length = dictionary->rows;

    //float *dPtr;
    //Get the total number of training data
    for(i = 0; i < numClasses; i++)
        totalData += data[i].getTrainSize();

    //CvMat* trainData = cvCreateMat(totalData, dictionary->rows, CV_32FC1);
    //CvMat* dataLabel = cvCreateMat(totalData, 1, CV_32FC1);

    float** trainData = new float* [totalData];
    float* dataLabel = new float [totalData];
    for(i = 0; i < totalData; i++)
        trainData[i] = new float [dictionary->rows];

     // For each class
    for(i = 0; i < numClasses; i++)
    {
        // Get the number of images
        size = data[i].getTrainSize();
        for(j = 0; j < size; j++)
        {
            l++;
            //Attach the label to it
            //dataLabel->data.fl[l] = (float)data[i].getLabel();
            //dPtr = (float*)(trainData->data.ptr + l*trainData->step);
            dataLabel[l] = (float)data[i].getLabel();
            // Copy the histograms
            for(k = 0; k < length; k++)
            {
                //dPtr[k] = trainObject[i].histogramSet.histogram[j][k];
                trainData[l][k] = trainObject[i].histogramSet[j].histogram[k];
            }
        }
    }

    CvSVMParams SVMParam_CV;
    SVMParam_CV.svm_type = type;
    SVMParam_CV.kernel_type = kernel;
    SVMParam_CV.degree = degree;
    SVMParam_CV.gamma = gamma;
    SVMParam_CV.coef0 = coef0;
    SVMParam_CV.C = C;
    SVMParam_CV.nu = nu;
    SVMParam_CV.p = p;
    SVMParam_CV.class_weights = NULL;
    SVMParam_CV.term_crit = cvTermCriteria(termType, iterations, eps);

    CvMat *dataHeader = cvCreateMatHeader(totalData, dictionary->rows, CV_32FC1);
	CvMat *labelHeader = cvCreateMatHeader(totalData, 1, CV_32FC1);
    cvInitMatHeader(dataHeader, totalData, dictionary->rows, CV_32FC1, trainData);
	cvInitMatHeader(labelHeader, totalData, 1, CV_32FC1, dataLabel);
    //Train the SVM
    //CvSVM svm(trainData, dataLabel, 0, 0,
    //    CvSVMParams(CvSVM::C_SVC, CvSVM::LINEAR, 0, 0, 0, 2,
     //   0, 0, 0, cvTermCriteria(CV_TERMCRIT_EPS,0, 0.01)));

    strcpy(classifierFile, fileName);
    //if(SVMModel_CV != NULL)
    //    delete SVMModel_CV;
    CvSVM *SVMModel_CV = new CvSVM;
    SVMModel_CV->clear();
    SVMModel_CV->train_auto(dataHeader, labelHeader, 0, 0, SVMParam_CV, 10);
    SVMModel_CV->save(classifierFile);

    cvReleaseMatHeader(&dataHeader);
    cvReleaseMatHeader(&labelHeader);
    for(i = 0; i < totalData; i++)
        delete [] trainData[i];
    delete [] trainData;
    delete [] dataLabel;
    classifierType = CVSVM_CLASSIFIER;

    return SVMModel_CV;
}

CvNormalBayesClassifier* BagOfFeatures::trainNormBayes_CV()
{
    int i, j, k, l = -1;
    int totalData = 0;
    int size, length = dictionary->rows;

    float *dPtr;
    //Get the total number of training data
    for(i = 0; i < numClasses; i++)
        totalData += data[i].getTrainSize();

    CvMat* trainData = cvCreateMat(totalData, dictionary->rows, CV_32FC1);
    CvMat* dataLabel = cvCreateMat(totalData, 1, CV_32SC1);

    ofstream histFile;
    histFile.open("Histograms");
     // For each class
    for(i = 0; i < numClasses; i++)
    {
        // Get the number of images
        size = data[i].getTrainSize();
        for(j = 0; j < size; j++)
        {
            l++;
            //Attach the label to it
            CV_MAT_ELEM(*dataLabel, int, l, 0) = (int)data[i].getLabel();
            histFile << CV_MAT_ELEM(*dataLabel, int, l, 0) << endl;
            dPtr = (float*)(trainData->data.ptr + l*trainData->step);
            // Copy the histograms
            for(k = 0; k < length; k++)
            {
                dPtr[k] = trainObject[i].histogramSet[j].histogram[k];
                histFile << dPtr[k] << "\t";
            }
            histFile << endl;
        }
    }

    CvNormalBayesClassifier *NBModel_CV = new CvNormalBayesClassifier;
       //if(NBModel_CV != NULL)
    //    delete NBModel_CV;
    NBModel_CV->clear();
    //CvNormalBayesClassifier NBModel_CV(trainData, dataLabel, 0, 0);
    NBModel_CV->train(trainData, dataLabel, NULL, NULL, false);
    NBModel_CV->save("NB.xml");

    cvReleaseMat(&trainData);
    cvReleaseMat(&dataLabel);

    classifierType = CVNORM_BAYES_CLASSIFIER;

    return NBModel_CV;

}

/*
float* BagOfFeatures::resultsTraining()
{
    int i, j, k;
    int size;
    float classification;
    double t;
    float* results = new float [numClasses];

    for(i = 0; i < numClasses; i++)
    {
        results[i] = 0;
        size = data[i].getTrainSize();
        for(j = 0; j < size; j++)
        {
            if(classifierType == LIBSVM_CLASSIFIER)
            {
                struct svm_node* trainData = new struct svm_node [dictionary->rows+1];
                for(k = 0; k < dictionary->rows; k++)
                {
                    trainData[k].index = k;
                    trainData[k].value = trainObject[i].histogramSet.histogram[j][k];
                }
                trainData[k].index = -1;

                double *values = new double [numClasses*(numClasses - 1)/2];
                classification = svm_predict_values(SVMModel, trainData, values);
                t = fabs((double)data[i].getLabel()-classification);
                if(t < .5)
                    results[i]++;

                delete [] trainData;
                delete [] values;
            }
            else if(classifierType == CVSVM_CLASSIFIER)
            {
                CvMat* sample = cvCreateMat(1, dictionary->rows, CV_32FC1);
                for(k = 0; k < dictionary->rows; k++)
                {
                    CV_MAT_ELEM(*sample, float, 0, k) = trainObject[i].histogramSet.histogram[j][k];//assign value
                }

                classification = SVMModel_CV.predict(sample);
                t = fabs((float)data[i].getLabel()-classification);
                if(t < .5)
                    results[i]++;

                cvReleaseMat(&sample);
            }
            else if(classifierType == CVNORM_BAYES_CLASSIFIER)
            {
                CvMat* sample = cvCreateMat(1, dictionary->rows, CV_32FC1);
                for(k = 0; k < dictionary->rows; k++)
                {
                    CV_MAT_ELEM(*sample, float, 0, k) = trainObject[i].histogramSet.histogram[j][k];//assign value
                }

                classification = NBModel_CV.predict(sample);
                t = fabs((float)data[i].getLabel()-classification);
                if(t < .5)
                    results[i]++;

                cvReleaseMat(&sample);

            }

        }
        results[i] /= (float)size;
    }

    return results;
}
*/

float* BagOfFeatures::resultsTraining()
{
    int i, j, k;
    int size;
    float classification;
    double t;
    float* results = new float [numClasses];
    for(i = 0; i < numClasses; i++)
    {
        results[i] = 0;
        size = data[i].getTrainSize();
        for(j = 0; j < size; j++)
        {
            struct svm_node* trainData = new struct svm_node [dictionary->rows+1];
            for(k = 0; k < dictionary->rows; k++)
            {
                trainData[k].index = k+1;
                trainData[k].value = trainObject[i].histogramSet[j].histogram[k];
            }
            trainData[k].index = -1;

            double *values = new double [numClasses*(numClasses - 1)/2];
            classification = svm_predict_values(SVMModel, trainData, values);
            t = fabs((double)data[i].getLabel()-classification);
            if(t < .5)
                results[i]++;

            delete [] trainData;
            delete [] values;
        }
        results[i] /= size;
    }
    return results;
}

float* BagOfFeatures::resultsTraining(CvSVM *SVMModel_CV)
{
    int i, j, k;
    int size;
    float classification;
    double t;
    float* results = new float [numClasses];
    for(i = 0; i < numClasses; i++)
    {
        results[i] = 0;
        size = data[i].getTrainSize();
        for(j = 0; j < size; j++)
        {
            CvMat* sample = cvCreateMat(1, dictionary->rows, CV_32FC1);
            for(k = 0; k < dictionary->rows; k++)
            {
                CV_MAT_ELEM(*sample, float, 0, k) = trainObject[i].histogramSet[j].histogram[k];//assign value
            }

            classification = SVMModel_CV->predict(sample);
            t = fabs((float)data[i].getLabel()-classification);
            if(t < .5)
                results[i]++;

            cvReleaseMat(&sample);
        }

        results[i] /= (float)size;
    }
    return results;
}

float* BagOfFeatures::resultsTraining(CvNormalBayesClassifier *NBModel_CV)
{
    int i, j, k;
    int size;
    float classification;
    float t;
    float* results = new float [numClasses];
    for(i = 0; i < numClasses; i++)
    {
        results[i] = 0;
        size = data[i].getTrainSize();
        CvMat* sample = cvCreateMat(size, dictionary->rows, CV_32FC1);
        CvMat* output = cvCreateMat(1, size, CV_32FC1);
        cvZero(output);
        for(j = 0; j < size; j++)
        {
            for(k = 0; k < dictionary->rows; k++)
            {
                CV_MAT_ELEM(*sample, float, j, k) = trainObject[i].histogramSet[j].histogram[k];//assign value
            }
        }

        NBModel_CV->predict(sample, output);
        for(j = 0; j < size; j++)
        {
            float* ptr = (float*)(output->data.ptr);
            classification = ptr[j];
            t = fabs((float)data[i].getLabel()-classification);
            if(t < .5)
                results[i]++;
        }

        results[i] /= (float)size;
        cvReleaseMat(&sample);
        cvReleaseMat(&output);

    }

    return results;
}

float* BagOfFeatures::resultsTest()
{
    int i, j, k;
    int size;
    float classification;
    float t;
    float* results = new float [numClasses];
    for(i = 0; i < numClasses; i++)
    {
        results[i] = 0;
        size = data[i].getTestSize();
        for(j = 0; j < size; j++)
        {
            struct svm_node* testData = new struct svm_node [dictionary->rows+1];
            for(k = 0; k < dictionary->rows; k++)
            {
                testData[k].index = k+1;
                testData[k].value = testObject[i].histogramSet[j].histogram[k];
            }
            testData[k].index = -1;

            double* values = new double [numClasses*(numClasses - 1)/2];
            classification = svm_predict_values(SVMModel, testData, values);
            //svm_check_probability_model(SVMModel);
            //svm_predict_probability(SVMModel, testData, values);
            //for(k = 0; k < numClasses*(numClasses-1)/2; k++)
            //   classification = values[k];
            t = fabs((float)data[i].getLabel()-classification);
            if(t < .5)
                results[i]++;

            delete [] testData;
            delete [] values;

        }
        results[i] /= (float)size;
    }

    return results;
}

float BagOfFeatures::predictClassification(ImageFeatures input, bool normalize)
{
    int i;
    float classification;
    HistogramFeatures bof(dictionary->rows, -1);
    //Build the bag of feature
    getHistogram(input, bof, dictionary, descrSize, normalize);
    //Classify
    if(classifierType == LIBSVM_CLASSIFIER)
    {
        struct svm_node* testData = new struct svm_node [dictionary->rows+1];
        for(i = 0; i < dictionary->rows; i++)
        {
            testData[i].index = i+1;
            testData[i].value = bof.histogram[i];
        }
        testData[i].index = -1;

        double* values = new double [numClasses*(numClasses - 1)/2];
        classification = svm_predict_values(SVMModel, testData, values);
        delete [] testData;
        delete [] values;
    }
    return classification;
}

/*
float* BagOfFeatures::resultsValidation()
{
    int i, j, k;
    int size;
    float classification;
    float t;

    float* results = new float [numClasses];

    for(i = 0; i < numClasses; i++)
    {
        results[i] = 0;
        size = data[i].getValidSize();
        for(j = 0; j < size; j++)
        {
            if(classifierType == LIBSVM_CLASSIFIER)
            {
                struct svm_node* validData = new struct svm_node [dictionary->rows+1];
                for(k = 0; k < dictionary->rows; k++)
                {
                    validData[k].index = k;
                    validData[k].value = validObject[i].histogramSet.histogram[j][k];
                }
                validData[k].index = -1;

                double *values = new double [numClasses*(numClasses - 1)/2];
                classification = svm_predict_values(SVMModel, validData, values);
                t = fabs((float)data[i].getLabel()-classification);
                if(t < .5)
                    results[i]++;

                delete [] validData;
                delete [] values;
            }
            else if(classifierType == CVSVM_CLASSIFIER)
            {
                CvMat* sample = cvCreateMat(1, dictionary->rows, CV_32FC1);
                for(k = 0; k < dictionary->rows; k++)
                {
                    CV_MAT_ELEM(*sample, float, 0, k) = validObject[i].histogramSet.histogram[j][k];//assign value
                }

                classification = SVMModel_CV.predict(sample);
                t = fabs((float)data[i].getLabel()-classification);
                if(t < .5)
                    results[i]++;

                cvReleaseMat(&sample);
            }
            else if(classifierType == CVNORM_BAYES_CLASSIFIER)
            {
                CvMat* sample = cvCreateMat(1, dictionary->rows, CV_32FC1);
                for(k = 0; k < dictionary->rows; k++)
                {
                    CV_MAT_ELEM(*sample, float, 0, k) = validObject[i].histogramSet.histogram[j][k];//assign value
                }

                classification = NBModel_CV.predict(sample);
                t = fabs((float)data[i].getLabel()-classification);
                if(t < .5)
                    results[i]++;

                cvReleaseMat(&sample);

            }

        }
        results[i] /= (float)size;
    }

    return results;
}
*/
/*
float* BagOfFeatures::resultsTest()
{
    int i, j, k;
    int size;
    float classification;
    float t;
    float* results = new float [numClasses];
    for(i = 0; i < numClasses; i++)
    {
        results[i] = 0;
        size = data[i].getTestSize();
        for(j = 0; j < size; j++)
        {
           if(classifierType == LIBSVM_CLASSIFIER)
            {
                struct svm_node* testData = new struct svm_node [dictionary->rows+1];
                for(k = 0; k < dictionary->rows; k++)
                {
                    testData[k].index = k;
                    testData[k].value = testObject[i].histogramSet.histogram[j][k];
                }
                testData[k].index = -1;

                double* values = new double [numClasses*(numClasses - 1)/2];
                classification = svm_predict_values(SVMModel, testData, values);
                //svm_check_probability_model(SVMModel);
                //svm_predict_probability(SVMModel, testData, values);
                //for(k = 0; k < numClasses*(numClasses-1)/2; k++)
                //   classification = values[k];
                t = fabs((float)data[i].getLabel()-classification);
                if(t < .5)
                    results[i]++;

                delete [] testData;
                delete [] values;
            }
            else if(classifierType == CVSVM_CLASSIFIER)
            {
                CvMat* sample = cvCreateMat(1, dictionary->rows, CV_32FC1);
                for(k = 0; k < dictionary->rows; k++)
                {
                    CV_MAT_ELEM(*sample, float, 0, k) = testObject[i].histogramSet.histogram[j][k];//assign value
                }

                classification = SVMModel_CV.predict(sample);
                t = fabs((float)data[i].getLabel()-classification);
                if(t < .5)
                    results[i]++;

                cvReleaseMat(&sample);
            }
             else if(classifierType == CVNORM_BAYES_CLASSIFIER)
            {
                CvMat* sample = cvCreateMat(1, dictionary->rows, CV_32FC1);
                for(k = 0; k < dictionary->rows; k++)
                {
                    CV_MAT_ELEM(*sample, float, 0, k) = testObject[i].histogramSet.histogram[j][k];//assign value
                }

                classification = NBModel_CV.predict(sample);
                t = fabs((float)data[i].getLabel()-classification);
                if(t < .5)
                    results[i]++;

                cvReleaseMat(&sample);
            }
        }
        results[i] /= (float)size;
    }

    return results;
}
*/


void copySIFTPts(ImageFeatures &dst, feature* src, const int size, const int length)
{
	int i, j;
	// Check to make sure it hasn't been allocated yet
	if(dst.checkAlloc())
		dst.dealloc();
	// Allocate some memory
	dst.alloc(length, size);
	for(i = 0; i < size; i++)
	{
	    float descript[length];
	    //double max = 0.0;
	    // First step is to normalize the sift vector
	    // Find the magnitude of the vector
	    //for(j = 0; j < length; j++)
	    //{
	    //    max += (float)(src[i].descr)[j];
	    //}
	    //double magnitude = sqrt(magnitude);
	    // Normalize by dividing by magnitude
	    for(j = 0; j < length; j++)
	    {
	        descript[j] = (float)(src[i].descr)[j];// / magnitude; /// max;
	        //cout << descript[j] << "\t";
	    }
	    //cout << endl << endl;
	    // Copy the descriptor
		dst.copyDescriptorAt(descript, i);
	}
}

int getSIFT(char* fileName, ImageFeatures &dst,
            int lvls, double sigma, double thresh1, int thresh2,
            int dbl, int width, int bins, int sizeMin, int sizeMax)
{
    int count;
    IplImage* dataImage;
    cout << "Loading image: " << fileName << endl;
    dataImage = cvLoadImage(fileName);
    IplImage *dataGray = cvCreateImage(cvSize(dataImage->width, dataImage->height), 8, 1);
    // Convert to grayscale
    cvCvtColor(dataImage, dataGray, CV_BGR2GRAY);

    IplImage *resized = preProcessImages(dataGray, sizeMin, sizeMax);

    struct feature *siftFeatures = NULL;
    // Extract the sift features from the images
    //Default:  SIFT_INTVLS, SIFT_SIGMA, SIFT_CONTR_THR, SIFT_CURV_THR,
    //          SIFT_IMG_DBL, SIFT_DESCR_WIDTH, SIFT_DESCR_HIST_BINS
    count = _sift_features(resized, &siftFeatures, lvls, sigma, thresh1, thresh2, dbl, width, bins);
    cout << "Found " << count << " SIFT interest points" << endl;

    // Copy the descriptors into the feature set
    copySIFTPts(dst, siftFeatures, count, width*width*bins);

    // Release Memory
    free(siftFeatures);
    cvReleaseImage(&dataImage);
    cvReleaseImage(&dataGray);
    cvReleaseImage(&resized);

    return count;
}

int getSIFT(IplImage* input, ImageFeatures &dst, int lvls,
            double sigma, double thresh1, int thresh2,
            int dbl, int width, int bins, int sizeMin, int sizeMax)
{
    int count;
    struct feature *siftFeatures = NULL;
    IplImage *dataGray = cvCreateImage(cvSize(input->width, input->height), 8, 1);
    // Convert to grayscale
    cvCvtColor(input, dataGray, CV_BGR2GRAY);

    IplImage *resized = preProcessImages(dataGray, sizeMin, sizeMax);
    // Extract the sift features from the images
    //Default:  SIFT_INTVLS, SIFT_SIGMA, SIFT_CONTR_THR, SIFT_CURV_THR,
    //          SIFT_IMG_DBL, SIFT_DESCR_WIDTH, SIFT_DESCR_HIST_BINS
    count = _sift_features(input, &siftFeatures, lvls, sigma, thresh1, thresh2, dbl, width, bins);
    //cout << "Found " << count << " SIFT interest points" << endl;
    // Copy the descriptors into the feature set
    copySIFTPts(dst, siftFeatures, count, width*width*bins);
    // Release Memory
    free(siftFeatures);
    cvReleaseImage(&dataGray);
    cvReleaseImage(&resized);

    return count;
}


void copySURFPts(ImageFeatures &dst, const IpVec src, const int length)
{
	int i, size;
	Ipoint temp;
	size = src.size();
	//Check if the object has been allocated
	//Deallocate it first
	if(dst.checkAlloc())
			dst.dealloc();
	// Allocated with the correct values
	dst.alloc(length, size);
	for(i = 0; i < size; i++)
	{
		temp = src.at(i);
		//float mag = 0;
		//for(j = 0; j < length; j++)
        //    mag += temp.descriptor[j]*temp.descriptor[j];
        //mag = sqrt(mag);
        //for(j = 0; j < length; j++)
        //    temp.descriptor[j] *= mag;
		// Copy each descriptor into the ImageFeature
		dst.copyDescriptorAt(temp.descriptor, i);

	}
}

int getSURF(char* fileName, ImageFeatures &dst, bool invariant,
            int octaves, int intervals, int step, float thresh,
            int sizeMin, int sizeMax)
{
    IpVec temp;
    IplImage* dataImage = NULL;

    cout << "Loading image: " << fileName << endl;
    dataImage = cvLoadImage(fileName);
    IplImage *dataGray = cvCreateImage(cvSize(dataImage->width, dataImage->height), 8, 1);
    // Convert to grayscale
    cvCvtColor(dataImage, dataGray, CV_BGR2GRAY);

    //Resize the images
    IplImage *resized = preProcessImages(dataGray, sizeMin, sizeMax);

    // Detect the SURF features
    surfDetDes(resized, temp, invariant, octaves, intervals, step, thresh);

    cout << "OpenSURF found: " << temp.size() << " interest points" << endl;
    /*
    drawIpoints(resized, temp, 3);

    IplImage* display = cvCreateImage(cvSize(resized->width*4, resized->height*4), resized->depth, resized->nChannels);
    cvResize(resized, display, CV_INTER_CUBIC);
    cvShowImage("Extracted SURF", display);
    cvWaitKey(150);
    cvReleaseImage(&display);
    */
    // Copy the SURF feature into the feature object
    copySURFPts(dst, temp, 64);

    cvReleaseImage(&dataImage);
    cvReleaseImage(&dataGray);
    cvReleaseImage(&resized);

    return temp.size();
}

int getSURF(IplImage* input, ImageFeatures &dst, bool invariant,
            int octaves, int intervals, int step, float thresh,
            int sizeMin, int sizeMax)
{
    IpVec temp;
    IplImage *dataGray = cvCreateImage(cvSize(input->width, input->height), 8, 1);
    // Convert to grayscale
    cvCvtColor(input, dataGray, CV_BGR2GRAY);

    //Resize the images
    IplImage *resized = preProcessImages(dataGray, sizeMin, sizeMax);

    // Detect the SURF features
    surfDetDes(resized, temp, invariant, octaves, intervals, step, thresh);
    //cout << "Extracted " << temp.size() << " features..." << endl;
    /*
    drawIpoints(resized, temp, 3);
    IplImage* display = cvCreateImage(cvSize(resized->width*4, resized->height*4), resized->depth, resized->nChannels);
    cvResize(resized, display, CV_INTER_CUBIC);
    cvShowImage("Extracted SURF", display);
    cvWaitKey(60);
    cvReleaseImage(&display);
    */
    // Copy the SURF feature into the feature object
    copySURFPts(dst, temp, 64);
    cvReleaseImage(&dataGray);
    cvReleaseImage(&resized);
    return temp.size();
}

void getHistogram(ImageFeatures src, HistogramFeatures &dst, CvMat* dict, int len, bool normalize)
{
    int i;
    int minIndex;
    int size = src.size;
    // For each features in that image
    for(i = 0; i < size; i++)
    {
        // Find the best match, and add it to the bin of the histogram
        minIndex = findDictionaryMatch(src.descriptors[i], dict, len);
        // Increment the histogram where the vector belongs
        dst.addToBin(minIndex);
    }
    if(normalize)
        dst.normalizeHist();
}

int findDictionaryMatch(float* descriptor, CvMat* dict, int length)
{
    int j, k;
    int count = dict->rows;
    int minIndex;

    double minDistance = 999999.;
	double tempDistance = 0;

    CvMat* vect1 = cvCreateMat(1, length, CV_32FC1);
	CvMat* vect2 = cvCreateMat(1, length, CV_32FC1);
	//CvMat* id = cvCreateMat(length, length, CV_32FC1);
	//cvSetIdentity(id);

	float* ptr1 = NULL;
	float* ptr2 = NULL;

    ptr1 = (float *)(vect1->data.ptr);
    // Copy the vector in the vector 1;
    for(j = 0; j < length; j++)
    {
        ptr1[j] = descriptor[j];
    }

    // For each dictionary word
    for(j = 0; j < count; j++)
    {
        //cvGetRow(dict, vect2, j);
        //tempDistance = cvMahalanobis(vect1, vect2, id);
        // Get the second vector (word from the list)
        ptr2 = (float*)(vect2->data.ptr);
        for(k = 0; k < length; k++)
        {
            ptr2[k] = CV_MAT_ELEM(*dict, float, j, k);
        }
        tempDistance = 0;
        for(k = 0; k < length; k++)
        {
            tempDistance += (ptr2[k] - ptr1[k])*(ptr2[k] - ptr1[k]);
        }
        // calculate the euclidean distance
        tempDistance /= (double)length; //sqrt(tempDistance);*/

        if(tempDistance < minDistance)
        {
            // get the smallest distance and keep track of the index of the min
            minDistance = tempDistance;
            minIndex = j;
        }
    }

    cvReleaseMat(&vect1);
	cvReleaseMat(&vect2);

    return minIndex;
}

IplImage* preProcessImages(const IplImage* input, int minSize, int maxSize)
{
    int width = input->width;
    int height = input->height;
    int minSide;
    double ratio;

    if(width < height)
        minSide = width;
    else
        minSide = height;

    if(minSide < minSize)
        ratio = (double)minSize / (double)minSide;
    else if(minSide > maxSize)
        ratio = (double)maxSize / (double)minSide;
    else
        ratio = 1.0;

    IplImage* temp = cvCreateImage(cvSize(width*ratio, height*ratio), input->depth, input->nChannels);
    IplImage* output = cvCreateImage(cvSize(width*ratio, height*ratio), input->depth, input->nChannels);
    //Resize based on the ratio
    cvResize(input, temp, CV_INTER_AREA);
    //Equalize the histograms of the images

    //cvEqualizeHist(temp, output);
    cvNormalize(temp, output, 0, 255, CV_MINMAX);

    cvReleaseImage(&temp);

    return output;
}
