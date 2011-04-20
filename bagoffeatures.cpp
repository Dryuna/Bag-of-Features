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
    params.numClasses = 0;
    params.numFeatures = 0;
    params.numImages = 0;
    params.featureLength = 0;
    testObject = NULL;
    validObject = NULL;
    trainObject = NULL;
    SVMModel = NULL;
    //SVMModel_CV = NULL;
    //NBModel_CV = NULL;
}

BagOfFeatures::BagOfFeatures(BoFParameters p, DataSet* val)
{
    int i;

    params = p;
    SVMModel = NULL;

    testObject = new ObjectSet [params.numClasses];
    validObject = new ObjectSet [params.numClasses];
    trainObject = new ObjectSet [params.numClasses];
    data = new DataSet [params.numClasses];

    int train, valid, test, label;

    for(i = 0; i < params.numClasses; i++)
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
    delete [] testObject;
    delete [] validObject;
    delete [] trainObject;
    delete [] data;
    if(SVMModel)
        svm_destroy_model(SVMModel);
    //SVMModel_CV.clear();
    //NBModel_CV.clear();
}

void BagOfFeatures::alloc(BoFParameters p, DataSet* val)
{
    int i;
    if(data != NULL)
    {
        delete [] testObject;
        delete [] validObject;
        delete [] trainObject;
        delete [] data;
    }

    params = p;
    if(SVMModel)
        svm_destroy_model(SVMModel);
    SVMModel = NULL;

    testObject = new ObjectSet [params.numClasses];
    validObject = new ObjectSet [params.numClasses];
    trainObject = new ObjectSet [params.numClasses];
    data = new DataSet [params.numClasses];

    int train, valid, test, label;

    for(i = 0; i < params.numClasses; i++)
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


void BagOfFeatures::extractFeatures(ImageFeatures &f, cv::Mat img)
{
    cv::Mat processed;
    params.(*preprocess)(img, processed);

    if(params.featureType == FEATURES_SIFT)
    {
        f.extractSIFT_CV(processed,
                    params.siftParams.detectionThreshold,
                    params.siftParams.edgeThreshold,
                    params.verbose);
    }
    else if(params.featureType == FEATURES_SURF)
    {
        f.extractSURF_CV(processed,
                    params.surfParams.hessianThreshold,
                    params.surfParams.nOctives,
                    params.surfParams.nLayers,
                    params.surfParams.extended,
                    params.verbose);
    }
}


void BagOfFeatures::trainSVM_CV()
{
    int i, j, k, l = -1;
    int totalData = 0;

    //Get the total number of training data
    for(i = 0; i < params.numClasses; i++)
        totalData += data[i].getTrainSize();

    cv::Mat trainData(totalData, codex.size, CV_32FC1);
    cv::Mat dataLabel(totalData, 1, CV_32FC1);

    // For each class
    for(i = 0; i < params.numClasses; i++)
    {
        // Get the number of images
        int size = data[i].getTrainSize();
        for(j = 0; j < size; j++)
        {
            l++;
            float* lPtr = dataLabel.ptr<float>(l);
            lPtr[0] = (float)data[i].getLabel();
            //cout << lPtr[0] << " ";
            float* dPtr = trainData.ptr<float>(l);
            // Copy the histograms
            for(k = 0; k < codex.size; k++)
            {
                dPtr[k] = trainObject[i].histogramSet[j].histogram[k];
                //cout << dPtr[k] << " ";
            }
            //cout << endl;
        }
    }

    CvSVMParams SVMParam_CV;
    SVMParam_CV.svm_type = params.svmParams.type;
    SVMParam_CV.kernel_type = params.svmParams.kernel;
    SVMParam_CV.degree = params.svmParams.degree;
    SVMParam_CV.gamma = params.svmParams.gamma;
    SVMParam_CV.coef0 = params.svmParams.coef0;
    SVMParam_CV.C = params.svmParams.C;
    SVMParam_CV.nu = params.svmParams.nu;
    SVMParam_CV.p = params.svmParams.p;
    SVMParam_CV.class_weights = 0;
    SVMParam_CV.term_crit = cvTermCriteria(params.svmParams.termType,
                                           params.svmParams.iterations,
                                           params.svmParams.eps);

    if(!SVMModel_CV.train_auto(trainData,
                                dataLabel,
                                cv::Mat(),
                                cv::Mat(),
                                SVMParam_CV,
                                params.svmParams.kFold))
        cout << "Training failed..." << endl;
    else
        cout << "Training successful..." << endl;

}

bool BagOfFeatures::trainSVM()
{
    if(SVMModel != NULL)
    {
        svm_destroy_model(SVMModel);
        //svm_destroy_param(&SVMParam);
    }

    int i, j, k, l = -1;
    int totalData = 0;
    int size, length = codex.size;
    int count;
    //Get the total number of training data
    for(i = 0; i < params.numClasses; i++)
        totalData += data[i].getTrainSize();

    // Set up the data
    struct svm_problem SVMProblem;
    SVMProblem.l = totalData;
    SVMProblem.y = new double [totalData];
    SVMProblem.x = new struct svm_node* [totalData];

    // For each class
    for(i = 0; i < params.numClasses; i++)
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
                //cout << trainObject[i].histogramSet[j].histogram[k] << " ";
                if(trainObject[i].histogramSet[j].histogram[k] != 0)
                {
                    SVMProblem.x[l][count].index = k+1;
                    SVMProblem.x[l][count].value = trainObject[i].histogramSet[j].histogram[k];
                    count++;
                }
            }
            //cout << endl;
            SVMProblem.x[l][count].index = -1;
            SVMProblem.y[l] = data[i].getLabel();
        }
    }

    // Types
    SVMParam.svm_type = params.svmParams.type;
    SVMParam.kernel_type = params.svmParams.kernel;
    // Parameters
    SVMParam.degree = params.svmParams.degree;
    SVMParam.gamma = params.svmParams.gamma;
    SVMParam.coef0 = params.svmParams.coef0;
    SVMParam.C = params.svmParams.C;
    // For training only
    SVMParam.cache_size = params.svmParams.cache;
    SVMParam.eps = params.svmParams.eps;
    SVMParam.nu = params.svmParams.nu;
    SVMParam.shrinking = params.svmParams.shrinking;
    SVMParam.probability = params.svmParams.probability;
    // Don't change the weights
    SVMParam.nr_weight = params.svmParams.weight;


    double* target = new double [totalData];
    svm_cross_validation(&SVMProblem, &SVMParam, 10, target);
    delete [] target;

    if(svm_check_parameter(&SVMProblem, &SVMParam) != NULL)
    {
        //svm_cross_validation(&SVMProblem, &SVMParam, 10, target);
        SVMModel = svm_train(&SVMProblem, &SVMParam);
        return true;
    }
    else
    {
        cout << "SVM Parameters are not feasible!" << endl;
        return false;
    }
    //svm_save_model("svmSURF800",SVMModel);
    //classifierType = LIBSVM_CLASSIFIER;
}


void BagOfFeatures::buildBoF()
{
    int i, j;
    int train, valid, test, label;

    params.numFeatures = 0;

    //First extracting the features
    for(i = 0; i < params.numClasses; ++i)
    {
        data[i].getDataInfo(train, valid, test, label);
        for(j = 0; j < train; ++j)
        {
            trainObject[i].featureSet[j].extractSIFT_CV(data[i].getDataList(j),
                    params.siftParams.detectionThreshold,
                    params.siftParams.edgeThreshold,
                    true);
            //extractFeatures(trainObject[i].featureSet[j],
             //               data[i].getDataList(j));
            params.numFeatures += trainObject[i].featureSet[j].size;

        }
        for(j = 0; j < valid; ++j)
        {
            validObject[i].featureSet[j].extractSIFT_CV(data[i].getDataList(train+j),
                    params.siftParams.detectionThreshold,
                    params.siftParams.edgeThreshold,
                    true);
            //extractFeatures(validObject[i].featureSet[j],
             //               data[i].getDataList(train+j));
        }
        for(j = 0; j < test; ++j)
        {
            testObject[i].featureSet[j].extractSIFT_CV(data[i].getDataList(train+valid+j),
                    params.siftParams.detectionThreshold,
                    params.siftParams.edgeThreshold,
                    true);
            //extractFeatures(testObject[i].featureSet[j],
            //                data[i].getDataList(train+valid+j));
        }
    }

    cout << "Total number of training features: " << params.numFeatures << endl;

    //delete [] data;

    codex.alloc(params.clustParams.numClusters, params.featureLength);
    //Next building the dictionary
    codex.buildKClustering(trainObject,
                            params.numClasses,
                            params.numFeatures,
                            params.featureLength,
                            params.clustParams.numClusters,
                            params.clustParams.numPass,
                            params.clustParams.method,
                            params.clustParams.distance);

    codex.calcCentroid();

    cout << "Building the histograms..." << endl;

    for(i = 0; i < params.numClasses; ++i)
    {
        int label = data[i].getLabel();
        //cout << "Class " << i << "\n\t";
        trainObject[i].buildBoFs(codex, label);
        validObject[i].buildBoFs(codex, label);
        testObject[i].buildBoFs(codex, label);
    }
}

void BagOfFeatures::train()
{
    trainSVM_CV();
    //if(!trainSVM())
    //    cout << "Training Failed Because LibSVM is a horrible library..." << endl;
}

void BagOfFeatures::test()
{
    double results;
    for(int i = 0; i < params.numClasses; ++i)
    {
        int label = data[i].getLabel();
        results = trainObject[i].predict(SVMModel_CV, label);
        cout << "Training dataset accuracy for class " << label
            << ": " << results << endl;
        results = validObject[i].predict(SVMModel_CV, label);
        cout << "Validation dataset accuracy for class " << label
            << ": " << results << endl;
        results = testObject[i].predict(SVMModel_CV, label);
        cout << "Test dataset accuracy for class " << label
            << ": " << results << endl;
    }
}

void BagOfFeatures::processDataSet(DataSet set, int obj)
{
    int train, valid, test, label;
    int i;
    cv::Mat img;
    set.getDataInfo(train, valid, test, label);
    for(i = 0; i < train; ++i)
    {
        img = cv::imread(set.getDataList(i), 0);
        extractFeatures(trainObject[obj].featureSet[i], img);
        params.numFeatures += trainObject[i].featureSet[j].size;
    }
    for(i = 0; i < valid; ++i)
    {
        img = cv::imread(set.getDataList(i+train), 0);
        extractFeatures(validObject[obj].featureSet[i], img);
    }
    for(i = 0; i < test; ++i)
    {
        img = cv::imread(set.getDataList(i+train+valid), 0);
        extractFeatures(testObject[obj].featureSet[i], i);
    }
}


