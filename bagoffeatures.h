
#include <ml.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include "libSVM/svm.h"
#include "datasetinfo.h"
#include "imagefeatures.h"
#include "Surf/surflib.h"

extern "C"
{
    #include "libCluster/cluster.h"
    #include "Sift/include/sift.h"
}

#define LIBSVM_CLASSIFIER 1
#define CVSVM_CLASSIFIER 2
#define CVNORM_BAYES_CLASSIFIER 3


IplImage* preProcessImages(const IplImage* input, int minSize, int maxSize);

class BagOfFeatures
{
    public:
        BagOfFeatures();
        BagOfFeatures(const int n, DataSet* val);
        ~BagOfFeatures();

        // Allocates the Bag of Features
        void allocBoF(const int n, DataSet* val);

        int getNumFeatures()
        {
            return numFeatures;
        };
/*
        // Feature Extraction
        // Using SIFT Features
        bool extractSIFTFeatures(int lvls,
                                double sigma,
                                double thresh1,
                                int thresh2,
                                int dbl,
                                int width,
                                int bins,
                                int sizeMin,
                                int sizeMax);
        // Using SURF Features
        bool extractSURFFeatures(bool invariant,
                                int octaves,
                                int intervals,
                                int step,
                                float thresh,
                                int sizeMin,
                                int sizeMax);


        // Clustering Methods
        //Hierarchical Clustering
        bool buildHierarchicalTree(int transpose,
                                char dist,
                                char method,
                                double** distmatrix);
        bool cutHierarchicalTree(int numClusters);

        //K-Means
        bool buildKMeans(int numClusters,
                         CvTermCriteria criteria,
                         int repeat);

        // C-Clustering lib kCluster function
        bool buildKClustering(int numClusters,
                            int pass,
                            char method,
                            char dist);

        // Building the Histograms
        bool buildBofHistograms(bool normalize);
*/
        void filterDictionary(double R);

        // Training the BoF
        void trainSVM(int type, int kernel, double degree, double gamma, double coef0,
                        double C, double cache, double eps, double nu,
                        int shrinking, int probability, int weight);

        //Training using the opencv function
        CvSVM* trainSVM_CV(int type, int kernel, double degree, double gamma,
                         double coef0, double C, double nu, double p, int termType,
                         int iterations, double eps, char* fileName);

        CvNormalBayesClassifier* trainNormBayes_CV();

        // Computing the results
        float* resultsTraining();
        float* resultsTraining(CvNormalBayesClassifier *NBModel_CV);
        float* resultsTraining(CvSVM *SVMModel_CV);
        float* resultsValidation();
        float* resultsValidation(CvNormalBayesClassifier *NBModel_CV);
        float* resultsTest();
        float* resultsTest(CvNormalBayesClassifier *NBModel_CV);

        float predictClassification(ImageFeatures input, bool normalize);

        void process();


    private:
        //Data
        ObjectSet *testObject;
        ObjectSet *validObject;
        ObjectSet *trainObject;
        DataSet *data;
        int numClasses;
        int numFeatures;
        int descrSize;

        //For Hierarchical Clustering
        double** hClusterData;
        Node* hTree;

        //Visual Dictionary
        //CvMat* dictionary;
        Dictionary codex;

        //Classifiers
        int classifierType;
        struct svm_parameter SVMParam;
        struct svm_model *SVMModel;

        // The OpenCV algorithms, don't work well right now
        //CvSVM SVMModel_CV;
        //CvNormalBayesClassifier NBModel_CV;

        char classifierFile[64];
};
