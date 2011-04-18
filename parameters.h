#define CLUSTERING_K_MEANS 0

#define FEATURES_SIFT 0
#define FEATURES_SURF 1

#define CLASSIFIER_SVM 0
#define CLASSIFIER_NAIVE_BAYES 1



struct ClusteringParameters
{
    int numClusters;
    int numPass;
    char method;
    char distance;
};

struct SIFTParameters
{
    double detectionThreshold;
    double edgeThreshold;
};

struct SURFParameters
{
    double hessianThreshold;
    int nOctives;
    int nLayers;
    bool extended;
};

struct SVMParameters
{
    int type;
    int kernel;
    double degree;
    double gamma;
    double coef0;
    double C;
    double cache;
    double eps;
    double nu;
    double p;
    int termType;
    int iterations;
    int shrinking;
    int probability;
    int weight;
    int kFold;
};

class preProcessBase
{

}

class BoFParameters
{
    public:
        //Clustering parameters
        ClusteringParameters clustParams;

        //Feature Parameters
        SIFTParameters siftParams;
        SURFParameters surfParams;

        //Classifier parameters
        SVMParameters svmParams;

        int numClasses;
        int numImages;
        int numFeatures;
        int featureLength;

        int classifierType;
        int clusterType;
        int featureType;

        bool verbose;
};
