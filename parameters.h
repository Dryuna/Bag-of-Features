
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
    int shrinking;
    int probability;
    int weight;
};

class BoFParameters
{
    public:
        ClusteringParameters clustParams;
        SIFTParameters siftParams;
        SVMParameters svmParams;

        int numClasses;
        int numImages;
        int numFeatures;
        int featureLength;

        int classifierType;
        int clusterType;
        int featureType;
}
