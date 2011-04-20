
#include <ml.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>


#include "datasetinfo.h"
#include "imagefeatures.h"
#include "parameters.h"

#include "Surf/surflib.h"
#include "libSVM/svm.h"

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
        BagOfFeatures(BoFParameters p, DataSet* val);
        ~BagOfFeatures();

        // Allocates the Bag of Features
        void alloc(BoFParameters p, DataSet* val);

        void extractFeatures(ImageFeatures &f, cv::Mat img);

        void buildBoF();
        void train();
        void testDataSet();

    private:

        void processDataSet(DataSet set, int obj);
        //Data
        ObjectSet *testObject;
        ObjectSet *validObject;
        ObjectSet *trainObject;
        DataSet *data;

        BoFParameters params;

        //For Hierarchical Clustering
        double** hClusterData;
        Node* hTree;

        //Visual Dictionary
        //CvMat* dictionary;
        Dictionary codex;

        //Classifiers
        struct svm_parameter SVMParam;
        struct svm_model *SVMModel;
        bool trainSVM();

        // The OpenCV algorithms, don't work well right now
        CvSVM SVMModel_CV;
        void trainSVM_CV();
        //CvNormalBayesClassifier NBModel_CV;
};
