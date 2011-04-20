/*!
*\file datasetinfo.h
*\author David Leblanc
*\date 01/22/2011
*
*This file contains the declaration of the DataSet Class
*Which is used to keep track of the sizes of data sets used
*(Training, Validation, Testing), to load the sets of images,
*and to shuffle the data.
*/

/*!
 * \class DataSet
 *
 * \brief Handles the data set structures, locations, and sizes
 *
 * This class holds the sizes for the training, validation, and
 * testing sets. It also holds the \p int label assigned to the
 * DataSet. It also holds a list of image names in an array of
 * \p char which will be loaded by the main driving during the
 * training or testing phases. This class allows the user to load,
 * save, and shuffle the DataSet.
 *
 * \author David Leblanc
 * \date 01/22/2011
 */
class DataSet
{
    public:
        // Constructor
        /*!
        \brief Default Constructor

        This constructor sets all private members to 0, requiring
        the user to call the setDataSize() function to set the
        member. It does not allocate any memory.
        */
        DataSet();

        /*!
        \brief Constructor with known set sizes

        This constructor takes in the parameters which set the private
        members of the DataSet class. This constructor computes the
        \p total by summing up the train, valid, and test members. It
        also allocates the \p char list of image names, and sets the length
        of it. It also assigns the label to the DataSet, therefore there
        is no need to call the setDataSize() function.

        \param t the number of training images
        \param v the number of validation images
        \param s the number of test images
        \param l the label assigned to the DataSet
        \param len the max \p length for the image names.

        */
        // Set the number of train, validation, and test data
        DataSet(int t, int v, int s, int l, int len);

        DataSet(const DataSet& data);

        /*!
        \brief Constructor with known total size, and desired data splitting.

        This constructor takes in the total number of images, and
        divides it into its corresponding sets (training, validation, and test)
        based on a certain percentage, usually 60% for training,
        20% for validation, and 20% for test. It also assigns the label
        to the set, and allocates the array of \p char image names.

        \param t the total number of images for the DataSet.
        \param t_p the percentage of images used for training.
        \param v_p the percentage of images used for validation.
        \param s_p the percentage of images used for testing.
        \param l the label assigned to the DataSet.
        \param len the max \p length of the image names.
        */
        // Set the total number of images, and the % of each set
        DataSet(int t, double t_p, double v_p, double s_p, int l, int len);
        // Destructor

        /*!
        \brief Deallocates the array of image names.
        */
        ~DataSet();

        /*!
        \brief Deep copy assignment operator.

        This operator copies the right-hand side DataSet
        into a new DataSet. The operator does a deep copy of the
        image names. If the memory was previously allocated
        for the set, it is first deallocated, then reallocated
        and copied based on the \p rhs.

        \param rhs a DataSet to be copied.

        \return a deep copy of the \p rhs.
        */
        DataSet &operator=(const DataSet &rhs);

        /*!
        \brief Retrieves the number of training, validation,
        and test sets, and also the \p label.

        \param t value of \p train.
        \param v value of \p valid.
        \param s value of \p test.
        \param l value of \p label.
        */
        // Get the information
        void getDataInfo(int& t, int& v, int& s, int &l);

        /*!
        \brief Retrieves the number of training images.

        \return the value of \p train.
        */
        int getTrainSize();

        /*!
        \brief Retrieves the number of validation images.

        \return the value of \p valid.
        */
        int getValidSize();

        /*!
        \brief Retrieves the number of test images.

        \return the value of \p test.
        */
        int getTestSize();

        /*!
        \brief Retrieves the label for the DataSet.

        \return the value of \p label.
        */
        int getLabel();

        /*!
        \brief Retrieves the total number of images.

        \return the value of \p total.
        */
        int getTotal();

        /*!
        \brief Retrieves the image name at an index in the list

        \param index the index of the list of image names

        \return the pointer to the name of the image at \p index.
        */
        char* getDataList(int index);

        void setDataSize(int t, int v, int s, int l, int len);

        // Load the image names that are located at a specified point
        bool loadDataSetAt(const char* location, const char* type, int digit);
        // Load the image names from a file
        bool loadDataSetFromFile(const char* fileName);
        // Save the set into a file
        bool saveDataSet(const char* fileName);

        // Shuffles the data arround for validation
        void shuffleDataSet(int times);

    private:
        /*! Holds the number of training samples */
        int train;
        /*! Holds the number of samples used for validation */
        int valid;
        /*! Holds the number of samples used for testing */
        int test;
        /*! Holds the number of samples in the DataSet */
        int total;
        /*! Holds an assigned label for the DataSet */
        int label;
        /*! Contains the list of all image names for the DataSet.
            Indexes [ 0 -> \p train - \p 1 ] are the images used for
            training;
            Indexes [ \p train -> \p train + \p valid - \p 1 ]
            are used for validation;
            Indexes [ \p train + \p valid -> \p total - \p 1 ] are
            used for testing.*/
        char** dataList;
        /*! The maximum size of the \p char string image name. */
        int length;
};
