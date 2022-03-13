//This Program recognizes characters
//Will need to teach it 
//Maahir Mawla 


#include<opencv2/core/core.hpp>

#include<opencv2/highgui/highgui.hpp>;

#include<opencv2/imgproc/imgproc.hpp>

#include<opencv2/ml/ml.hpp>



#include<iostream>

#include<vector>



//global variables

const int MIN_CONTOUR_AREA = 100;



const int RESIZED_IMAGE_WIDTH = 20;

const int RESIZED_IMAGE_HEIGHT = 30;



using namespace std;

using namespace cv;



int main() 
{

    //inputs the image 

    Mat imgTrainingCharacters;

    //turns the jpeg or png image into a grayscale image

    //ie turns a coloured image into gray (or better known to lose colour)

    Mat imgGrayscale;

    //declare the different images that will be shown to the program

    Mat imgBlurred;

    //thresholding perameters

    
    Mat imgThresh;

    Mat imgThreshCopy;

      

    //declare the contours

    vector<vector<Point> > ptContours;

    vector<Vec4i> v4iHierarchy;

    //this is what will be used for training classifications

    //due to the data that the KNN object Knearest requires, a single Mat has to be declared

    Mat matClassificationInts;



    Mat matTrainingImagesAsFlattenedFloats;



    // vector contains the chars that I want the program to be able to recognize is

    //all the numbers between 0-9

        //this is because these are the only numbers that make up every other number

    //all the letters of the alphabet between A-Z

    vector<int> character = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',

        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',

        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',

        'U', 'V', 'W', 'X', 'Y', 'Z' };



    //this function is what reads the numbers that wil be given to the program in order to read it and identify what it is

    imgTrainingCharacters = imread("betterchar.jpg");



    //display an error message if no image of the characters is able to be open

    if (imgTrainingCharacters.empty())

    {

        cout << "Error!! Image can not be read from the file" << endl;

        return(0);

    }


    //converts the image given into grayscale

    cvtColor(imgTrainingCharacters, imgGrayscale, COLOR_BGR2GRAY);

    //input the image

    GaussianBlur(imgGrayscale,

        //output image

        imgBlurred,

        //turn the window width and the height to the specific amount of pixels for every single character

        Size(7.5, 7.5),
        //sigma value, determines how much the image will be blurred, zero makes function choose the sigma value

        //to be honest this part is a little foggy learn more from online

        0);

     
    // filter image from grayscale to black and white

    adaptiveThreshold(imgBlurred,

        // output image

        imgThresh,

        // make pixels that pass the threshold white

        255,

        //gaussian was used rather than mean because results were far better

        ADAPTIVE_THRESH_GAUSSIAN_C,

        //this is what makes it so that everything in the background will be black and the foreground will be white

        THRESH_BINARY_INV,

        // size of a pixel neighborhood used to calculate threshold value (change to what you want or works)

        11,

        2);

    //show the threshold image in order to see if the threshold works and if the opencv main aspect works

    imshow("imgThresh", imgThresh);

    // make a copy of the thresh image, this in necessary because findContours (the next function) modifies the image

    imgThreshCopy = imgThresh.clone();

    //input image, this function will modify the image

    //imgthreshcopy will make a copy of the threshold we made

    findContours(imgThreshCopy,

        //output the contours

        ptContours,

        v4iHierarchy,

        //this retrieves the external contourss (outermost)

        RETR_EXTERNAL,

        //compress horizontal, vertical, and diagonal segments and leave only their end points

        CHAIN_APPROX_SIMPLE);



    for (int i = 0; i < ptContours.size(); i++)

    {

        // if contour is big enough to consider get the bounding rect

        if (contourArea(ptContours[i]) > MIN_CONTOUR_AREA)

        {
            Rect boundingRect = cv::boundingRect(ptContours[i]);

            //while asking for the specific characters, put a red rectangle around the charcaters

            rectangle(imgTrainingCharacters, boundingRect, Scalar(0, 0, 255), 2);

            // get ROI image of bounding rect

            //ROI is an image that you want to filter or some other operation

            Mat matROI = imgThresh(boundingRect);

            Mat matROIResized;

            //resize image because this will be more consistent for recognition and storage for all the characters being read

            resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));

            //show ROI image to reference

            imshow("matROI", matROI);
            cout << "matROI" << endl;
            //show resized ROI image for reference

            imshow("matROIResized", matROIResized);
            cout << "resized" << endl;
            //show training numbers image, this will now have red rectangles drawn on it

            imshow("imgTrainingCharacters", imgTrainingCharacters);
            cout << "training" << endl;


            //will be used to exit the program using a key presss

            int exitp = waitKey(0);



            //exit if esc (escape button) is pressed

            if (exitp == 27) 
            {

                return(0);

            }

            //if the char is in the list of chars we are looking for add the classifcations to the integer list of chars

            else if (find(character.begin(), character.end(), exitp) != character.end()) {

                //where the integer is added to the list of chars

                matClassificationInts.push_back(exitp);

                //adds the trainning image

                Mat matImageFloat;

                //this converts Mat into float

                matROIResized.convertTo(matImageFloat, CV_32FC1);



                Mat matImageFlattenedFloat = matImageFloat.reshape(1, 1);



                matTrainingImagesAsFlattenedFloats.push_back(matImageFlattenedFloat);

            }

        }

    }

    //output this so I know that the training in the program is done

    cout << "training complete" << endl;


    
    //This is what saves the classifications

    //open the classifications file

    FileStorage fsClassifications("classifications.xml", FileStorage::WRITE);



    if (fsClassifications.isOpened() == false)

    {

        //test to know whether the file can be opened

        cout << "Unable to open training classifications file" << endl;

        return(0);

    }



    fsClassifications << "classifications" << matClassificationInts;

    fsClassifications.release();



    //save training images to file

    //open the training images file

    FileStorage fsTrainingImages("images.xml", FileStorage::WRITE);



    // if the file was not opened successfully

    if (fsTrainingImages.isOpened() == false)

    {

        cout << "error, unable to open training images file, exiting program" << endl;

        return(0);

    }



    //write training images into images section of images file

    fsTrainingImages << "images" << matTrainingImagesAsFlattenedFloats;

    fsTrainingImages.release();



    return(0);
    
    
}

