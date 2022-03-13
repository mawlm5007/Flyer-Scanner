//Final Project Main
//Maahir Mawla 
//
//this program will read characters using a KNN algorithm and clasifcations and image 
//files that was created from a seperate program


#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<sstream>

using namespace std;

//global variables
const int MIN_CONTOUR_AREA = 100;

const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

//
class ContourWithData 
{
    public:
    //declare all variables
    std::vector<cv::Point> ptContour;   
    //create a rectangle around possible character 
    //this rect will have certain x y points around the character found 
    cv::Rect rectangle;                     
    float area;                              

    //boolean to check to see if the contour is valid                                            
    bool checkIfContourIsValid() 
    {                              
        if (area < MIN_CONTOUR_AREA)
        {
            return false;
        }
        
        return true;                                            
    }

    //this function sorts the contours from when they were found from left to right 
    //this is because in english we read from left to right 
    static bool sortLeftRight(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) 
    {      
        return(cwdLeft.rectangle.x < cwdRight.rectangle.x);                                                   
    }

};


int main() 
{
   //declare variables 
    vector<ContourWithData> allContoursWithData;          
    vector<ContourWithData> validContoursWithData;         
                                                              
    //this mat variable will read in the classifications from the classifcation file retrieved from the previous program
    cv::Mat matClassificationInts;      

    //open the classification files 
    //to read we need to use opencv library ability, READ
    cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::READ);        

    //if the file was not open successfully output an error message
    //isOpened will return true if the file was opened
    //that is why if it is made equal to false then that means it was not able to be opened
    if (fsClassifications.isOpened() == false) 
    {                                                    
        cout << "error, unable to open training classifications file, exiting program"<<endl;    
        return(0);                                                                                  
    }

    //now read the classifications file into mat classifications variable 
    fsClassifications["classifications"] >> matClassificationInts;    
    //exit and close the classification file after use 
    fsClassifications.release();                                       
                                                                        
    //read in the image 
    cv::Mat matTrainingImagesAsFlattenedFloats;         

    //open the images file that will be read 
    cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ);          

    //again used .isOpened to see whether the file was opened or not
    //if not opened the proper message should be displayed 
    if (fsTrainingImages.isOpened() == false) 
    {                                                 
        cout << "error, unable to open training images file, exiting program" << endl;
        return(0);                                                                              
    }

    //now have the image file data be read into the variable and stored 
    fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;           
    //close the image file
    fsTrainingImages.release();                                                 

                                                                               
    //use the Knearest function which will use KNN algorithm 
    //KNN stand for K nearest neighbors
    //simple classification algorithm which was the easiest to find online which is why it is used 
    //there is no actual training phase, just uses the data from the other program to figure out the character - it also keeps all
    //the training data 
    //this is why ALL the data is needed for this 
    //an object is classified by a majority of votes compared to other characters (neighbors) and assigns character to closest class
    cv::Ptr<cv::ml::KNearest>  kNearest(cv::ml::KNearest::create());            
                                                                            
    kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);

    //read in the image actually being used
    cv::Mat matNumbers = cv::imread("nofrills.jpg");            

    //show error message if the image is not able to be opened
    if (matNumbers.empty()) 
    {                                
        cout << "error: image not read from file"<<endl;         
        return(0);                                                  
    }
    
    //declare all variables
    cv::Mat matGrayscale;           
    cv::Mat matBlurred;             
    cv::Mat matThresh;              
    cv::Mat matThreshCopy;          

    //actual opencv part of program (image processing part)
    //matGrayscale will convert the image to grayscale (grayscale is when there is no colour there and the image is shades of gray)
    cv::cvtColor(matNumbers, matGrayscale, cv::COLOR_BGR2GRAY);         

    //blur the image      
    //adjust the window width and height in pixels
    //the zero is a sigma value which determines how much image will be blurred, zero makes the program choose because 
    //the actual logic behind it made no sense to me
    cv::GaussianBlur(matGrayscale, matBlurred, cv::Size(5, 5), 0);                        

    //filter image from grayscale to black and white
    //255 is what makes the pixels that pass the threshold full white (can change to any colour)
    //uses gaussian (math makes no sense) 
    //set size of pixel used to calculate threshold value (11)
    cv::adaptiveThreshold(matBlurred, matThresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);                                  

    //make a copy of threshold because ptContours will modifty it 
    matThreshCopy = matThresh.clone();              

    //declare all variables
    vector<vector<cv::Point> > ptContours;        
    vector<cv::Vec4i> v4iHierarchy;                    

    //input image COPYYY
    //find the contours and compress horizontal, vertical, and diagonal end points 
    cv::findContours(matThreshCopy, ptContours, v4iHierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);               

    for (int i = 0; i < ptContours.size(); i++) 
    {               
        ContourWithData contourWithData;   
        //assign contour to contour found with previous data
        contourWithData.ptContour = ptContours[i];    
        //get the bounding rect (this is the rectangle)
        contourWithData.rectangle = cv::boundingRect(contourWithData.ptContour);
        //calculate the area of the contour
        contourWithData.area = cv::contourArea(contourWithData.ptContour); 
        //add contour with data object to list of all contours with data
        allContoursWithData.push_back(contourWithData);                                     
    }

    //for all contours, check if it is valid, and then add to valid contour list
    for (int i = 0; i < allContoursWithData.size(); i++) 
    {                      
        if (allContoursWithData[i].checkIfContourIsValid())
        {                   
            validContoursWithData.push_back(allContoursWithData[i]);           
        }
    }

    // sort contours from left to right
    std::sort(validContoursWithData.begin(), validContoursWithData.end(), ContourWithData::sortLeftRight);

    //variable that will contain the final string
    std::string strFinalString;         

    for (int i = 0; i < validContoursWithData.size(); i++) 
    {            

        //draw a green rectangle around the characters seperately                                                                
        cv::rectangle(matNumbers,                            
            validContoursWithData[i].rectangle, cv::Scalar(0, 255, 0), 2);                                           

        cv::Mat matROI = matThresh(validContoursWithData[i].rectangle);          

        cv::Mat matROIResized;
        // resize image, this will be more consistent for recognition and storage
        cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     

        cv::Mat matROIFloat;
        //convert Mat to float, necessary for call to find_nearest
        matROIResized.convertTo(matROIFloat, CV_32FC1);             

        cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);

        cv::Mat matCurrentChar(0, 0, CV_32F);

        //now find the nearest possible character based off the classifications
        kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);     

        float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);

        //add the character that the program thinks is and add to the a list
        strFinalString = strFinalString + char(int(fltCurrentChar));       
    }

    //output the string
    std::cout << endl << "numbers read = " << strFinalString << endl;       

    //show the image with the green boxes around it
    cv::imshow("matNumbers", matNumbers);     

    //show the threshold image 
    //cv::imshow("matThreshold", matThresh)

    //exit program with any key 
    cv::waitKey(0);                                         

    return(0);
}


