#include <opencv2/opencv.hpp>
#include<vector>

using namespace cv;

//Global declarations.
int low_h=0,low_s=0,low_v=65;
int high_h=255,high_s=100,high_v=115;
Scalar color = Scalar( 0,0,255 );
RNG rng(12345);

void imagePreprocessing(Mat& inputImage,Mat& outputImage);
void contourGeneration(Mat& inputImage,std::vector<std::vector<Point>>& outputContour,std::vector<Vec4i>& outputHierarchy);
void momentGeneration(std::vector<std::vector<Point>> inputContour,std::vector<int>& outputX,std::vector<int>& outputY);
void centerGeneration(std::vector<Point>& inputContour,int& outputX,int& outputY);
void finalImageOutput(std::vector<std::vector<Point>> inputContour,std::vector<Vec4i>& hierarchy,int& inputXCenter,int& inputYCenter,std::vector<int>& inputXMoment, std::vector<int>& inputYMoment,Mat& outputImage);
void arrowDetector(Mat& frame);

int main()
{
    VideoCapture capture(0);
    Mat frame;

    namedWindow("Initial",1);
    namedWindow("Mask",1);

    while(capture.isOpened())
    {
        int c = waitKey(5);
        capture>>frame;

        arrowDetector(frame);
        imshow("Initial",frame);
        if(c=='q')
            break;
    }
    destroyAllWindows();
    return 0;
}


/**
* Changes the color space of input from BGR(Blue Green Red) to HSV(Hue Saturation Value).
* Converting to HSV decreases sensitivity of the image vector to color change
*
* Sets a mask on the image within the specified HSV limits(Defined globally as {low_h,low_s,low_v}{high_h,high_s,high_v})
* Mask limits are determined based on color of the desired object. Note HSV is a cylindrical color space and does not follow a circular
* coloring scheme.
*
* @deprecated Blurs the input image for lower sensitivity with a (3,3) kernel. (Deprecated due to loss of sharpness)
* @param {inputImage} The image to be processed(in cv::Mat)
* @param {outputImage} The output of the function after processing(in cv::Mat)
* @return void
*/

void imagePreprocessing(Mat& inputImage,Mat& outputImage)
{
    cvtColor(inputImage,outputImage,COLOR_BGR2HSV);
    inRange(outputImage, Scalar(low_h, low_s, low_v), Scalar(high_h, high_s, high_v), outputImage);
    erode(outputImage,outputImage,getStructuringElement(MORPH_RECT,Size(3,3)),Point(-1,-1),1);
    dilate(outputImage,outputImage,getStructuringElement(MORPH_RECT,Size(3,3)),Point(-1,-1),1);
}


/**
* Generates contours from a preprocessed mask. Applies polygon approximation on the generated contours with a
* specified epsilon. Rejects all contours which have more or less than 7 vertices(arrow). Rejects all contours whose 
* area is below a pre-determined threshold.
*
* @param {inputImage} The image mask from which contours must be generated.(in cv::Mat)
* @param {outputContour} Final vector of contours generated after all processes.(in std::vector<std::vector<cv::Point>>)
* @param {outputHierarchy} Provides a definition of all parent and child contours.
* @return void
*/

void contourGeneration(Mat& inputImage,std::vector<std::vector<Point>>& outputContour,std::vector<Vec4i>& outputHierarchy)
{
    std::vector<std::vector<Point> > contours;
    findContours( inputImage, contours, outputHierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );

    for(int i=0;i<contours.size();i++)
    {
        std::vector<Point> temp;
        approxPolyDP( Mat(contours[i]), temp, 9, true ); //Epsilon = 9
        if(temp.size()==7)
        {
            if(contourArea(temp)>2500)
            {
                outputContour.push_back(temp);
            }
        }
    }
}


/**
* Generates the moments from a given contour, These moments are used to calculate the COM(Center of mass) of the obtained contour.
*
* @param {inputContour} The contour from which moments are to be generated(in std::vector<cv::Point>)
* @param {outputX} The X coordinate of the center of mass of the given contour
* @param {outputY} The Y coordinate of the center of mass of the given contour
* @return void
*
* Currently takes only singular contour. Needs to be iterative for better output.
*/

void momentGeneration(std::vector<std::vector<Point>> inputContour,std::vector<int>& outputX,std::vector<int>& outputY)
{
    for(int i=0;i<inputContour.size();i++)
    {
        Moments p = moments(inputContour[i]);
        outputX.push_back(int(p.m10/p.m00));
        outputY.push_back(int(p.m01/p.m00));
    }
}


/**
* Calculates the rectangle with minimum area around a contour. From the vertices of the rectangle the center is approximated.
*
* @param {inputContour} The contour from which center is to be generated(in std::vector<cv::Point>)
* @param {outputX} The X coordinate of the center of the given contour
* @param {outputY} The Y coordinate of the center of the given contour
* @return void
*/

void centerGeneration(std::vector<Point>& inputContour,int& outputX,int& outputY)
{
    RotatedRect box = minAreaRect(inputContour);
    Point2f vtx[4];
    box.points(vtx);
    outputX = (((vtx[0].x+vtx[2].x)/2)+((vtx[1].x+vtx[3].x)/2))/2;
    outputY = (((vtx[0].y+vtx[2].y)/2)+((vtx[1].x+vtx[3].x)/2))/2;
}


/**
* Loops through contours and sets text according to the direction of the arrow.
* Moment center compared to bounding rectangle center is used to obtain direction of the arrow.
* 
* @param {inputContour} The contours from which direction is to be generated(in std::vector<std::vector<cv::Point>>)
* @param {hierarchy} The definition of parent to child links inside the contour.
* @param {inputXCenter} X coordinate of the center of bounding box
* @param {inputYCenter} Y coordinate of the center of bounding box
* @param {inputXMoment} X coordinated of each moment center(in std::vector<int>)
* @param {inputYMoment} X coordinated of each moment center(in std::vector<int>)
* @param {outputImage} Final output to be generated.
* @return void
*/

void finalImageOutput(std::vector<std::vector<Point>> inputContour,std::vector<Vec4i>& hierarchy,int& inputXCenter,int& inputYCenter,std::vector<int>& inputXMoment, std::vector<int>& inputYMoment,Mat& outputImage)
{
    for(int i=0;i<inputContour.size();i++)
    {
        centerGeneration(inputContour[i],inputXCenter,inputYCenter);
        std::string s;
        std::cout<<"Moment: "<<inputXMoment[i]<<" "<<inputYMoment[i]<<std::endl;
        std::cout<<"Center: "<<inputXCenter<<" "<<inputYCenter<<std::endl;
        int tangent;
        int percentage;
        if(inputXMoment[i]>inputXCenter)
            s = "Left";
        else if(inputXMoment[i]<inputXCenter)
            s = "Right";
        if(inputXMoment[i]!=inputXCenter)
        {
            tangent = (inputYMoment[i]-inputYCenter)/(inputXMoment[i]-inputXCenter);
            percentage = abs(100/(tangent+1));
        }
        else{
            percentage = 100;
        }
        std::string t = std::to_string(percentage);
        std::cout<<t<<std::endl;
        putText(outputImage,t,Point(inputXCenter,inputYCenter),0,1,Scalar(0,0,255),2);
        putText(outputImage, s, Point(inputXCenter,inputYCenter),0, 1,Scalar(0,0,255),2);
        drawContours( outputImage, inputContour, (int)i, color, 3, LINE_8, hierarchy, 0 );
    }
}

/**
* Wrapper function to complete arrow_detection operation.
* 
* @param {frame} Image input and also image to be written on.
* @return void
*/

void arrowDetector(Mat& frame)
{
    int xm,ym;
    Mat processed_frame;
    std::vector<std::vector<Point>> contoursOUT;
    std::vector<Vec4i> hierarchy;

    imagePreprocessing(frame,processed_frame);
    contourGeneration(processed_frame,contoursOUT,hierarchy);
        
    std::vector<int> x_temp;
    std::vector<int> y_temp;

    if(contoursOUT.size()>0)
        momentGeneration(contoursOUT,x_temp,y_temp);
    finalImageOutput(contoursOUT,hierarchy,xm,ym,x_temp,y_temp,frame);
    imshow("Mask",processed_frame); //Debug only
}
