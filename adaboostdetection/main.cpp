//
//  main.cpp
//  latentSVMdetection
//
//  Created by Matias on 4/7/15.
//  Copyright (c) 2015 Matias. All rights reserved.
//

#include <iostream>
#include <string>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"


#define RESIZE_FACTOR		1.0

#if defined(WIN32) || defined(_WIN32)
#include <io.h>
#else
#include <dirent.h>
#endif

#ifdef HAVE_CVCONFIG_H
#include <cvconfig.h>
#endif

#ifdef HAVE_TBB
#include "tbb/task_scheduler_init.h"
#endif

#define HAAR    1
//#define HOG 1

using namespace std;
using namespace cv;

static void help()
{
    cout << "This program demonstrated the use of the adaboost detector." << endl <<
    "It reads in a trained object models and then uses them to detect the objects in an images." << endl <<
    endl;
}

#ifdef HAAR
static void detectAndDrawObjects( Mat& image, CascadeClassifier& detector)
#else
static void detectAndDrawObjects( Mat& image, HOGDescriptor& detector)
#endif
{
    vector<Rect> detections;
    
    //Transform to grayscale
    //Mat frame_gray;
    //cvtColor( image, frame_gray, CV_BGR2GRAY );
    //equalizeHist( frame_gray, frame_gray );
    
    TickMeter tm;
    tm.start();
#ifdef HAAR
        detector.detectMultiScale( image, detections,1.5,1, CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT,Size(30, 30));
#else
        detector.detectMultiScale(image, detections, 0, Size(8,8), Size(32,32), 1.05, 2);
#endif
    tm.stop();
    
    cout << "Detected rois " << detections.size() << " Detection time = " << tm.getTimeSec() << " sec" << endl;
    string score;
    
    for( size_t i = 0; i < detections.size(); i++ )
    {
            Rect od=detections[i];

            rectangle( image, od, 10 , 1 );
            char score[256];
            sprintf(score,"%2.4f",(float)1);

    }
}

static void readDirectory( const string& directoryName, vector<string>& filenames, bool addDirectoryName=true )
{
    filenames.clear();
    
#if defined(WIN32) | defined(_WIN32)
    struct _finddata_t s_file;
    string str = directoryName + "\\*.*";
    
    intptr_t h_file = _findfirst( str.c_str(), &s_file );
    if( h_file != static_cast<intptr_t>(-1.0) )
    {
        do
        {
            if( addDirectoryName )
                filenames.push_back(directoryName + "\\" + s_file.name);
            else
                filenames.push_back((string)s_file.name);
        }
        while( _findnext( h_file, &s_file ) == 0 );
    }
    _findclose( h_file );
#else
    DIR* dir = opendir( directoryName.c_str() );
    if( dir != NULL )
    {
        struct dirent* dent;
        while( (dent = readdir(dir)) != NULL )
        {
            if( addDirectoryName )
                filenames.push_back( directoryName + "/" + string(dent->d_name) );
            else
                filenames.push_back( string(dent->d_name) );
        }
        
        closedir( dir );
    }
#endif
    
    sort( filenames.begin(), filenames.end() );
}

int main(int argc, char* argv[])
{
    help();
    
    string images_folder, day_folder, seq_folder;
    float overlapThreshold = 0.2f;
    int c, fr = 0;
    FILE * fid_fppi;
    char rois_filename[256];
    

    images_folder = "../../Test/unknown";
    vector<string> images_filenames;
    readDirectory( images_folder, images_filenames );

#ifdef HAAR
    CascadeClassifier detector;
    detector.load("../../Models/haarcascade_upperbody.xml");
    //detector.load("../../Models/visionary-car-truck_HAAR.xml");

    if( detector.empty() )
    {
        cout << "Models cann't be loaded" << endl;
        exit(-1);
    }
#else
    HOGDescriptor detector("../../Models/hogcascade_pedestrians.xml");
    //detector.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
#endif
    
    
    cout << "overlapThreshold = " << overlapThreshold << endl;
    
    sprintf(rois_filename,"latentDetection_GS_10_09_df_%s.dat",seq_folder.c_str());
    fid_fppi = fopen(rois_filename,"w+");
    for( size_t i = 0; i < images_filenames.size(); i++ )
    {
        Mat rzimg;
        Size dsize = Size(0,0);
        Mat image = imread( images_filenames[i] );
        if( image.empty() )  continue;
        fr++;
        cout << "Process image " << images_filenames[i] << endl;
        resize( image,rzimg,dsize,RESIZE_FACTOR,RESIZE_FACTOR);
        detectAndDrawObjects( rzimg, detector);
        
        imshow( "result", rzimg );
        
        c = waitKey();
        if( (char)c == '\x1b' )
            break;
    }
    fclose(fid_fppi);
    return 0;
}
