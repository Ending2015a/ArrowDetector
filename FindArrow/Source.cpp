#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include <cmath>

#define __ARROW_DEBUG_MODE___

#include "Color.h"
#include "Detector.h"
#include "ArrowDetector.h"


#define __ValidCorner 150
#define __ValidParallel 30





double getNorm(cv::Point v){
	return sqrt(pow((double)v.x, 2) + pow((double)v.y, 2));
}

double getNorm2(cv::Point v){
	return pow((double)v.x, 2) + pow((double)v.y, 2);
}

double getAngle(cv::Point s1, cv::Point e1, cv::Point s2, cv::Point e2, bool direct = false){
	cv::Point v1 = cv::Point(e1.x - s1.x, e1.y - s1.y);
	cv::Point v2 = cv::Point(e2.x - s2.x, e2.y - s2.y);

	double nv1 = getNorm(v1);
	double nv2 = getNorm(v2);

	double angle = acos(((double)(v1.x*v2.x + v1.y*v2.y) / (nv1*nv2))) * 180 / CV_PI;

	if (direct == false)return angle;

	if (v1.x * v2.y - v2.x * v1.y < 0)return angle;
	else return -angle;
}

double getAngle(cv::Point v1, cv::Point v2){

	double nv1 = getNorm(v1);
	double nv2 = getNorm(v2);

	double angle = acos(((double)(v1.x*v2.x + v1.y*v2.y) / (nv1*nv2))) * 180 / CV_PI;

	if (v1.x * v2.y - v2.x * v1.y < 0)return angle;
	else return -angle+360.0;
}

double getAngle(cv::Point s, cv::Point m, cv::Point e){


	cv::Point v1 = cv::Point(s.x - m.x, s.y - m.y);
	cv::Point v2 = cv::Point(e.x - m.x, e.y - m.y);

	double nv1 = getNorm(v1);
	double nv2 = getNorm(v2);

	double angle = acos(((double)(v1.x*v2.x + v1.y*v2.y) / (nv1*nv2))) * 180 / CV_PI;
	
	if (v1.x * v2.y - v2.x * v1.y < 0)return angle;
	else return -angle;
}

cv::Point getVert(cv::Point s, cv::Point e, cv::Point t){
	cv::Point v1 = cv::Point(e.x - s.x, e.y - s.y);
	cv::Point v2 = cv::Point(t.x - s.x, t.y - s.y);

	double a = (v2.x*v1.x + v2.y*v1.y) / getNorm2(v1);
	cv::Point proj((int)(v1.x*a), (int)(v1.y*a));

	cv::Point L(v2.x - proj.x, v2.y - proj.y);
	return L;
}

bool validpoints(cv::Mat &debug, std::vector<double> &angles, std::vector<cv::Point> &cont){
	std::vector<double> va;
	std::vector<cv::Point> vc;

	//check valid point
	for (int i = 0; i < angles.size(); i++){
		if (fabs(angles[i]) < __ValidCorner){
			va.push_back(angles[i]);
			vc.push_back(cont[i]);
		}
	}

	if (va.size() < 6)return false;

	angles = va;
	cont = vc;

	if (va.size() != angles.size())std::cout << "ERROR" << std::endl;

	int p[2] = {};
	int c = 0;

	for (int i = 0; i < angles.size(); i++){
		if (angles[i] > 0 && angles[i] < __ValidCorner){
			if (c == 2)return false;
			p[c++] = i;
		}
	}

	if (c < 2)return false;

	cv::circle(debug, cont[p[0]], 2, cv::Scalar(255, 255, 0), 2);
	cv::circle(debug, cont[p[1]], 2, cv::Scalar(255, 255, 0), 2);


	size_t e1 = (p[0] - 1 < 0 ? p[0] - 1 + cont.size() : p[0] - 1);
	size_t e2 = (p[1] + 1 >= cont.size() ? p[1] + 1 - cont.size() : p[1] + 1);

	double vvag = getAngle(cont[p[0]], cont[e1], cont[p[1]], cont[e2]);

	
	std::stringstream ss;
	ss << vvag;
	cv::putText(debug, ss.str(), cont[e1], cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 0, 255), 1);
	cv::circle(debug, cont[e1], 2, cv::Scalar(255, 0, 255), 2);
	cv::circle(debug, cont[e2], 2, cv::Scalar(255, 0, 255), 2);
	/*
	ss = std::stringstream();

	ss << "(" << cont[p[0]].x << "," << cont[p[0]].y << ") -> (" << cont[e1].x << "," << cont[e1].y << ")";
	cv::putText(debug, ss.str(), cont[e1], cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 0, 255), 1);

	ss = std::stringstream();

	ss << "(" << cont[p[1]].x << "," << cont[p[1]].y << ") -> (" << cont[e2].x << "," << cont[e2].y << ")";
	cv::putText(debug, ss.str(), cont[e2], cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 0, 255), 1);
	*/


	if (fabs(vvag) < __ValidParallel){
		cv::line(debug, cont[p[0]], cont[e1], cv::Scalar(255, 0, 255), 2);
		cv::line(debug, cont[p[1]], cont[e2], cv::Scalar(255, 0, 255), 2);
		if (e1 != 0){
			std::vector<double> sav;
			std::vector<cv::Point>sct;
			bool ret = false;
			for (size_t i = e1; !ret || (ret && i != e1); i++){
				if (i == angles.size())i = 0, ret = true;
				sav.push_back(angles[i]);
				sct.push_back(cont[i]);
			}
			angles = sav;
			cont = sct;
		}
		return true;
	}

	e1 = (p[0] + 1 >= cont.size() ? p[0] + 1 - cont.size() : p[0] + 1);
	e2 = (p[1] - 1 < 0 ? p[1] - 1 + cont.size() : p[1] - 1);

	vvag = getAngle(cont[p[0]], cont[e1], cont[p[1]], cont[e2]);

	//cv::line(debug, cont[p[0]], cont[e1], cv::Scalar(0, 255, 255), 2);
	//cv::line(debug, cont[p[1]], cont[e2], cv::Scalar(0, 255, 255), 2);
	ss = std::stringstream();
	ss << vvag;
	cv::putText(debug, ss.str(), cv::Point((cont[e1].x + cont[e2].x) / 2, (cont[e1].y + cont[e2].y) / 2), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 255), 1);
	cv::circle(debug, cont[e1], 2, cv::Scalar(0, 255, 255), 2);
	cv::circle(debug, cont[e2], 2, cv::Scalar(0, 255, 255), 2);
	/*
	ss = std::stringstream();

	ss << "(" << cont[p[0]].x << "," << cont[p[0]].y << ") -> (" << cont[e1].x << "," << cont[e1].y << ")";
	cv::putText(debug, ss.str(), cont[e1], cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 255), 1);

	ss = std::stringstream();

	ss << "(" << cont[p[1]].x << "," << cont[p[1]].y << ") -> (" << cont[e2].x << "," << cont[e2].y << ")";
	cv::putText(debug, ss.str(), cont[e2], cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 255), 1);
	*/


	if (fabs(vvag) < __ValidParallel){
		cv::line(debug, cont[p[0]], cont[e1], cv::Scalar(0, 255, 255), 2);
		cv::line(debug, cont[p[1]], cont[e2], cv::Scalar(0, 255, 255), 2);
		
		

		std::vector<double> sav;
		std::vector<cv::Point>sct;
		bool ret = false;
		for (size_t i = e1; !ret ||(ret && i != e1); i--){
			if (i == -1)i = angles.size() - 1, ret = true;
			sav.push_back(angles[i]);
			sct.push_back(cont[i]);
		}
		if (sav.size() != angles.size())std::cout << sav.size() << ", " << angles.size() << " ERROR" << std::endl;
		angles = sav;
		cont = sct;
		return true;
	}
	return false;
}

bool validArrow(cv::Mat &result, std::vector<double> &angles, std::vector <cv::Point> &cont, double &dir){

	int p[2], c = 0;
	if (!validpoints(result, angles, cont))return false;

	for (int i = 0; i < angles.size(); i++){
		if (angles[i] > 0 && angles[i] < __ValidCorner){
			if (c == 2)return false;
			p[c++] = i;
		}
	}
	
	double maxL = 0;
	int maxp = 0;
	cv::Point maxv(0,0);
	for (int i = p[0] + 1; i < p[1]; i++){
		
		cv::Point lv = getVert(cont[p[0]], cont[p[1]], cont[i]);
		double l = getNorm2(lv);
		if (maxL < l){
			maxL = l;
			maxp = i;
			maxv = lv;
		}
	}
	if (maxv.x == 0 && maxv.y == 0)return false;

	cv::circle(result, cont[maxp], 2, cv::Scalar(211, 0, 148), 2);
	/*
	if (p[1] - p[0] < 3)return false;

	cv::circle(result, cont[p[0]], 2, cv::Scalar(255, 255, 0), 2);
	cv::circle(result, cont[p[1]], 2, cv::Scalar(255, 255, 0), 2);

	int ls = p[0]+1, rs = p[1]-1;

	while (1){
		if (!(angles[ls] < 0 && angles[ls] > -__ValidCorner))ls++;
		if (ls >= rs)return false;
		else break;
	}

	while (1){
		if (!(angles[rs] < 0 && angles[rs] > -__ValidCorner))rs--;
		if (rs <= ls)return false;
		else break;
	}

	cv::circle(result, cont[ls], 2, cv::Scalar(255, 0, 255), 2);
	cv::circle(result, cont[rs], 2, cv::Scalar(255, 0, 255), 2);*/

	dir = getAngle(cv::Point(0, 1), maxv);
	return true;
}


cv::Mat detectArrow(cv::Mat &frame){
	cv::Mat image = frame.clone();
	cv::Mat hsvimage;
	//cv::bilateralFilter(image, hsvimage, 5, 150, 150);
	cv::cvtColor(image, hsvimage, CV_BGR2HSV);

	cv::Mat thres;
	//colorEdgeDetection(image, thres, true);
	cv::inRange(hsvimage, cv::Scalar(120, 50, 50), cv::Scalar(179, 255, 255), thres);

	//colorEdgeDetection(image, image, true);
	
	//std::vector<cv::Vec4i> lines;

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(thres, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	std::vector<std::vector<cv::Point> > contours_poly(contours.size());
	
	//std::vector<cv::Point2f> center(contours.size());
	//std::vector<float> radius(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
		
		//cv::minEnclosingCircle((cv::Mat)contours_poly[i], center[i], radius[i]);
	}

	std::vector<std::vector<double>> cangles;
	std::vector<std::vector<cv::Point>> cont;
	std::vector<cv::Rect> boundRect;

	cv::Mat result = cv::Mat::zeros(image.size(), CV_8UC3);

	for (int i = 0; i< contours_poly.size(); i++){
		std::vector<cv::Point> &c = contours_poly[i];


		if (c.size() < 6)continue;


		cv::drawContours(result, contours_poly, i, cv::Scalar(255,255,255), 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
		//cv::rectangle(result, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
		//cv::circle(result, center[i], (int)radius[i], color, 2, 8, 0);

		std::vector<double> angles;

		
		
		for (int j = 0; j < c.size(); j++){
			cv::Point startPoint = (j - 1 < 0 ? c[j - 1 + c.size()] : c[j - 1]);
			cv::Point midPoint = c[j];
			cv::Point endPoint = (j + 1 >= c.size() ? c[j + 1 - c.size()] : c[j + 1]);

			double angle = getAngle(startPoint, midPoint, endPoint);
			angles.push_back(angle);

			std::stringstream ss;
			ss << j;
			cv::putText(result, ss.str(), midPoint, cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 0), 1);
		}

		cangles.push_back(angles);
		boundRect.push_back(cv::boundingRect(cv::Mat(c)));
		cont.push_back(c);
	}

	//cv::Mat debug = cv::Mat::zeros(image.size(), CV_8UC3);
	for (int i = 0; i < cangles.size(); i++){
		cv::Point center = cv::Point(boundRect[i].x + boundRect[i].width / 2, boundRect[i].y + boundRect[i].height / 2);

		std::string s;
		cv::Scalar bc;
		double dir=0;
		if (validArrow(result, cangles[i], cont[i], dir)){
			cv::rectangle(result, boundRect[i], cv::Scalar(0, 255, 0), 1);
			std::stringstream ss;
			ss << dir;
			cv::putText(result, ss.str(), center, cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 0), 1);
		}
		
	}
	
	return result;
}






int main(void){
	cv::VideoCapture cap(1);

	if (!cap.isOpened()){
		std::cout << "Device cannot open..." << std::endl;
		return 0;
	}

	cv::Mat frame;
	ending::ArrowDetector detector;
	while (1){
		cap >> frame;

		if (frame.size().width <= 0 || frame.size().height <= 0){
			cv::waitKey(10);
			continue;
		}

		
		detector.detect(frame);
		ending::Arrows &arrows = detector.getArrows();

		for (int i = 0; i < arrows.size(); i++){
			cv::rectangle(frame, arrows[i].getBoundingBox(), cv::Scalar(0, 0, 255), 1);
		}

		cv::hconcat(frame, detector.DEBUG_img, frame);
		cv::imshow("result", frame);

		int key = cv::waitKey(10);
		if (key == 27)break;
	}


	return 0;
}


/*
int main(void){

	cv::VideoWriter writer;
	bool start = false;
	char name[50] = {};
	int clip = 0;

	cv::VideoCapture cap(0);

	if (!cap.isOpened()){
		std::cout << "Device cannot open..." << std::endl;
		return 0;
	}

	cv::Mat frame;
	while (1){
		cap >> frame;

		if (frame.size().width <= 0 || frame.size().height <= 0){
			cv::waitKey(10);
			continue;
		}

		cv::Mat result = detectArrow(frame);
		cv::hconcat(frame, result, result);
		cv::imshow("result", result);

		if (start)writer << result;

		int key = cv::waitKey(10);
		if (key == 27)break;
		else if (key == ' '){
			if (start == false){
				start = true;
				sprintf(name, "%03d.wmv", clip++);
				if (!writer.isOpened()){
					writer.open(name, CV_FOURCC('W', 'M', 'V', '2'), 25, cv::Size(result.cols, result.rows), true);
				}
			} else{
				start = false;
				writer.release();
			}
		}
	}


	return 0;
	}*/