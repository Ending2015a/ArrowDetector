
#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <vector>
#include <string>

#include <cmath>
#include <cstdio>

#include "Detector.h"


namespace ending{


	template<typename TYPE > class Vector;
	class Arrow;
	class ArrowDetector;


	template<typename TYPE>
	class Vector{
	public:
		TYPE x, y;

		Vector(TYPE _x, TYPE _y){
			x = _x;
			y = _y;
		}

		Vector(cv::Point s, cv::Point e){
			x = e.x - s.x;
			y = e.y - s.y;
		}

		Vector(const Vector &v){
			x = v.x;
			y = v.y;
		}

		Vector &operator=(const Vector &v){
			x = v.x;
			y = v.y;
			return *this;
		}

		Vector &operator+=(const Vector &v){
			this->x += v.x;
			this->y += v.y;
			return *this;
		}

		Vector &operator-=(const Vector &v){
			this->x -= v.x;
			this->y -= v.y;
			return *this;
		}

		Vector &operator*=(const TYPE &a){
			this->x *= a;
			this->y *= a;
		}

		Vector &operator/=(const TYPE &a){
			this->x /= a;
			this->y /= a;
		}

		Vector operator+(const Vector &v)const{
			Vector ans(*this);
			return ans+=v;
		}

		Vector operator-(const Vector &v)const{
			Vector ans(*this);
			return ans -= v;
		}

		Vector operator*(const TYPE &a)const{
			Vector ans(*this);
			return ans *= v;
		}

		Vector operator/(const TYPE &a)const{
			Vector ans(*this);
			return ans /= v;
		}

		friend std::ostream &operator<<(std::ostream &out, const Vector<TYPE> &v){
			out << "(" << v.x << ", " << v.y << ")";
			return out;
		}

		TYPE operator*(const Vector<TYPE> &v){
			TYPE a = this->x * v.x + this->y * v.y;
			return a;
		}

		TYPE operator/(const Vector<TYPE> &v){
			TYPE a = this->x * v.y - this->y * v.x;
			return a;
		}

		double norm()const{
			return sqrt(pow((double)x, 2) + pow((double)y, 2));
		}

		double norm2()const{
			return pow((double)x, 2) + pow((double)y, 2);
		}

		//this to v
		double angle(const Vector<TYPE> &v, bool direct = false)const{
			double nv1 = this->norm();
			double nv2 = v.norm();

			double angle = acos(((double)(x*v.x + y*v.y) / (nv1*nv2))) * 180 / CV_PI;

			if (direct == false)return angle;

			if (x * v.y - v.x * y < 0)return angle;
			else return -angle;
		}

		//this proj to v
		Vector proj(const Vector<TYPE> &v)const{
			double a = (x*v.x + y*v.y) / v.norm2();
			Vector<TYPE> p((TYPE)(v.x*a), (TYPE)(v.y*a));
			return p;
		}

		//this vert to v
		Vector vert(const Vector<TYPE> &v)const{
			Vector<TYPE> p = proj(v);
			return (*this) - p;
		}

		static double angle(cv::Point s1, cv::Point e1, cv::Point s2, cv::Point e2, bool direct = false){
			Vector<int> v1(s1, e1), v2(s2,e2);
			return v1.angle(v2, direct);
		}

		static Vector vert(cv::Point s, cv::Point e, cv::Point t){
			Vector<TYPE> v1(s, e), v2(s, t);
			return v2.vert(v1);
		}
	};


	class Arrow{
		friend class ArrowDetector;
	private:
		cv::Point _pos;
		cv::Point _head;
		double _ort;
		std::vector<cv::Point> _point;
		std::vector<double> _angle;
		cv::Point _neck[2];
		cv::Point _tail[2];
		cv::Point _side[2];

		cv::Rect _boundrect;
	public:
		Arrow(){ }
		Arrow(const Arrow &a){
			_pos = a._pos;
			_head = a._head;
			_ort = a._ort;
			_point = a._point;
			_angle = a._angle;
			_boundrect = a._boundrect;
		}


	public:
		cv::Point getPosition(){
			return _pos;
		}

		cv::Point getTop(){
			return _head;
		}

		double getOrient(){
			return _ort;
		}

		std::vector<double> getAngle(){
			return _angle;
		}

		std::vector<cv::Point> getPoint(){
			return _point;
		}

		cv::Point *getNeck(){
			return _neck;
		}

		cv::Point *getTail(){
			return _tail;
		}

		cv::Point *getSide(){
			return _side;
		}

		cv::Rect getBoundingBox(){
			return _boundrect;
		}


	};

	typedef std::vector<Arrow> Arrows;


	class ArrowDetector{
	private:
		Arrows arrows;
		cv::Scalar _lower;
		cv::Scalar _upper;
		double vcorner;
		double vparal;
		double vacute;

#ifdef __ARROW_DEBUG_MODE___
	public:
		cv::Mat DEBUG_img;
#endif

	private:
		void threscolor(cv::Mat &image){
			cv::Mat hsvimage;
			cv::cvtColor(image, hsvimage, CV_BGR2HSV);

			if (_lower[0] <= _upper[0]){
				cv::inRange(hsvimage, _lower, _upper, image);
			} else{
				cv::inRange(hsvimage, _lower, cv::Scalar(179, _upper[1], _upper[2]), image);
				cv::Mat temp;
				cv::inRange(hsvimage, cv::Scalar(0, _lower[1], _lower[2]), _upper, temp);
				cv::bitwise_or(image, temp, image);
			}
			

		}

		int test(std::vector<double> &ang, std::vector<cv::Point> &cont, size_t n1, size_t t1, size_t n2, size_t t2){
			double vvag = Vector<int>::angle(cont[n1], cont[t1], cont[n2], cont[t2]);
			/*
#ifdef __ARROW_DEBUG_MODE___
			std::stringstream ss;
			ss << vvag;
			cv::putText(DEBUG_img, ss.str(), cont[t1], cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 0, 255), 1);
			cv::circle(DEBUG_img, cont[t1], 2, cv::Scalar(255, 0, 255), 2);
			cv::circle(DEBUG_img, cont[t2], 2, cv::Scalar(255, 0, 255), 2);
#endif*/


			if (fabs(vvag) < vparal){
#ifdef __ARROW_DEBUG_MODE___
				std::stringstream ss;
				ss << vvag;
				cv::putText(DEBUG_img, ss.str(), cont[t1], cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 0, 255), 1);
				cv::circle(DEBUG_img, cont[t1], 2, cv::Scalar(0, 255, 255), 2);
				cv::circle(DEBUG_img, cont[t2], 2, cv::Scalar(0, 255, 255), 2);
				cv::line(DEBUG_img, cont[n1], cont[t1], cv::Scalar(0, 255, 255), 2);
				cv::line(DEBUG_img, cont[n2], cont[t2], cv::Scalar(0, 255, 255), 2);
#endif
				return 1;
			}
			else if (ang[t1] < vacute && ang[t2] < vacute){
#ifdef __ARROW_DEBUG_MODE___
				std::stringstream ss;
				ss << vvag;
				cv::putText(DEBUG_img, ss.str(), cont[t1], cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 0, 255), 1);
				cv::circle(DEBUG_img, cont[t1], 2, cv::Scalar(255, 0, 255), 2);
				cv::circle(DEBUG_img, cont[t2], 2, cv::Scalar(255, 0, 255), 2);
				cv::line(DEBUG_img, cont[n1], cont[t1], cv::Scalar(255, 0, 255), 2);
				cv::line(DEBUG_img, cont[n2], cont[t2], cv::Scalar(255, 0, 255), 2);
#endif
				return 2;
			}
			return 0;

		}

		bool validpoints(std::vector<double> &angles, std::vector<cv::Point> &cont, Arrow &arrow){
			std::vector<double> va;
			std::vector<cv::Point> vc;

			bool findTail = false;
			size_t tailp = -1;
			bool findSide = false;

			//check valid point --- if angle bigger than vcorner -> invalid point 
			for (int i = 0; i < angles.size(); i++){
				if (fabs(angles[i]) < vcorner){
					va.push_back(angles[i]);
					vc.push_back(cont[i]);
				}
			}

			// point less then 6 -> not enough 
			if (va.size() < 6)return false;

			angles = va;
			cont = vc;


			int p[2] = {};
			int c = 0;

			//find neck
			for (int i = 0; i < angles.size(); i++){
				if (angles[i] > 0 && angles[i] < vcorner){
					if (c == 2)return false;
					p[c++] = i;
				}
			}

			//neck is less than 2
			if (c < 2)return false;

#ifdef __ARROW_DEBUG_MODE___
			cv::circle(DEBUG_img, cont[p[0]], 2, cv::Scalar(255, 255, 0), 2);
			cv::circle(DEBUG_img, cont[p[1]], 2, cv::Scalar(255, 255, 0), 2);
#endif
			
			
			
			//first test
			size_t e1 = (p[0] - 1 < 0 ? p[0] - 1 + cont.size() : p[0] - 1);
			size_t e2 = (p[1] + 1 >= cont.size() ? p[1] + 1 - cont.size() : p[1] + 1);

			int result = test(angles, cont, p[0], e1, p[1], e2);


			if (result == 1){
				findTail = true;
				tailp = e1;
				arrow._tail[0] = cont[e1];
				arrow._tail[1] = cont[e2];
				arrow._neck[0] = cont[p[0]];
				arrow._neck[1] = cont[p[1]];

			} else if(result == 2){
				findSide = true;
				arrow._side[0] = cont[e1];
				arrow._side[1] = cont[e2];
				arrow._neck[1] = cont[p[0]];
				arrow._neck[0] = cont[p[1]];
			}
			else return false;


			e1 = (p[0] + 1 >= cont.size() ? p[0] + 1 - cont.size() : p[0] + 1);
			e2 = (p[1] - 1 < 0 ? p[1] - 1 + cont.size() : p[1] - 1);

			result = test(angles, cont, p[0], e1, p[1], e2);

			if (result == 1){
				if (findSide == false)return false;
				arrow._tail[0] = cont[e1];
				arrow._tail[1] = cont[e2];

				std::vector<double> sav;
				std::vector<cv::Point>sct;
				bool ret = false;
				for (size_t i = e1; !ret || (ret && i != e1); i--){
					if (i == -1)i = angles.size() - 1, ret = true;
					sav.push_back(angles[i]);
					sct.push_back(cont[i]);
				}

				arrow._angle = sav;
				arrow._point = sct;
				
			}
			else if (result == 2){
				if (findTail == false)return false;
				arrow._side[0] = cont[e1];
				arrow._side[1] = cont[e2];

				if (tailp != 0){
					std::vector<double> sav;
					std::vector<cv::Point>sct;

					for (size_t i = 0; i < angles.size(); i++){
						sav.push_back(angles[(i + tailp) % angles.size()]);
						sct.push_back(cont[(i + tailp) % angles.size()]);
					}

					/*
					bool ret = false;
					for (size_t i = tailp; !ret || (ret && i != tailp); i++){
						if (i == angles.size())i = 0, ret = true;
						sav.push_back(angles[i]);
						sct.push_back(cont[i]);
					}*/
					arrow._angle = sav;
					arrow._point = sct;
				}

				
			}
			else return false;
			return true;
			/*
			double vvag = Vector<int>::angle(cont[p[0]], cont[e1], cont[p[1]], cont[e2]);

#ifdef __ARROW_DEBUG_MODE___
			std::stringstream ss;
			ss << vvag;
			cv::putText(DEBUG_img, ss.str(), cont[e1], cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 0, 255), 1);
			cv::circle(DEBUG_img, cont[e1], 2, cv::Scalar(255, 0, 255), 2);
			cv::circle(DEBUG_img, cont[e2], 2, cv::Scalar(255, 0, 255), 2);
#endif
			


			if (fabs(vvag) < vparal){
#ifdef __ARROW_DEBUG_MODE___
				cv::line(DEBUG_img, cont[p[0]], cont[e1], cv::Scalar(255, 0, 255), 2);
				cv::line(DEBUG_img, cont[p[1]], cont[e2], cv::Scalar(255, 0, 255), 2);
#endif
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
				findTail = true;
			} else if(angles[e1] < vacute && angles[e2] < vacute){
				findSide = true;
			}


			//second test
			e1 = (p[0] + 1 >= cont.size() ? p[0] + 1 - cont.size() : p[0] + 1);
			e2 = (p[1] - 1 < 0 ? p[1] - 1 + cont.size() : p[1] - 1);

			vvag = getAngle(cont[p[0]], cont[e1], cont[p[1]], cont[e2]);

			//cv::line(debug, cont[p[0]], cont[e1], cv::Scalar(0, 255, 255), 2);
			//cv::line(debug, cont[p[1]], cont[e2], cv::Scalar(0, 255, 255), 2);
#ifdef __ARROW_DEBUG_MODE___
			ss = std::stringstream();
			ss << vvag;
			cv::putText(DEBUG_img, ss.str(), cv::Point((cont[e1].x + cont[e2].x) / 2, (cont[e1].y + cont[e2].y) / 2), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 255), 1);
			cv::circle(DEBUG_img, cont[e1], 2, cv::Scalar(0, 255, 255), 2);
			cv::circle(DEBUG_img, cont[e2], 2, cv::Scalar(0, 255, 255), 2);
#endif


			if (fabs(vvag) < vparal){

#ifdef __ARROW_DEBUG_MODE___
				cv::line(debug, cont[p[0]], cont[e1], cv::Scalar(0, 255, 255), 2);
				cv::line(debug, cont[p[1]], cont[e2], cv::Scalar(0, 255, 255), 2);

#endif

				std::vector<double> sav;
				std::vector<cv::Point>sct;
				bool ret = false;
				for (size_t i = e1; !ret || (ret && i != e1); i--){
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
			*/
		}

		bool validArrow(std::vector<double> &angles, std::vector <cv::Point> &cont, Arrow &arrow){
			if (!validpoints(angles, cont, arrow))return false;

			int p[2], c = 0;
			for (int i = 0; i < angles.size(); i++){
				if (angles[i] > 0 && angles[i] < vcorner){
					if (c == 2)return false;
					p[c++] = i;
				}
			}

			double maxL = 0;
			int maxp = 0;
			Vector<int> maxv(0, 0);
			for (int i = p[0] + 1; i < p[1]; i++){

				Vector<int> lv = Vector<int>::vert(cont[p[0]], cont[p[1]], cont[i]);
				double l = lv.norm2();
				if (maxL < l){
					maxL = l;
					maxp = i;
					maxv = lv;
				}
			}
			if (maxv.x == 0 && maxv.y == 0)return false;

#ifdef __ARROW_DEBUG_MODE___
			cv::circle(DEBUG_img, cont[maxp], 2, cv::Scalar(211, 0, 148), 2);
#endif
			arrow._ort = Vector<int>(0, 1).angle(maxv);
			arrow._head = cont[maxp];

			return true;
		}

		void findArrow(cv::Mat &image){


			//step1: find contours
			std::vector<std::vector<cv::Point> > contours;
			std::vector<cv::Vec4i> hierarchy;

			cv::findContours(image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));


			//step2: approx contours to poly
			std::vector<std::vector<cv::Point> > contours_poly(contours.size());

			for (int i = 0; i < contours.size(); i++)
			{
				cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
			}


			std::vector<std::vector<double>> cangles;
			std::vector<std::vector<cv::Point>> cont;
			std::vector<cv::Rect> boundRect;



			for (int i = 0; i < contours_poly.size(); i++){

				std::vector<cv::Point> &c = contours_poly[i];
				if (c.size() < 6)continue;

#ifdef __ARROW_DEBUG_MODE___
				cv::drawContours(DEBUG_img, contours_poly, i, cv::Scalar(255, 255, 255), 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
#endif
				
				std::vector<double> angles;

				for (int j = 0; j < c.size(); j++){
					cv::Point sP = (j - 1 < 0 ? c[j - 1 + c.size()] : c[j - 1]);
					cv::Point mP = c[j];
					cv::Point eP = (j + 1 >= c.size() ? c[j + 1 - c.size()] : c[j + 1]);

					double angle = Vector<int>::angle(mP, sP, mP, eP, true);
					angles.push_back(angle);
#ifdef __ARROW_DEBUG_MODE___
					std::stringstream ss;
					ss << j;
					cv::putText(DEBUG_img, ss.str(), mP, cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 0), 1);
#endif
				}

				cangles.push_back(angles);
				boundRect.push_back(cv::boundingRect(cv::Mat(c)));
				cont.push_back(c);
			}

			for (int i = 0; i < cangles.size(); i++){
				cv::Point center = cv::Point(boundRect[i].x + boundRect[i].width / 2, boundRect[i].y + boundRect[i].height / 2);

				Arrow arrow;

				if (validArrow(cangles[i], cont[i], arrow)){
					
					arrow._pos = center;
					arrow._boundrect = boundRect[i];
#ifdef __ARROW_DEBUG_MODE___
					cv::rectangle(DEBUG_img, boundRect[i], cv::Scalar(0, 255, 0), 1);
					std::stringstream ss;
					ss << arrow.getOrient();
					cv::putText(DEBUG_img, ss.str(), center, cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 0), 1);
#endif
					arrows.push_back(arrow);
				}
			}
		}


	public:

		ArrowDetector(cv::Scalar lower = cv::Scalar(120,50,50) , cv::Scalar upper = cv::Scalar(179,255,255), 
			double validCorner = 150, double validParal = 30, double validAcute = 90){
			_lower = lower;
			_upper = upper;
			vcorner = validCorner;
			vparal = validParal;
			vacute = validAcute;
		}

		//image = CV_8UC3
		size_t detect(cv::Mat &image, cv::Scalar lower, cv::Scalar upper){
			Arrows().swap(arrows);
			cv::Mat img = image.clone();

#ifdef __ARROW_DEBUG_MODE___
			DEBUG_img = cv::Mat::zeros(image.size(), CV_8UC3);
#endif

			_lower = lower;
			_upper = upper;
			if (image.channels() == 3)threscolor(img);

			findArrow(img);

			return arrows.size();
		}

		size_t detect(cv::Mat &image){
			return detect(image, _lower, _upper);
		}

		Arrows getArrows(){
			return arrows;
		}


	};
}