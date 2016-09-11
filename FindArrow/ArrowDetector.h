
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

#define __ValidCorner 150
#define __ValidParallel 30

namespace ending{
	class Arrow{

	};

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

		static Vector<int> vert(cv::Point s, cv::Point e, cv::Point t){
			Vector<int> v1(s, e), v2(s, t);
			return v2.vert(v1);
		}
	};
}