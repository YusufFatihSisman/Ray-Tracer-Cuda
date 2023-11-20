#ifndef VEC3_H
#define VEC3_H
#include <cmath>
#include <iostream>

using std::sqrt;

class vec3{
    public:
        __host__ __device__ vec3(){
            x = 0;
            y = 0;
            z = 0;
        }
        __host__ __device__ vec3(double x, double y, double z){
            this->x = x;
            this->y = y;
            this->z = z;
        }

        __host__ __device__ vec3 operator-() const {return vec3(-x, -y, -z);}
        
        __host__ __device__ vec3& operator+=(const vec3 &v){
            x += v.x;
            y += v.y;
            z += v.z;
            return *this;
        }

        __host__ __device__ vec3& operator*=(const double t){
            x *= x*t;
            y *= y*t;
            z *= z*t;
            return *this;        
        }

        __host__ __device__ vec3& operator/=(const double t){
            return *this *= 1/t;
        }

        __host__ __device__ double length() const{
            return sqrt(x*x + y*y + z*z);
        }
    
        double x;
        double y;
        double z;
        
};

using point3 = vec3;   // 3D point
using color = vec3;    // RGB color

inline std::ostream& operator<<(std::ostream &out, const vec3 &v) {
    return out << v.x << ' ' << v.y << ' ' << v.z;
}

__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.x + v.x, u.y + v.y, u.z + v.z);
}

__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.x - v.x, u.y - v.y, u.z - v.z);
}

__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.x * v.x, u.y * v.y, u.z * v.z);
}

__host__ __device__ inline vec3 operator*(double t, const vec3 &v) {
    return vec3(t*v.x, t*v.y, t*v.z);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, double t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(vec3 v, double t) {
    return (1/t) * v;
}

__host__ __device__ inline double dot(const vec3 &u, const vec3 &v) {
    return u.x * v.x
         + u.y * v.y
         + u.z * v.z;
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.y * v.z - u.z * v.y,
                u.z * v.x - u.x * v.z,
                u.x * v.y - u.y * v.x);
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

#endif