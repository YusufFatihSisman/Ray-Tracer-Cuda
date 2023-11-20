#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray{
    public:
        __device__ ray(const point3& origin, const vec3& direction): org(origin), dir(direction){}

        __device__ point3 at(double t) const{
            return org + t*dir;
        }

        __device__ point3 origin() const  { return org; }
        __device__ vec3 direction() const { return dir; }
        
    private:
        point3 org;
        vec3 dir;
};

#endif