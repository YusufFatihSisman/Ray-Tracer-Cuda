#ifndef FACELIST_H
#define FACELIST_H

#include "face.h"

#include <memory>
#include <vector>

class faceList{
    public:
        __device__ faceList(){}

        __device__ bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const;

        face** objects;
        int length;
};

__device__ bool faceList::hit(const ray& r, double t_min, double t_max, hit_record& rec) const{
    hit_record temp;
    bool hit = false;
    double closest_t = t_max;

    for(int i = 0; i < length; i++){
        face currentFace = **(objects+i);
        if(currentFace.hit(r, t_min, closest_t, temp)){
            hit = true;
            rec = temp;
            closest_t = temp.t;
        }
    }
    return hit;
}

#endif