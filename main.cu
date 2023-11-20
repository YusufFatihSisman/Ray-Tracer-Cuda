#include <iostream>
#include <fstream>
#include <vector>
#include "rapidxml-1.13/rapidxml.hpp"
#include "structs.h"
#include "color.h"
#include "vec3.h"
#include "ray.h"
#include "face.h"
#include "faceList.h"
#include "pointLight.h"
#include <thread>
#include "time.h"
#include <float.h>
#include <sstream>

using namespace std;
using namespace rapidxml;

int maxraytracedepth = -1;
color background;
color ambientlight;
vector<double> vertexdata;
vector<Mesh> meshes;
vector<Material> materials;
vector<pointLight> pointLightVector;
Camera camera;

face** faces;
int faceLength = 0;
pointLight** pointLights;

template <typename T> 
void parse_string_and_assign(string str, T* array){
    stringstream ss(str);
    string word;
    int j = 0;
    while (ss >> word) {
        if(typeid(T) == typeid(int))
            array[j] = atoi(word.c_str());
        else if(typeid(T) == typeid(double))
            array[j] = atof(word.c_str());
        j++;
    }
}

template <typename T> 
void parse_string_and_assign_vector(string str, vector<T>& vec){
    stringstream ss(str);
    string word;
    while (ss >> word) {
        if(typeid(T) == typeid(int))
            vec.push_back(atoi(word.c_str()));
        else if(typeid(T) == typeid(double))
            vec.push_back(atof(word.c_str()));
    }
}

void parse_xml(string filename){
    xml_document<> doc;
    xml_node<> * root_node = NULL;

    ifstream theFile (filename);
	if(!theFile)
		return;
    vector<char> buffer((istreambuf_iterator<char>(theFile)), istreambuf_iterator<char>());
    buffer.push_back('\0');

    doc.parse<0>(&buffer[0]);
   
    root_node = doc.first_node("scene");

    string data = root_node->first_node("maxraytracedepth")->value();
    maxraytracedepth = atoi(data.c_str());

    data =  root_node->first_node("background")->value();
    double bg[3];
    parse_string_and_assign<double>(data, bg);
    background = color(bg[0], bg[1], bg[2]);
    
    double position[3];
    double gaze[3];
    double up[3];
    int nearplane[4];
    double neardistance;
    int imageresolution[2]; 

    xml_node<>* cameraNode =  root_node->first_node("camera");
    data = cameraNode->first_node("position")->value();
    parse_string_and_assign<double>(data, position);   
    data = cameraNode->first_node("gaze")->value();
    parse_string_and_assign<double>(data, gaze);
    data = cameraNode->first_node("up")->value();
    parse_string_and_assign<double>(data, up);
    data = cameraNode->first_node("nearplane")->value();
    parse_string_and_assign<int>(data, nearplane);
    data = cameraNode->first_node("neardistance")->value();
    neardistance = atof(data.c_str());
    data = cameraNode->first_node("imageresolution")->value();
    parse_string_and_assign<int>(data, imageresolution);
    
    initCamera(camera, position, gaze, up, nearplane, neardistance, imageresolution);
    xml_node<>* light_node =  root_node->first_node("lights");

    data =  light_node->first_node("ambientlight")->value();
    double amb[3];
    parse_string_and_assign<double>(data, amb);
    ambientlight = color(amb[0], amb[1], amb[2]);

    //point lights
    for (xml_node<> * point_light_node = light_node->first_node("pointlight"); point_light_node; point_light_node = point_light_node->next_sibling())
    {
        data = point_light_node->first_attribute("id")->value();
        int id = atoi(data.c_str());

        data = point_light_node->first_node("position")->value();
        parse_string_and_assign<double>(data, position);

        double intensity[3];
        data = point_light_node->first_node("intensity")->value(); 
        parse_string_and_assign<double>(data, intensity);

        pointLight pLight = pointLight(id, point3(position[0], position[1], position[2]), color(intensity[0], intensity[1], intensity[2]));
        pointLightVector.push_back(pLight);
    }

    //materials
    xml_node<>* materials_node =  root_node->first_node("materials");
    for (xml_node<> * material_node = materials_node->first_node("material"); material_node; material_node = material_node->next_sibling())
    {
        data = material_node->first_attribute("id")->value();
        int id = atoi(data.c_str());

        double ambient[3];
        data = material_node->first_node("ambient")->value();
        parse_string_and_assign<double>(data, ambient);

        double diffuse[3];
        data = material_node->first_node("diffuse")->value();
        parse_string_and_assign<double>(data, diffuse);
    
        double specular[3];
        data = material_node->first_node("specular")->value();
        parse_string_and_assign<double>(data, specular);

        data = material_node->first_node("phongexponent")->value();
        double phongexponent = atof(data.c_str());

        double reflectance[3];
        data = material_node->first_node("mirrorreflectance")->value();
        parse_string_and_assign<double>(data, reflectance);

        Material material;
        initMaterial(material, id, ambient, diffuse, specular, phongexponent, reflectance);
        materials.push_back(material);

        theFile.close();
    }

    data = root_node->first_node("vertexdata")->value();
    parse_string_and_assign_vector(data, vertexdata);

   //objects
    xml_node<>* objects_node =  root_node->first_node("objects");
    for (xml_node<> * mesh_node = objects_node->first_node("mesh"); mesh_node; mesh_node = mesh_node->next_sibling())
    {
        
        data = mesh_node->first_attribute("id")->value();
        int id = atoi(data.c_str());
        
        data = mesh_node->first_node("materialid")->value();

        int materialId = atoi(data.c_str());
        
        vector<int> faces;
        data = mesh_node->first_node("faces")->value();
        parse_string_and_assign_vector<int>(data, faces);

        Mesh mesh;
        initMesh(mesh, id, materialId, faces);
        meshes.push_back(mesh);
    }

}       

__device__ color ray_color(const ray& r, faceList* faces, int faceLength, pointLight* d_pointLights, int lightAmount, int depth, color background, color ambientlight) {
    hit_record hr;             
    color c = color(0.0, 0.0, 0.0);

    if((*faces).hit(r, 0.0, DBL_MAX, hr)){
        for(int i = 0; i < lightAmount; i++){
            pointLight light =  *(d_pointLights+i);
            if(depth > 0 && (hr.mat_ptr.reflectance.x == 1.0 || hr.mat_ptr.reflectance.y == 1.0 || hr.mat_ptr.reflectance.z == 1.0)){
                vec3 w0 = unit_vector(r.origin() - hr.p);
                vec3 wr = -w0 + 2*hr.normal*(dot(hr.normal, w0));
                c += light.illuminate(r, hr, (*faces).objects, faceLength) + hr.mat_ptr.reflectance * ray_color(ray((hr.p + wr * 0.0001), wr), faces, faceLength, d_pointLights, lightAmount, depth-1, background, ambientlight);
            }else{
                c += light.illuminate(r, hr, (*faces).objects, faceLength);  
            }
        }           
        return c + hr.mat_ptr.ambient * ambientlight; 
    }               
    return background;          
}

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

void setData(){    
    for(int i = 0; i < meshes.size(); i++){
        for(int j = 0; j < meshes[i].faces.size(); j += 3)
        {
            faceLength++;
        }
    }

    faces = (face**)malloc(sizeof(face*) * faceLength);

    int addIndex = 0;
    for(int i = 0; i < meshes.size(); i++){
        int j = 0;
        while(j < meshes[i].faces.size()){
            int index = (meshes[i].faces[j]-1)*3;
            point3 p0 = point3(vertexdata[index], vertexdata[index+1], vertexdata[index+2]);
            index =  (meshes[i].faces[j+1]-1)*3;
            point3 p1 = point3(vertexdata[index], vertexdata[index+1], vertexdata[index+2]);
            index =  (meshes[i].faces[j+2]-1)*3;
            point3 p2 = point3(vertexdata[index], vertexdata[index+1], vertexdata[index+2]);
            Material mt;
            for(int k = 0; k < materials.size(); k++){
                if(meshes[i].materialId == materials[k].id){
                    mt = materials[k];  
                }   
            }
            *(faces + addIndex) = new face(p0, p1, p2, mt);

            j += 3;
            addIndex++;
        }
    }       

    pointLights = (pointLight**)malloc(sizeof(pointLight*) * pointLightVector.size());

    for(int i = 0; i < pointLightVector.size(); i++)
    {
        *(pointLights + i) = new pointLight(pointLightVector[i].id, pointLightVector[i].p, pointLightVector[i].i);
    }

}

__global__ void render(color *colorArray, faceList* faces, int faceLength, pointLight* d_pointLights, int lightAmount, int maxraytracedepth, color background, color ambientlight, int nx, int ny, point3 origin, double l, double r, double b, double t, vec3 u, vec3 v, point3 m, point3 q) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= nx) || (j >= ny)) return;

    int pixel_index = j*nx + i;
    double su = (r-l)*(i + 0.5)/nx;  
    double sv = (t-b)*(j + 0.5)/ny;
    point3 s = q + su*u - sv*v;
    vec3 dir = s - origin;   
    ray currentRay(origin, dir); 

    color pixel_color = ray_color(currentRay, faces, faceLength, d_pointLights, lightAmount, maxraytracedepth, background, ambientlight);
    colorArray[pixel_index] = normalizeColor(pixel_color);
}


__global__ void fill_device(faceList *faceList, face currentFace, int index, int faceLength)
{
    if(threadIdx.x == 0 && blockIdx.x == 0) 
    {
        if(index == 0){
            (*faceList).length = faceLength;
            (*faceList).objects = new face*[faceLength];
        }
        *((*faceList).objects + index) = new face(currentFace.vertices[0], currentFace.vertices[1], currentFace.vertices[2], currentFace.mat_ptr);
    }
}

__global__ void empty_device(faceList *faceList, int faceLength)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i >= faceLength)
        return;
    
    delete(*((*faceList).objects + i));

    if(i == faceLength - 1)
    {
        delete((*faceList).objects);
    }
}

__global__ void fill_pointlights(pointLight* pointLights, pointLight pl, int index)
{
    if(threadIdx.x == 0 && blockIdx.x == 0) 
    {
        *(pointLights + index) = pointLight(pl.id, pl.p, pl.i);
    }
}

int main(int argc, char** argv){
	if(argc == 1)
		return 0;
    time_t start, end;
    time(&start);

    parse_xml(argv[1]);
	if(maxraytracedepth == -1)
		return 0;
	
    setData();   

    int nx = camera.imageresolution[0];
    int ny = camera.imageresolution[1];
    color* array;

    checkCudaErrors(cudaMallocManaged((void **)&array, sizeof(color)*ny*nx));

    point3 origin = point3(camera.position.x, camera.position.y, camera.position.z);
    double l = camera.nearplane[0];
    double r = camera.nearplane[1]; 
    double b = camera.nearplane[2]; 
    double t = camera.nearplane[3];
    double dist = camera.neardistance;
    vec3 w = -vec3(camera.gaze.x, camera.gaze.y, camera.gaze.z);
    vec3 v = vec3(camera.up.x, camera.up.y, camera.up.z);
    vec3 u = cross(v, w);       
    point3 m = origin + -w*dist;   
    point3 q = m + l*u + t*v;

    faceList* d_facelist;
    pointLight* d_pointLights;


    checkCudaErrors(cudaMalloc(&d_facelist, sizeof(faceList)));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMalloc(&d_pointLights, sizeof(pointLight*) * pointLightVector.size()));
    checkCudaErrors(cudaDeviceSynchronize());

    for(int i = 0; i < faceLength; i++)
    {
        fill_device<<<1,1>>>(d_facelist, **(faces + i), i, faceLength);
    }
    for(int i = 0; i < pointLightVector.size(); i++)
    {
        fill_pointlights<<<1,1>>>(d_pointLights, **(pointLights + i), i);
    }
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    int tx = 8;
    int ty = 8;
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    render<<<blocks, threads>>>(array, d_facelist, faceLength, d_pointLights, pointLightVector.size(), maxraytracedepth, background, ambientlight, nx, ny, origin, l, r, b, t, u, v, m, q);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
	time(&end);
	double time_taken = double(end - start);
    std::cerr << "Time taken before print : " << fixed
         << time_taken;
    std::cerr << " sec " << endl;
    
	std::cout << "P3\n" << nx << ' ' << ny << "\n255\n";
    for(int j = 0; j < ny; j++){    
        //std::cerr << "\rLine Completed: " << j << ' ' << std::flush;
		for (int i = 0; i < nx; i++) {
            int pixel_index = j*nx + i;
			std::cout << array[pixel_index] << "\n";
		}
	}
    
	checkCudaErrors(cudaDeviceSynchronize());
    int block = faceLength/tx > 0 ? faceLength/tx : 1;
    empty_device<<<block,tx>>>(d_facelist, faceLength);
    block = pointLightVector.size()/tx > 0 ? faceLength/tx : 1;
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(array));
    checkCudaErrors(cudaFree(d_pointLights));
    
    time(&end);
    time_taken = double(end - start);
    std::cerr << "Time taken by program is : " << fixed
         << time_taken;
    std::cerr << " sec " << endl;
    
    return 0;
}
