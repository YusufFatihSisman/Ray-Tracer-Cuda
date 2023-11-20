# Ray-Tracer-Cuda
* To build, run **nvcc -o main.o -c main.cu**, **nvcc -o structs.o -c structs.cu**, **nvcc -o raytrace main.o structs.o** respectively
* To execute, run **./raytrace scene_name.xml > output_name.ppm** (xml scene and ppm output)
## Results
* **scene_xml** <br />
![Alt text](https://github.com/YusufFatihSisman/Ray-Tracer-Cuda/blob/main/image.jpg)
* **scene_complex_allreflection_xml** <br />
![Alt text](https://github.com/YusufFatihSisman/Ray-Tracer-Cuda/blob/main/image_complex_allreflection.jpg)
