# Spatial Frequency Extraction using Gradient-liked Operator - Three Dimension (SFEGO_3D)
## PyCUDA Version
### Introduction
- In 3D data, There are Multi-dimensional Ensemble Empirical Mode Decomposition (MEEMD) and Three-dimensional Empirical Mode Decomposition(TEMD) can decompose the data into several 3D Intrinsic Mode Functions (TIMFs)

- Now, we are using Spherical Gradient-liked Operator that can choose different Radius=Wavelength (different spatial frequency) to do Differential=Gradient on 3D data to get the vector map (magnitude and direction) Then we can do Integral on vector map to get 3D Spatial Data that contain such a spatial frequency information in specific Radius

- This work using Spherical Coordinate System to generate (build_list_3d_sphere) the Sphere parameter within execute_radius

- And we are calcualte the average of each semisphere on different angles (traverse each point of the surface of sphere and calcuate the average difference of positive semisphere and negative semisphere that is the differential calculation)

- To achieve the faster way to calculate the 3D Spatial Data. The Dynamic Programing (generate_surface_dp_list) is used to cache next Semisphere indexs of Positive Side and Negative Side.

### 3D Data
- Magnetic Resonance Imaging (MRI)

- Computed Tomography (CT)

- Atmosphere Data

- The Gravity Data in the space. Consider you have a moon size spaceship that want to pass our solar system with very high speed... Each planet on our solar system have higher mass than your spaceship. So... if you want to use different size of the anti-gravity generator to cancel entire full spectrum gravity...

### Hardware Requirement
- Require NVIDIA GPU to execute CUDA Kernel Code

- Recommend to use NVIDIA GPU with 1GB+ VRAM (VRAM usage is depend on Data Size and default_radius)

### Execution
- python SFEGO_3D.py
