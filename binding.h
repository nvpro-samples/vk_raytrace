// Binding sets for the scene

// Sets
#define S_ACCEL 0  // Acceleration structure
#define S_OUT 1    // Offscreen output image
#define S_SCENE 2  // Scene data
#define S_ENV 3    // Environment / Sun & Sky
#define S_WF 4     // Wavefront extra data

// Acceleration Structure - Set 0
#define B_TLAS 0

// Output image - Set 1
#define B_SAMPLER 0  // As sampler
#define B_STORE 1    // As storage

// Scene Data - Set 2
#define B_CAMERA 0
#define B_VERTICES 1
#define B_NORMALS 2
#define B_TEXCOORDS 3
#define B_INDICES 4
#define B_MATERIALS 5
#define B_MATRICES 6
#define B_TEXTURES 7
#define B_TANGENTS 8
#define B_COLORS 9
#define B_PRIMLOOKUP 10

// Environment - Set 3
#define B_SUNANDSKY 0
#define B_HDR 1
#define B_IMPORT_SMPL 2
