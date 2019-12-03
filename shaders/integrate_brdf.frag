#version 450

// This shader computes a glossy BRDF map to be used with the Unreal 4 PBR shading model as
// described in
//
// "Real Shading in Unreal Engine 4" by Brian Karis
// http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
//


layout (location = 0) in vec2 inUV;
layout (location = 0) out vec4 outColor;

const float PI = 3.14159265359;

// See http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
float radinv(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 hammersley_2d(uint i, uint N)
{
    return vec2(float(i)/float(N), radinv(i));
}

vec3 ggx_sample(vec2 xi, vec3 normal, float alpha)
{
    // compute half-vector in spherical coordinates
    float phi = 2.0 * PI * xi.x;
    float cos_theta = sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y));
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    return vec3(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta);
}

float geometry_schlick_ggx(float ndotv, float roughness)
{
    // note that we use a different k for IBL
    float a = roughness;
    float k = (a * a) / 2.0;

    float nom   = ndotv;
    float denom = ndotv * (1.0 - k) + k;

    return nom / denom;
}

float geometry_smith(vec3 normal, vec3 view, vec3 light, float roughness)
{
    float ndotv = max(dot(normal, view), 0.0);
    float ndotl = max(dot(normal, light), 0.0);
    float g1 = geometry_schlick_ggx(ndotv, roughness);
    float g2 = geometry_schlick_ggx(ndotl, roughness);

    return g1 * g2;
}

vec2 integrate_brdf(float ndotv, float roughness)
{
    vec3 view;
    view.x = sqrt(1.0 - ndotv * ndotv); // sin
    view.y = 0.0;
    view.z = ndotv;

    float A = 0.0;
    float B = 0.0;

    const vec3 normal    = vec3(0.0, 0.0, 1.0);

    const uint nsamples = 1024u;
    float alpha = roughness * roughness;

    for(uint i = 0u; i < nsamples; ++i)
    {
        vec2 xi = hammersley_2d(i, nsamples);
        vec3 h0  = ggx_sample(xi, normal, alpha);
        vec3 h = vec3(h0.y, -h0.x, h0.z);
        
        vec3 light  = normalize(2.0 * dot(view, h) * h - view);

        float ndotl = max(light.z, 0.0);
        float ndoth = max(h.z, 0.0);
        float vdoth = max(dot(view, h), 0.0);

        if(ndotl > 0.0)
        {
            float G = geometry_smith(normal, view, light, roughness);
            float G_Vis = (G * vdoth) / (ndoth * ndotv);
            float Fc = pow(1.0 - vdoth, 5.0);

            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    A /= float(nsamples);
    B /= float(nsamples);
    return vec2(A, B);
}

void main() 
{
	vec2 brdf = integrate_brdf(inUV.s, 1.0-inUV.t);
	outColor = vec4(brdf,0.0,1.0);
}