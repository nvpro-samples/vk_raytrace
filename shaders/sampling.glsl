//! #version 430

/*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

const float ONE_OVER_PI = 0.3183099;
const float PI          = 3.141592653589;

//-------------------------------------------------------------------------------------------------
// random number generator based on the Optix SDK
//-------------------------------------------------------------------------------------------------

uint tea(uint val0, uint val1)
{
  uint v0 = val0;
  uint v1 = val1;
  uint s0 = 0;

  for(uint n = 0; n < 16; n++)
  {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }

  return v0;
}

// Generate random unsigned int in [0, 2^24)
uint lcg(inout uint prev)
{
  uint LCG_A = 1664525u;
  uint LCG_C = 1013904223u;
  prev       = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

uint lcg2(inout uint prev)
{
  prev = (prev * 8121 + 28411) % 134456;
  return prev;
}

// Generate random float in [0, 1)
float rnd(inout uint prev)
{
  return (float(lcg(prev)) / float(0x01000000));
}

uint rot_seed(uint seed, uint frame)
{
  return seed ^ frame;
}


uint hash_value(uint a)
{
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);

  // Keep only the 16 lower bits to avoid precision issues in radinv
  // returning either c1 or c2 without conditions
  const uint c1   = (a & 0x0000FFFF);
  const uint c2   = (a & 0xFFFF0000) >> 16;
  const int  mask = -(int(c1 != 0));
  return (c1 & mask) | ((c2 & ~mask));
}

uint get_prime(uint prime_perm)
{
  return prime_perm & 0xFFFF;
}

uint get_permTN2(uint prime_perm)
{
  return prime_perm >> 16;
}

#define PRIME_PERM_COUNT 360
// Embed prime numbers in the first 16 bits, and permutations in the last 16
// bits
uint prime_permTN2[PRIME_PERM_COUNT] = {
    2 + (1 << 16),       3 + (1 << 16),       5 + (3 << 16),       7 + (3 << 16),       11 + (4 << 16),
    13 + (9 << 16),      17 + (7 << 16),      19 + (5 << 16),      23 + (9 << 16),      29 + (18 << 16),
    31 + (18 << 16),     37 + (8 << 16),      41 + (13 << 16),     43 + (31 << 16),     47 + (9 << 16),
    53 + (19 << 16),     59 + (36 << 16),     61 + (33 << 16),     67 + (21 << 16),     71 + (44 << 16),
    73 + (43 << 16),     79 + (61 << 16),     83 + (60 << 16),     89 + (56 << 16),     97 + (26 << 16),
    101 + (71 << 16),    103 + (32 << 16),    107 + (77 << 16),    109 + (26 << 16),    113 + (95 << 16),
    127 + (92 << 16),    131 + (47 << 16),    137 + (29 << 16),    139 + (61 << 16),    149 + (57 << 16),
    151 + (69 << 16),    157 + (115 << 16),   163 + (63 << 16),    167 + (92 << 16),    173 + (31 << 16),
    179 + (104 << 16),   181 + (126 << 16),   191 + (50 << 16),    193 + (80 << 16),    197 + (55 << 16),
    199 + (152 << 16),   211 + (114 << 16),   223 + (80 << 16),    227 + (83 << 16),    229 + (97 << 16),
    233 + (95 << 16),    239 + (150 << 16),   241 + (148 << 16),   251 + (55 << 16),    257 + (80 << 16),
    263 + (192 << 16),   269 + (71 << 16),    271 + (76 << 16),    277 + (82 << 16),    281 + (109 << 16),
    283 + (105 << 16),   293 + (173 << 16),   307 + (58 << 16),    311 + (143 << 16),   313 + (56 << 16),
    317 + (177 << 16),   331 + (203 << 16),   337 + (239 << 16),   347 + (196 << 16),   349 + (143 << 16),
    353 + (278 << 16),   359 + (227 << 16),   367 + (87 << 16),    373 + (274 << 16),   379 + (264 << 16),
    383 + (84 << 16),    389 + (226 << 16),   397 + (163 << 16),   401 + (231 << 16),   409 + (177 << 16),
    419 + (95 << 16),    421 + (116 << 16),   431 + (165 << 16),   433 + (131 << 16),   439 + (156 << 16),
    443 + (105 << 16),   449 + (188 << 16),   457 + (142 << 16),   461 + (105 << 16),   463 + (125 << 16),
    467 + (269 << 16),   479 + (292 << 16),   487 + (215 << 16),   491 + (182 << 16),   499 + (294 << 16),
    503 + (152 << 16),   509 + (148 << 16),   521 + (144 << 16),   523 + (382 << 16),   541 + (194 << 16),
    547 + (346 << 16),   557 + (323 << 16),   563 + (220 << 16),   569 + (174 << 16),   571 + (133 << 16),
    577 + (324 << 16),   587 + (215 << 16),   593 + (246 << 16),   599 + (159 << 16),   601 + (337 << 16),
    607 + (254 << 16),   613 + (423 << 16),   617 + (484 << 16),   619 + (239 << 16),   631 + (440 << 16),
    641 + (362 << 16),   643 + (464 << 16),   647 + (376 << 16),   653 + (398 << 16),   659 + (174 << 16),
    661 + (149 << 16),   673 + (418 << 16),   677 + (306 << 16),   683 + (282 << 16),   691 + (434 << 16),
    701 + (196 << 16),   709 + (458 << 16),   719 + (313 << 16),   727 + (512 << 16),   733 + (450 << 16),
    739 + (161 << 16),   743 + (315 << 16),   751 + (441 << 16),   757 + (549 << 16),   761 + (555 << 16),
    769 + (431 << 16),   773 + (295 << 16),   787 + (557 << 16),   797 + (172 << 16),   809 + (343 << 16),
    811 + (472 << 16),   821 + (604 << 16),   823 + (297 << 16),   827 + (524 << 16),   829 + (251 << 16),
    839 + (514 << 16),   853 + (385 << 16),   857 + (531 << 16),   859 + (663 << 16),   863 + (674 << 16),
    877 + (255 << 16),   881 + (519 << 16),   883 + (324 << 16),   887 + (391 << 16),   907 + (394 << 16),
    911 + (533 << 16),   919 + (253 << 16),   929 + (717 << 16),   937 + (651 << 16),   941 + (399 << 16),
    947 + (596 << 16),   953 + (676 << 16),   967 + (425 << 16),   971 + (261 << 16),   977 + (404 << 16),
    983 + (691 << 16),   991 + (604 << 16),   997 + (274 << 16),   1009 + (627 << 16),  1013 + (777 << 16),
    1019 + (269 << 16),  1021 + (217 << 16),  1031 + (599 << 16),  1033 + (447 << 16),  1039 + (581 << 16),
    1049 + (640 << 16),  1051 + (666 << 16),  1061 + (595 << 16),  1063 + (669 << 16),  1069 + (686 << 16),
    1087 + (305 << 16),  1091 + (460 << 16),  1093 + (599 << 16),  1097 + (335 << 16),  1103 + (258 << 16),
    1109 + (649 << 16),  1117 + (771 << 16),  1123 + (619 << 16),  1129 + (666 << 16),  1151 + (669 << 16),
    1153 + (707 << 16),  1163 + (737 << 16),  1171 + (854 << 16),  1181 + (925 << 16),  1187 + (818 << 16),
    1193 + (424 << 16),  1201 + (493 << 16),  1213 + (463 << 16),  1217 + (535 << 16),  1223 + (782 << 16),
    1229 + (476 << 16),  1231 + (451 << 16),  1237 + (520 << 16),  1249 + (886 << 16),  1259 + (340 << 16),
    1277 + (793 << 16),  1279 + (390 << 16),  1283 + (381 << 16),  1289 + (274 << 16),  1291 + (500 << 16),
    1297 + (581 << 16),  1301 + (345 << 16),  1303 + (363 << 16),  1307 + (1024 << 16), 1319 + (514 << 16),
    1321 + (773 << 16),  1327 + (932 << 16),  1361 + (556 << 16),  1367 + (954 << 16),  1373 + (793 << 16),
    1381 + (294 << 16),  1399 + (863 << 16),  1409 + (393 << 16),  1423 + (827 << 16),  1427 + (527 << 16),
    1429 + (1007 << 16), 1433 + (622 << 16),  1439 + (549 << 16),  1447 + (613 << 16),  1451 + (799 << 16),
    1453 + (408 << 16),  1459 + (856 << 16),  1471 + (601 << 16),  1481 + (1072 << 16), 1483 + (938 << 16),
    1487 + (322 << 16),  1489 + (1142 << 16), 1493 + (873 << 16),  1499 + (629 << 16),  1511 + (1071 << 16),
    1523 + (1063 << 16), 1531 + (1205 << 16), 1543 + (596 << 16),  1549 + (973 << 16),  1553 + (984 << 16),
    1559 + (875 << 16),  1567 + (918 << 16),  1571 + (1133 << 16), 1579 + (1223 << 16), 1583 + (933 << 16),
    1597 + (1110 << 16), 1601 + (1228 << 16), 1607 + (1017 << 16), 1609 + (701 << 16),  1613 + (480 << 16),
    1619 + (678 << 16),  1621 + (1172 << 16), 1627 + (689 << 16),  1637 + (1138 << 16), 1657 + (1022 << 16),
    1663 + (682 << 16),  1667 + (613 << 16),  1669 + (635 << 16),  1693 + (984 << 16),  1697 + (526 << 16),
    1699 + (1311 << 16), 1709 + (459 << 16),  1721 + (1348 << 16), 1723 + (477 << 16),  1733 + (716 << 16),
    1741 + (1075 << 16), 1747 + (682 << 16),  1753 + (1245 << 16), 1759 + (401 << 16),  1777 + (774 << 16),
    1783 + (1026 << 16), 1787 + (499 << 16),  1789 + (1314 << 16), 1801 + (743 << 16),  1811 + (693 << 16),
    1823 + (1282 << 16), 1831 + (1003 << 16), 1847 + (1181 << 16), 1861 + (1079 << 16), 1867 + (765 << 16),
    1871 + (815 << 16),  1873 + (1350 << 16), 1877 + (1144 << 16), 1879 + (1449 << 16), 1889 + (718 << 16),
    1901 + (805 << 16),  1907 + (1203 << 16), 1913 + (1173 << 16), 1931 + (737 << 16),  1933 + (562 << 16),
    1949 + (579 << 16),  1951 + (701 << 16),  1973 + (1104 << 16), 1979 + (1105 << 16), 1987 + (1379 << 16),
    1993 + (827 << 16),  1997 + (1256 << 16), 1999 + (759 << 16),  2003 + (540 << 16),  2011 + (1284 << 16),
    2017 + (1188 << 16), 2027 + (776 << 16),  2029 + (853 << 16),  2039 + (1140 << 16), 2053 + (445 << 16),
    2063 + (1265 << 16), 2069 + (802 << 16),  2081 + (932 << 16),  2083 + (632 << 16),  2087 + (1504 << 16),
    2089 + (856 << 16),  2099 + (1229 << 16), 2111 + (1619 << 16), 2113 + (774 << 16),  2129 + (1229 << 16),
    2131 + (1300 << 16), 2137 + (1563 << 16), 2141 + (1551 << 16), 2143 + (1265 << 16), 2153 + (905 << 16),
    2161 + (1333 << 16), 2179 + (493 << 16),  2203 + (913 << 16),  2207 + (1397 << 16), 2213 + (1250 << 16),
    2221 + (612 << 16),  2237 + (1251 << 16), 2239 + (1765 << 16), 2243 + (1303 << 16), 2251 + (595 << 16),
    2267 + (981 << 16),  2269 + (671 << 16),  2273 + (1403 << 16), 2281 + (820 << 16),  2287 + (1404 << 16),
    2293 + (1661 << 16), 2297 + (973 << 16),  2309 + (1340 << 16), 2311 + (1015 << 16), 2333 + (1649 << 16),
    2339 + (855 << 16),  2341 + (1834 << 16), 2347 + (1621 << 16), 2351 + (1704 << 16), 2357 + (893 << 16),
    2371 + (1033 << 16), 2377 + (721 << 16),  2381 + (1737 << 16), 2383 + (1507 << 16), 2389 + (1851 << 16),
    2393 + (1006 << 16), 2399 + (994 << 16),  2411 + (923 << 16),  2417 + (872 << 16),  2423 + (1860 << 16),
};

// Faure-Lemieux scrambled radical inverse
float radinv_fl(uint n, uint dim)
{
  dim            = dim % PRIME_PERM_COUNT;
  const uint p_p = prime_permTN2[dim];

  uint base = get_prime(p_p);
  uint f    = (dim < PRIME_PERM_COUNT) ? get_permTN2(p_p) : 1;

  float       val     = 0.f;
  const float invBase = 1.f / float(base);  // use __uint2float_rn?
  float       invBi   = invBase;

  while(n > 0)
  {
    uint d_i = (f * (n % base)) % base;
    val += float(d_i) * invBi;  // use __uint2float_rn?
    n /= base;
    invBi *= invBase;
  }
  return val;
}

// Compute radical inverse of n to the base 2.
float radinv2(uint n)
{
  return uintBitsToFloat(0x3F800000 | ((bitfieldReverse(n)) >> 9)) - 1.0f;
}


//-------------------------------------------------------------------------------------------------
// Avoiding self intersections (see Ray Tracing Gems, Ch. 6)
//-------------------------------------------------------------------------------------------------

vec3 offsetRay(in vec3 p, in vec3 n)
{
  const float intScale   = 256.0f;
  const float floatScale = 1.0f / 65536.0f;
  const float origin     = 1.0f / 32.0f;

  ivec3 of_i = ivec3(intScale * n.x, intScale * n.y, intScale * n.z);

  vec3 p_i = vec3(intBitsToFloat(floatBitsToInt(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                  intBitsToFloat(floatBitsToInt(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                  intBitsToFloat(floatBitsToInt(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

  return vec3(abs(p.x) < origin ? p.x + floatScale * n.x : p_i.x,  //
              abs(p.y) < origin ? p.y + floatScale * n.y : p_i.y,  //
              abs(p.z) < origin ? p.z + floatScale * n.z : p_i.z);
}


//-------------------------------------------------------------------------------------------------
// Environment
//-------------------------------------------------------------------------------------------------

struct Environment_sample_data
{
  uint  alias;
  float q;
  float pdf;
};

Environment_sample_data getSampleData(sampler2D sample_buffer, ivec2 idx)
{
  vec3 data = texelFetch(sample_buffer, idx, 0).xyz;

  Environment_sample_data sample_data;
  sample_data.alias = floatBitsToInt(data.x);
  sample_data.q     = data.y;
  sample_data.pdf   = data.z;
  return sample_data;
}

Environment_sample_data getSampleData(sampler2D sample_buffer, uint idx)
{
  uvec2 size = textureSize(sample_buffer, 0);
  uint  px   = idx % size.x;
  uint  py   = idx / size.x;
  return getSampleData(sample_buffer, ivec2(px, size.y - py - 1));  // Image is upside down
}


vec3 environment_sample(sampler2D lat_long_tex, sampler2D sample_buffer, inout uint seed, out vec3 to_light, out float pdf)
{
  const float environment_intensity_factor = 1.0f;
  const float M_PI                         = 3.141592653589793;
  const float M_ONE_OVER_PI                = 0.318309886183790671538;

  vec3 xi;
  xi.x = rnd(seed);
  xi.y = rnd(seed);
  xi.z = rnd(seed);

  uvec2 tsize  = textureSize(lat_long_tex, 0);
  uint  width  = tsize.x;
  uint  height = tsize.y;

  const uint size = width * height;
  const uint idx  = min(uint(xi.x * float(size)), size - 1);

  Environment_sample_data sample_data = getSampleData(sample_buffer, idx);

  uint env_idx;
  if(xi.y < sample_data.q)
  {
    env_idx = idx;
    xi.y /= sample_data.q;
  }
  else
  {
    env_idx = sample_data.alias;
    xi.y    = (xi.y - sample_data.q) / (1.0f - sample_data.q);
  }

  uint       py = env_idx / width;
  const uint px = env_idx % width;
  pdf           = getSampleData(sample_buffer, env_idx).pdf;
  py            = height - py - 1;  // Image is upside down


  // uniformly sample spherical area of pixel
  const float u       = float(px + xi.y) / float(width);
  const float phi     = u * (2.0f * M_PI) - M_PI;
  float       sin_phi = sin(phi);
  float       cos_phi = cos(phi);

  const float step_theta = M_PI / float(height);
  const float theta0     = float(py) * step_theta;
  const float cos_theta  = cos(theta0) * (1.0f - xi.z) + cos(theta0 + step_theta) * xi.z;
  const float theta      = acos(cos_theta);
  const float sin_theta  = sin(theta);
  to_light               = vec3(cos_phi * sin_theta, -cos_theta, sin_phi * sin_theta);

  // lookup filtered value
  const float v = theta * M_ONE_OVER_PI;
  return environment_intensity_factor * texture(lat_long_tex, vec2(u, v)).xyz;
}


//-------------------------------------------------------------------------------------------------
// Environment Sampling / Convertion
//-------------------------------------------------------------------------------------------------

// Return the UV in a lat-long HDR map
vec2 get_spherical_uv(vec3 v)
{
  float gamma = asin(-v.y);
  float theta = atan(v.z, v.x);

  vec2 uv = vec2(theta * ONE_OVER_PI * 0.5, gamma * ONE_OVER_PI) + 0.5;
  return uv;
}


// Sampling hemisphere around +Z
void cosine_sample_hemisphere(const float u1, const float u2, out vec3 p)
{
  // Uniformly sample disk.
  const float r   = sqrt(u1);
  const float phi = 2.0f * M_PIf * u2;
  p.x             = r * cos(phi);
  p.y             = r * sin(phi);

  // Project up to hemisphere.
  p.z = sqrt(max(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}

// Sampling hemisphere around +Z within a cone (angle)
void cosineMaxSampleHemisphere(const float r1, const float r2, const float angle, out vec3 d)
{
  float cosAngle = cos(angle);
  float phi      = 2.0f * M_PIf * r1;
  float r        = sqrt(1.0 - ((1.0 - r2 * (1.0 - cosAngle)) * (1.0 - r2 * (1.0 - cosAngle))));
  d.x            = cos(phi) * r;
  d.y            = sin(phi) * r;
  d.z            = 1.0 - r2 * (1.0 - cosAngle);
}

// Return the tangent and binormal from the incoming normal
void computeOrthonormalBasis(in vec3 normal, out vec3 tangent, out vec3 binormal)
{

  if(abs(normal.x) > abs(normal.z))
  {
    binormal.x = -normal.y;
    binormal.y = normal.x;
    binormal.z = 0;
  }
  else
  {
    binormal.x = 0;
    binormal.y = -normal.z;
    binormal.z = normal.y;
  }

  binormal = normalize(binormal);
  tangent  = cross(binormal, normal);
}

// Transform P to the domaine made by N-T-B
void inverse_transform(inout vec3 p, in vec3 normal, in vec3 tangent, in vec3 binormal)
{
  p = p.x * tangent + p.y * binormal + p.z * normal;
}

// Return the basis from the incoming normal: x=tangent, y=binormal, z=normal
void compute_default_basis(const vec3 normal, out vec3 x, out vec3 y, out vec3 z)
{
  // ZAP's default coordinate system for compatibility
  z              = normal;
  const float yz = -z.y * z.z;
  y = normalize(((abs(z.z) > 0.99999f) ? vec3(-z.x * z.y, 1.0f - z.y * z.y, yz) : vec3(-z.x * z.z, yz, 1.0f - z.z * z.z)));
  x = cross(y, z);
}

// Randomly sampling around +Z
vec3 samplingHemisphere(inout uint seed, in vec3 x, in vec3 y, in vec3 z)
{
  float r1 = rnd(seed);
  float r2 = rnd(seed);
  float sq = sqrt(1.0 - r2);

  vec3 direction = vec3(cos(2 * M_PIf * r1) * sq, sin(2 * M_PIf * r1) * sq, sqrt(r2));
  direction      = direction.x * x + direction.y * y + direction.z * z;
  seed++;

  return direction;
}
