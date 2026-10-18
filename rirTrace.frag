// rirTrace.frag  (GLSL ES 3.00 body; Three.js prepends #version + precision via glslVersion: THREE.GLSL3)
//
// One fragment == one ray. Each pass advances every ray by ONE bounce.
// Geometry comes from two float textures (one texel == one surfel / disc):
//   uTex1 = [ p.x, p.y, p.z, n.x ]
//   uTex2 = [ n.y, n.z, 0,   0   ]
//
// Ray state (ping-pong, 3 render targets / MRT outputs):
//   outState0 = [ origin.xyz, pathLength ]
//   outState1 = [ dir.xyz,    bounceCount ]          (bounceCount < 0 -> ray is dead)
//   outHit    = [ u, v, pathLengthAtListener, bounceCount ]
//               (u,v) = direction of ARRIVAL in the listener (camera) frame, equirectangular.
//               bounceCount < 0 -> no listener hit on this segment.
//
// Equirect convention (listener frame = camera frame: +X right, +Y up, -Z forward):
//   u = atan(x, -z) / 2pi + 0.5      -> u = 0.5 is straight ahead, 0.75 is to the right
//   v = asin(y) / pi + 0.5           -> v = 1 is straight up
//   Inverse: lon = (u-0.5)*2pi, lat = (v-0.5)*pi,
//            dir  = ( cos(lat) sin(lon),  sin(lat), -cos(lat) cos(lon) )
//   "dir" points from the listener TOWARDS the sound source / reflection the energy came from.

uniform sampler2D uTex1;
uniform sampler2D uTex2;
uniform sampler2D uState0;
uniform sampler2D uState1;

uniform int   uInit;            // 1 = emit new rays from the source, 0 = advance
uniform int   uSeed;            // changes every pass
uniform int   uRayTexSize;      // width of the ray-state textures (= one ray per texel)
uniform vec3  uSource;
uniform vec3  uListener;        // = camera world position (set by the JS driver)
uniform mat3  uWorldToListener; // = inverse of the camera's world rotation
uniform float uListenerRadius;  // detection sphere radius (m)
uniform float uSurfelRadius;    // disc radius, ~0.7 * sqrt(totalSurfaceArea / numSurfels)
uniform float uMaxPathLength;   // c * maxTime (m)
uniform float uScattering;      // 0 = purely specular, 1 = purely diffuse

layout(location = 0) out vec4 outState0;
layout(location = 1) out vec4 outState1;
layout(location = 2) out vec4 outHit;

const float PI  = 3.14159265358979;
// Self-hit guard: origin is pushed OFFSET along the normal (so the surface just left is behind the ray),
// hence the minimum hit distance can be tiny. A large T_MIN lets rays leak near room edges.
// (Keep OFFSET >> float32 ulp of your coordinates: ~1e-6 at 5 m, ~4e-6 at 30 m.)
const float OFFSET = 1e-4;
const float T_MIN  = 1e-5;

// ---------------------------------------------------------------- RNG (PCG)
uint pcg(uint v) {
    uint s = v * 747796405u + 2891336837u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
float rnd(inout uint st) {
    st = pcg(st);
    return float(st) * (1.0 / 4294967296.0);
}
vec3 randomUnitVector(inout uint st) {
    float z   = 1.0 - 2.0 * rnd(st);
    float phi = 2.0 * PI * rnd(st);
    float r   = sqrt(max(0.0, 1.0 - z * z));
    return vec3(r * cos(phi), r * sin(phi), z);
}

// world-space ray travel direction -> equirect uv of the arrival direction (listener frame)
vec2 arrivalUV(vec3 travelDirWorld) {
    vec3 dl = uWorldToListener * (-travelDirWorld);
    float u = (abs(dl.x) + abs(dl.z) < 1e-6) ? 0.5 : atan(dl.x, -dl.z) / (2.0 * PI) + 0.5;
    float v = asin(clamp(dl.y, -1.0, 1.0)) / PI + 0.5;
    return vec2(u, v);
}

// ---------------------------------------------------------------- main
void main() {
    outState0 = vec4(0.0);
    outState1 = vec4(0.0, 0.0, 0.0, -1.0);
    outHit    = vec4(0.0, 0.0, 0.0, -1.0);       // invalid by default

    ivec2 px = ivec2(gl_FragCoord.xy);
    // NOT textureSize(uState0): in the very first emit pass uState0 is a target that has never been rendered,
    // which three.js replaces by a 1x1 placeholder -> ids collapse to px.x + px.y (only 127 distinct directions)
    uint  id = uint(px.y * uRayTexSize + px.x);
    uint  rng = pcg(id ^ pcg(uint(uSeed) * 2654435761u + 12345u));

    // ------------------------------------------------ emit
    if (uInit == 1) {
        vec3 d = randomUnitVector(rng);          // omnidirectional source
        outState0 = vec4(uSource, 0.0);
        outState1 = vec4(d, 0.0);
        return;
    }

    // ------------------------------------------------ advance
    vec4 s0 = texelFetch(uState0, px, 0);
    vec4 s1 = texelFetch(uState1, px, 0);
    if (s1.w < 0.0) { outState0 = s0; outState1 = s1; return; }   // dead ray

    vec3  o  = s0.xyz;
    float L  = s0.w;
    vec3  d  = s1.xyz;
    float nb = s1.w;

    // nearest surfel (ray vs. oriented disc, brute force)
    ivec2 gs = textureSize(uTex1, 0);
    int   N  = gs.x * gs.y;
    float r2 = uSurfelRadius * uSurfelRadius;

    float tHit = 1e30;
    vec3  nHit = vec3(0.0, 1.0, 0.0);

    for (int i = 0; i < N; i++) {
        ivec2 tc = ivec2(i % gs.x, i / gs.x);
        vec4 a = texelFetch(uTex1, tc, 0);
        vec2 b = texelFetch(uTex2, tc, 0).xy;
        vec3 p = a.xyz;
        vec3 n = vec3(a.w, b.x, b.y);

        float dn = dot(d, n);
        if (abs(dn) < 1e-5) continue;
        float t = dot(p - o, n) / dn;
        if (t < T_MIN || t >= tHit) continue;
        vec3 q = o + d * t - p;
        if (dot(q, q) > r2) continue;
        tHit = t;
        nHit = n;
    }

    if (tHit > 1e29) {                       // escaped through a gap -> kill ray
        outState0 = s0;
        outState1 = vec4(d, -1.0);
        return;
    }

    // listener detection on the segment [0, tHit]
    {
        vec3  oc   = o - uListener;
        float b    = dot(oc, d);
        float c    = dot(oc, oc) - uListenerRadius * uListenerRadius;
        float disc = b * b - c;
        if (disc > 0.0) {
            float sq = sqrt(disc);
            float t0 = -b - sq;
            float t1 = -b + sq;
            if (t1 > 0.0 && t0 < tHit) {
                float tc = clamp(-b, 0.0, tHit);        // closest approach -> arrival time
                outHit = vec4(arrivalUV(d), L + tc, nb);
            }
        }
    }

    // bounce
    float newL = L + tHit;
    vec3  P    = o + d * tHit;
    vec3  n    = (dot(nHit, d) > 0.0) ? -nHit : nHit;     // two-sided surfels

    vec3 dNew;
    if (rnd(rng) < uScattering) {
        dNew = normalize(n + randomUnitVector(rng));      // Lambert (cosine-weighted)
    } else {
        dNew = reflect(d, n);                             // specular
    }

    bool dead = newL > uMaxPathLength;
    outState0 = vec4(P + n * OFFSET, newL);
    outState1 = vec4(dNew, dead ? -1.0 : nb + 1.0);
}
