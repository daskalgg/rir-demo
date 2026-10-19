// rirTrace.frag  (GLSL ES 3.00 body; Three.js prepends #version + precision via glslVersion: THREE.GLSL3)
//
// One fragment == one ray. Each pass advances every ray by ONE bounce.
// Geometry: the room's triangles under a BVH (built on the CPU by makeBVHTextures in rirTracer.js), in two float
// textures BVH_TEX_W texels wide; texel i sits at (i % BVH_TEX_W, i / BVH_TEX_W).
//   uBVHNodes, 2 texels per node:  [ min.xyz, a ]  [ max.xyz, count ]
//              leaf (count > 0): triangles a .. a + count - 1;  inner (count == 0): children a and a + 1
//   uBVHTris,  3 texels per triangle:  [ v0.xyz, n.x ]  [ v1 - v0, n.y ]  [ v2 - v0, n.z ]   (n = unit face normal)
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

uniform sampler2D uBVHNodes;
uniform sampler2D uBVHTris;
uniform sampler2D uState0;
uniform sampler2D uState1;

uniform int   uInit;            // 1 = emit new rays from the source, 0 = advance
uniform int   uSeed;            // changes every pass
uniform int   uRayTexSize;      // width of the ray-state textures (= one ray per texel)
uniform vec3  uSource;
uniform vec3  uListener;        // = camera world position (set by the JS driver)
uniform mat3  uWorldToListener; // = inverse of the camera's world rotation
uniform float uListenerRadius;  // detection sphere radius (m)
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
const float BARY_EPS = 1e-6;   // barycentric slack, so rays can't slip through the shared edge of two triangles

const int BVH_TEX_W = 2048;    // = BVH_TEX_W in rirTracer.js
const int STACK = 48;          // > the builder's MAX_DEPTH
const int MAX_STEPS = 4096;    // traversal guard (GPU watchdog), never reached on a sane BVH

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

// ---------------------------------------------------------------- BVH
vec4 bvhTexel(sampler2D tex, int i) { return texelFetch(tex, ivec2(i % BVH_TEX_W, i / BVH_TEX_W), 0); }

// entry distance of the ray into box [lo, hi] (<= 0 if it starts inside), or 1e30 if it misses / is behind
float boxEntry(vec3 lo, vec3 hi, vec3 o, vec3 invD) {
    vec3 t0 = (lo - o) * invD, t1 = (hi - o) * invD;
    vec3 tn = min(t0, t1), tf = max(t0, t1);
    float tNear = max(max(tn.x, tn.y), tn.z), tFar = min(min(tf.x, tf.y), tf.z);
    return (tNear <= tFar && tFar > 0.0) ? tNear : 1e30;
}

// nearest triangle hit (t > T_MIN) along o + t d; returns t (1e30: none) and its face normal in nHit
float traceBVH(vec3 o, vec3 d, out vec3 nHit) {
    vec3  s    = vec3(greaterThanEqual(d, vec3(0.0))) * 2.0 - 1.0;
    vec3  invD = s / max(abs(d), vec3(1e-20));           // no division by zero on axis-aligned rays
    float tHit = 1e30;
    nHit = vec3(0.0, 1.0, 0.0);

    // stack of pending nodes as (a, count, entry distance): a node's own texels hold what's needed to visit it
    vec3 stack[STACK];
    int  sp = 0;
    {
        vec4 lo = bvhTexel(uBVHNodes, 0), hi = bvhTexel(uBVHNodes, 1);
        float t = boxEntry(lo.xyz, hi.xyz, o, invD);
        if (t < tHit) stack[sp++] = vec3(lo.w, hi.w, t);
    }
    for (int step = 0; step < MAX_STEPS && sp > 0; step++) {
        vec3 e = stack[--sp];
        if (e.z >= tHit) continue;                         // a closer hit was found since this was pushed
        int a = int(e.x), count = int(e.y);
        if (count > 0) {                                   // leaf: Moller-Trumbore against each triangle
            for (int k = 0; k < count; k++) {
                int  i  = 3 * (a + k);
                vec4 t0 = bvhTexel(uBVHTris, i), t1 = bvhTexel(uBVHTris, i + 1), t2 = bvhTexel(uBVHTris, i + 2);
                vec3 e1 = t1.xyz, e2 = t2.xyz;
                vec3 pv = cross(d, e2);
                float det = dot(e1, pv);
                if (abs(det) < 1e-12) continue;            // ray parallel to the triangle
                float inv = 1.0 / det;
                vec3 tv = o - t0.xyz;
                float u = dot(tv, pv) * inv;
                if (u < -BARY_EPS || u > 1.0 + BARY_EPS) continue;
                vec3 qv = cross(tv, e1);
                float v = dot(d, qv) * inv;
                if (v < -BARY_EPS || u + v > 1.0 + BARY_EPS) continue;
                float t = dot(e2, qv) * inv;
                if (t > T_MIN && t < tHit) { tHit = t; nHit = vec3(t0.w, t1.w, t2.w); }
            }
        } else {                                           // inner: push the children it hits, nearer one on top
            vec4 lo0 = bvhTexel(uBVHNodes, 2 * a),     hi0 = bvhTexel(uBVHNodes, 2 * a + 1);
            vec4 lo1 = bvhTexel(uBVHNodes, 2 * a + 2), hi1 = bvhTexel(uBVHNodes, 2 * a + 3);
            vec3 c0 = vec3(lo0.w, hi0.w, boxEntry(lo0.xyz, hi0.xyz, o, invD));
            vec3 c1 = vec3(lo1.w, hi1.w, boxEntry(lo1.xyz, hi1.xyz, o, invD));
            if (c0.z > c1.z) { vec3 tmp = c0; c0 = c1; c1 = tmp; }
            if (c1.z < tHit && sp < STACK) stack[sp++] = c1;
            if (c0.z < tHit && sp < STACK) stack[sp++] = c0;
        }
    }
    return tHit;
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

    // nearest surface along the ray
    vec3  nHit;
    float tHit = traceBVH(o, d, nHit);

    if (tHit > 1e29) {                       // left the room (open geometry) -> kill ray
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
    vec3  n    = (dot(nHit, d) > 0.0) ? -nHit : nHit;     // two-sided surfaces

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
