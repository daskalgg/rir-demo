uniform sampler2D uBVHNodes;
uniform sampler2D uBVHTris;
uniform sampler2D uState0;
uniform sampler2D uState1;

uniform int   uInit;
uniform int   uSeed;
uniform int   uRayTexSize;
uniform vec3  uSource;
uniform vec3  uListener;
uniform mat3  uWorldToListener;
uniform float uListenerRadius;
uniform float uMaxPathLength;
uniform float uScattering;

layout(location = 0) out vec4 outState0;
layout(location = 1) out vec4 outState1;
layout(location = 2) out vec4 outHit;

const float PI  = 3.14159265358979;
const float OFFSET = 1e-4;
const float T_MIN  = 1e-5;
const float BARY_EPS = 1e-6;

const int BVH_TEX_W = 2048;
const int STACK = 48;
const int MAX_STEPS = 4096;

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

vec2 arrivalUV(vec3 travelDirWorld) {
    vec3 dl = uWorldToListener * (-travelDirWorld);
    float u = (abs(dl.x) + abs(dl.z) < 1e-6) ? 0.5 : atan(dl.x, -dl.z) / (2.0 * PI) + 0.5;
    float v = asin(clamp(dl.y, -1.0, 1.0)) / PI + 0.5;
    return vec2(u, v);
}

vec4 bvhTexel(sampler2D tex, int i) { return texelFetch(tex, ivec2(i % BVH_TEX_W, i / BVH_TEX_W), 0); }

float boxEntry(vec3 lo, vec3 hi, vec3 o, vec3 invD) {
    vec3 t0 = (lo - o) * invD, t1 = (hi - o) * invD;
    vec3 tn = min(t0, t1), tf = max(t0, t1);
    float tNear = max(max(tn.x, tn.y), tn.z), tFar = min(min(tf.x, tf.y), tf.z);
    return (tNear <= tFar && tFar > 0.0) ? tNear : 1e30;
}

float traceBVH(vec3 o, vec3 d, out vec3 nHit) {
    vec3  s    = vec3(greaterThanEqual(d, vec3(0.0))) * 2.0 - 1.0;
    vec3  invD = s / max(abs(d), vec3(1e-20));
    float tHit = 1e30;
    nHit = vec3(0.0, 1.0, 0.0);

    vec3 stack[STACK];
    int  sp = 0;
    {
        vec4 lo = bvhTexel(uBVHNodes, 0), hi = bvhTexel(uBVHNodes, 1);
        float t = boxEntry(lo.xyz, hi.xyz, o, invD);
        if (t < tHit) stack[sp++] = vec3(lo.w, hi.w, t);
    }
    for (int step = 0; step < MAX_STEPS && sp > 0; step++) {
        vec3 e = stack[--sp];
        if (e.z >= tHit) continue;
        int a = int(e.x), count = int(e.y);
        if (count > 0) {
            for (int k = 0; k < count; k++) {
                int  i  = 3 * (a + k);
                vec4 t0 = bvhTexel(uBVHTris, i), t1 = bvhTexel(uBVHTris, i + 1), t2 = bvhTexel(uBVHTris, i + 2);
                vec3 e1 = t1.xyz, e2 = t2.xyz;
                vec3 pv = cross(d, e2);
                float det = dot(e1, pv);
                if (abs(det) < 1e-12) continue;
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
        } else {
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

void main() {
    outState0 = vec4(0.0);
    outState1 = vec4(0.0, 0.0, 0.0, -1.0);
    outHit    = vec4(0.0, 0.0, 0.0, -1.0);

    ivec2 px = ivec2(gl_FragCoord.xy);
    uint  id = uint(px.y * uRayTexSize + px.x);
    uint  rng = pcg(id ^ pcg(uint(uSeed) * 2654435761u + 12345u));

    if (uInit == 1) {
        vec3 d = randomUnitVector(rng);
        outState0 = vec4(uSource, 0.0);
        outState1 = vec4(d, 0.0);
        return;
    }

    vec4 s0 = texelFetch(uState0, px, 0);
    vec4 s1 = texelFetch(uState1, px, 0);
    if (s1.w < 0.0) { outState0 = s0; outState1 = s1; return; }

    vec3  o  = s0.xyz;
    float L  = s0.w;
    vec3  d  = s1.xyz;
    float nb = s1.w;

    vec3  nHit;
    float tHit = traceBVH(o, d, nHit);

    if (tHit > 1e29) {
        outState0 = s0;
        outState1 = vec4(d, -1.0);
        return;
    }

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
                float tc = clamp(-b, 0.0, tHit);
                outHit = vec4(arrivalUV(d), L + tc, nb);
            }
        }
    }

    float newL = L + tHit;
    vec3  P    = o + d * tHit;
    vec3  n    = (dot(nHit, d) > 0.0) ? -nHit : nHit;

    vec3 dNew;
    if (rnd(rng) < uScattering) {
        dNew = normalize(n + randomUnitVector(rng));
    } else {
        dNew = reflect(d, n);
    }

    bool dead = newL > uMaxPathLength;
    outState0 = vec4(P + n * OFFSET, newL);
    outState1 = vec4(dNew, dead ? -1.0 : nb + 1.0);
}
